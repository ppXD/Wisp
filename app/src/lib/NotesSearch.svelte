<script lang="ts">
  // Settings → Notes search: pick how the Library searches (full-text / semantic / hybrid) and which
  // embedding model powers semantic + hybrid. The model picker is a dropdown (trigger → drop-open
  // list, like the LIVE picker) so the list stays compact and aligned. Local models download
  // explicitly on a worker thread (never on a stray click) so the UI never freezes, and can be
  // deleted; cloud models run through the provider's API with the user's own key.
  import { onMount } from "svelte";
  import { slide } from "svelte/transition";
  import { invoke } from "@tauri-apps/api/core";
  import { i18n } from "$lib/i18n.svelte";
  import Modal from "$lib/Modal.svelte";

  type EmbModel = {
    id: string;
    label: string;
    dim: number;
    sizeMb: number;
    kind: "local" | "cloud";
    installed: boolean;
    ready: boolean;
    provider: string | null;
    active: boolean;
  };

  let mode = $state("fulltext");
  let models = $state<EmbModel[]>([]);
  let open = $state(false); // dropdown open
  let busy = $state(""); // id with an in-flight download/activate ("off" while clearing)
  let error = $state("");

  let deleteOpen = $state(false);
  let pendingDelete = $state<EmbModel | null>(null);
  let deleting = $state("");

  // Cloud API-key entry drafts, keyed by provider id.
  let keyDraft = $state<Record<string, string>>({});
  let savingKey = $state("");

  const active = $derived(models.find((m) => m.active) ?? null);
  const local = $derived(models.filter((m) => m.kind === "local"));
  const cloud = $derived(models.filter((m) => m.kind === "cloud"));
  // Distinct cloud providers whose key isn't saved yet, with a display name from the model label.
  const unkeyed = $derived(
    [
      ...new Map(
        cloud
          .filter((m) => !m.ready && m.provider)
          .map((m) => [m.provider as string, m.label.split("·")[0].trim()]),
      ),
    ].map(([provider, name]) => ({ provider, name })),
  );

  async function refresh() {
    models = await invoke<EmbModel[]>("list_embedding_models");
  }

  async function load() {
    try {
      const [m, list] = await Promise.all([
        invoke<string>("search_mode"),
        invoke<EmbModel[]>("list_embedding_models"),
      ]);
      mode = m;
      models = list;
    } catch (e) {
      error = String(e);
    }
  }
  onMount(load);

  async function setMode(m: string) {
    const prev = mode;
    mode = m;
    try {
      await invoke("set_search_mode", { mode: m });
    } catch (e) {
      mode = prev;
      error = String(e);
    }
  }

  // Explicit download, then activate — mirrors the ASR picker's download → select. Keeps the menu
  // open so its row spinner is visible during the (possibly long) download.
  async function download(id: string) {
    if (busy) return;
    busy = id;
    error = "";
    try {
      await invoke("download_embedding_model", { id });
      await invoke("set_embedding_model", { id });
      await refresh();
      open = false;
    } catch (e) {
      error = String(e);
    }
    busy = "";
  }

  // Activate a ready model (local installed, or cloud keyed), or turn embedding off (null), then
  // collapse the dropdown.
  async function activate(id: string | null) {
    if (busy) return;
    busy = id ?? "off";
    error = "";
    try {
      await invoke("set_embedding_model", { id });
      await refresh();
      open = false;
    } catch (e) {
      error = String(e);
    }
    busy = "";
  }

  // Route a click on a model row: a not-yet-downloaded local model downloads first; everything else
  // (installed local, keyed cloud) activates directly.
  function pick(m: EmbModel) {
    if (m.kind === "local" && !m.installed) download(m.id);
    else activate(m.id);
  }

  async function saveKey(provider: string) {
    const key = (keyDraft[provider] ?? "").trim();
    if (!key || savingKey) return;
    savingKey = provider;
    error = "";
    try {
      await invoke("set_cloud_key", { provider, key });
      keyDraft[provider] = "";
      await refresh();
    } catch (e) {
      error = String(e);
    }
    savingKey = "";
  }

  async function confirmDelete() {
    if (!pendingDelete) return;
    deleting = pendingDelete.id;
    error = "";
    try {
      await invoke("delete_embedding_model", { id: pendingDelete.id });
      await refresh();
    } catch (e) {
      error = String(e);
    }
    deleting = "";
    deleteOpen = false;
  }

  $effect(() => {
    if (!deleteOpen) pendingDelete = null;
  });

  function sizeLabel(mb: number): string {
    return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`;
  }
</script>

<svelte:window
  onkeydown={(e) => {
    if (e.key === "Escape") open = false;
  }}
/>

<p class="set-intro">{i18n.t.settings.searchIntro}</p>

<div class="field">
  <span class="field-label">{i18n.t.settings.searchMode}</span>
  <div class="seg">
    <button class="seg-btn" class:on={mode === "fulltext"} onclick={() => setMode("fulltext")}>
      {i18n.t.settings.modeFulltext}
    </button>
    <button class="seg-btn" class:on={mode === "semantic"} onclick={() => setMode("semantic")}>
      {i18n.t.settings.modeSemantic}
    </button>
    <button class="seg-btn" class:on={mode === "hybrid"} onclick={() => setMode("hybrid")}>
      {i18n.t.settings.modeHybrid}
    </button>
  </div>
</div>

<div class="field col">
  <span class="field-label">{i18n.t.settings.embedModel}</span>

  <div class="picker">
    <button class="trigger" class:open onclick={() => (open = !open)}>
      {#if busy}<span class="spin sm"></span>{/if}
      <span class="trigger-label">{active ? active.label : i18n.t.settings.embedOff}</span>
      <span class="caret"></span>
    </button>

    {#if open}
      <div class="menu" transition:slide={{ duration: 140 }}>
        <button class="opt" class:sel={!active} disabled={!!busy} onclick={() => activate(null)}>
          <span class="opt-name">{i18n.t.settings.embedOff}</span>
          <span class="opt-note">{i18n.t.settings.embedOffHint}</span>
        </button>

        <div class="sec">{i18n.t.settings.embedLocal}</div>
        {#each local as m (m.id)}
          <div class="opt-row">
            <button class="opt" class:sel={m.active} disabled={!!busy} onclick={() => pick(m)}>
              <span class="opt-name">{m.label}</span>
              {#if busy === m.id}
                <span class="spin"></span>
              {:else if m.active}
                <span class="tag on">{i18n.t.settings.embedActive}</span>
              {:else if m.installed}
                <span class="tag">{i18n.t.settings.embedInstalled}</span>
              {:else}
                <span class="opt-size">↓ {sizeLabel(m.sizeMb)}</span>
              {/if}
            </button>
            {#if m.installed && !busy}
              <button
                class="del"
                title={i18n.t.live.deleteModel.trashTitle(sizeLabel(m.sizeMb))}
                aria-label={i18n.t.live.deleteModel.trashAria(m.label, sizeLabel(m.sizeMb))}
                onclick={() => {
                  pendingDelete = m;
                  deleteOpen = true;
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12" />
                </svg>
              </button>
            {/if}
          </div>
        {/each}

        <div class="sec">{i18n.t.settings.embedCloud}</div>
        {#each cloud as m (m.id)}
          <button
            class="opt"
            class:sel={m.active}
            disabled={!!busy || !m.ready}
            onclick={() => pick(m)}
          >
            <span class="opt-name">{m.label}</span>
            {#if busy === m.id}
              <span class="spin"></span>
            {:else if m.active}
              <span class="tag on">{i18n.t.settings.embedActive}</span>
            {:else if !m.ready}
              <span class="opt-note">{i18n.t.settings.embedNeedsKey}</span>
            {:else}
              <span class="opt-size">{i18n.t.settings.embedApi}</span>
            {/if}
          </button>
        {/each}
      </div>
    {/if}
  </div>

  {#each unkeyed as p (p.provider)}
    <div class="keybox">
      <label class="keylabel" for={`k-${p.provider}`}>{i18n.t.settings.embedKeyLabel(p.name)}</label>
      <div class="keyrow">
        <input
          id={`k-${p.provider}`}
          class="keyinput"
          type="password"
          autocomplete="off"
          placeholder={i18n.t.settings.embedKeyPlaceholder}
          value={keyDraft[p.provider] ?? ""}
          oninput={(e) => (keyDraft[p.provider] = e.currentTarget.value)}
          onkeydown={(e) => {
            if (e.key === "Enter") saveKey(p.provider);
          }}
        />
        <button
          class="keysave"
          disabled={savingKey === p.provider || !(keyDraft[p.provider] ?? "").trim()}
          onclick={() => saveKey(p.provider)}
        >
          {savingKey === p.provider ? i18n.t.settings.embedWorking : i18n.t.settings.embedKeySave}
        </button>
      </div>
    </div>
  {/each}
</div>

{#if error}<p class="set-error">{error}</p>{/if}
<p class="set-note">{i18n.t.settings.embedCloudNote}</p>

<Modal bind:open={deleteOpen} title={i18n.t.live.deleteModel.title}>
  {#if pendingDelete}
    <p class="del-body">{i18n.t.live.deleteModel.body(pendingDelete.label, sizeLabel(pendingDelete.sizeMb))}</p>
    <p class="del-sub">{i18n.t.live.deleteModel.sub}</p>
    <div class="del-actions">
      <button class="btn ghost" disabled={!!deleting} onclick={() => (deleteOpen = false)}>
        {i18n.t.common.cancel}
      </button>
      <button class="btn danger" disabled={!!deleting} onclick={confirmDelete}>
        {deleting ? i18n.t.live.deleteModel.deleting : i18n.t.live.deleteModel.confirm}
      </button>
    </div>
  {/if}
</Modal>

<style>
  .set-intro,
  .set-note {
    margin: 0;
    font-size: 13px;
    line-height: 1.5;
    color: var(--muted);
  }
  .set-error {
    margin: 0;
    font-size: 12.5px;
    color: var(--stop);
  }

  .field {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .field.col {
    flex-direction: column;
    align-items: stretch;
    gap: 8px;
  }
  .field-label {
    font-size: 13px;
    color: var(--text);
  }

  .seg {
    display: inline-flex;
    padding: 2px;
    gap: 2px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 9px;
  }
  .seg-btn {
    font-family: inherit;
    font-size: 12px;
    color: var(--muted);
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 4px 11px;
    cursor: pointer;
    transition:
      background 0.12s,
      color 0.12s;
  }
  .seg-btn.on {
    background: var(--surface-active);
    color: var(--accent);
    font-weight: 500;
  }

  /* ── Model dropdown ── */
  .picker {
    position: relative;
  }
  .trigger {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: inherit;
    font-size: 13.5px;
    font-weight: 500;
    color: var(--text);
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: 9px;
    padding: 9px 12px;
    cursor: pointer;
    transition:
      border-color 0.15s,
      background 0.15s;
  }
  .trigger:hover,
  .trigger.open {
    border-color: var(--muted);
    background: var(--surface-active);
  }
  .trigger-label {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    text-align: left;
  }
  .caret {
    flex: none;
    width: 11px;
    height: 7px;
    background-color: var(--muted);
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='7' viewBox='0 0 11 7' fill='none' stroke='%23000' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M1.5 1.5L5.5 5.5L9.5 1.5'/%3E%3C/svg%3E")
      no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='7' viewBox='0 0 11 7' fill='none' stroke='%23000' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M1.5 1.5L5.5 5.5L9.5 1.5'/%3E%3C/svg%3E")
      no-repeat center;
    transition: transform 0.15s;
  }
  .trigger.open .caret {
    transform: rotate(180deg);
  }

  .menu {
    margin-top: 6px;
    display: flex;
    flex-direction: column;
    gap: 1px;
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    box-shadow: 0 14px 34px -10px rgba(40, 30, 20, 0.28);
    padding: 6px;
  }

  .sec {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 8px 10px 4px;
  }

  .opt-row {
    display: flex;
    align-items: center;
    gap: 2px;
  }
  .opt {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: inherit;
    font-size: 13.5px;
    color: var(--text);
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 8px 10px;
    cursor: pointer;
    text-align: left;
    transition: background 0.12s;
  }
  .opt:hover:not(:disabled) {
    background: var(--surface-active);
  }
  .opt:disabled {
    cursor: default;
  }
  .opt.sel {
    color: var(--accent);
    font-weight: 500;
  }
  .opt-name {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
    line-height: 1.35;
  }
  .opt-size {
    flex: none;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .opt-note {
    flex: none;
    font-size: 11.5px;
    color: var(--muted);
    white-space: nowrap;
  }
  .tag {
    flex: none;
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
  }
  .tag.on {
    color: var(--accent);
  }

  .del {
    flex: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    color: var(--muted);
    background: transparent;
    border: none;
    border-radius: 7px;
    cursor: pointer;
    transition:
      color 0.12s,
      background 0.12s;
  }
  .opt-row:hover .del {
    color: var(--text);
  }
  .del:hover {
    color: var(--stop);
    background: var(--surface-active);
  }

  .spin {
    flex: none;
    display: inline-block;
    width: 13px;
    height: 13px;
    border: 2px solid var(--border-strong);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  .spin.sm {
    width: 12px;
    height: 12px;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* ── Cloud key entry ── */
  .keybox {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 11px 13px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 11px;
  }
  .keylabel {
    font-size: 12px;
    font-weight: 500;
    color: var(--text);
  }
  .keyrow {
    display: flex;
    gap: 8px;
  }
  .keyinput {
    flex: 1;
    min-width: 0;
    font: inherit;
    font-size: 13px;
    color: var(--text);
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 7px 10px;
  }
  .keyinput:focus {
    outline: none;
    border-color: var(--accent);
  }
  .keysave {
    flex: none;
    font-family: inherit;
    font-size: 12.5px;
    font-weight: 500;
    color: #fff;
    background: var(--accent);
    border: none;
    border-radius: 8px;
    padding: 0 14px;
    cursor: pointer;
    transition:
      filter 0.15s,
      opacity 0.15s;
  }
  .keysave:hover:not(:disabled) {
    filter: brightness(1.05);
  }
  .keysave:disabled {
    opacity: 0.5;
    cursor: default;
  }

  /* ── Delete confirm ── */
  .del-body {
    margin: 0;
    font-size: 14px;
    line-height: 1.5;
    color: var(--text);
  }
  .del-sub {
    margin: 0;
    font-size: 12.5px;
    color: var(--muted);
  }
  .del-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 4px;
  }
  .btn {
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    border-radius: 9px;
    padding: 7px 14px;
    cursor: pointer;
    border: 1px solid transparent;
    transition:
      background 0.15s,
      opacity 0.15s;
  }
  .btn:disabled {
    opacity: 0.6;
    cursor: default;
  }
  .ghost {
    color: var(--text);
    background: transparent;
    border-color: var(--border-strong);
  }
  .ghost:hover:not(:disabled) {
    background: var(--surface-active);
  }
  .danger {
    color: #fff;
    background: var(--stop);
  }
  .danger:hover:not(:disabled) {
    filter: brightness(1.05);
  }
</style>
