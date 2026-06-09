<script lang="ts">
  // Settings → Notes search: pick how the Library searches (full-text / semantic / hybrid) and which
  // embedding model powers semantic + hybrid. Local models download explicitly (a Download button,
  // never on a stray click) on a worker thread so the UI never freezes, and can be deleted. Cloud
  // models run through the provider's API with the user's own key. Mirrors the LIVE / FILE pickers.
  import { onMount } from "svelte";
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

  // Explicit download, then activate — mirrors the ASR picker's download → select.
  async function download(id: string) {
    if (busy) return;
    busy = id;
    error = "";
    try {
      await invoke("download_embedding_model", { id });
      await invoke("set_embedding_model", { id });
      await refresh();
    } catch (e) {
      error = String(e);
    }
    busy = "";
  }

  // Activate a ready model (local installed, or cloud keyed), or turn embedding off (null).
  async function activate(id: string | null) {
    if (busy) return;
    busy = id ?? "off";
    error = "";
    try {
      await invoke("set_embedding_model", { id });
      await refresh();
    } catch (e) {
      error = String(e);
    }
    busy = "";
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

{#snippet row(m: EmbModel)}
  <li class="emb-row">
    <button
      class="emb-main"
      class:on={m.active}
      disabled={!!busy || !m.ready}
      onclick={() => activate(m.id)}
    >
      <span class="emb-name">
        {m.label}
        {#if m.active}
          <span class="badge on">{i18n.t.settings.embedActive}</span>
        {:else if m.kind === "local" && m.installed}
          <span class="badge">{i18n.t.settings.embedInstalled}</span>
        {/if}
      </span>
      <span class="emb-meta">
        {m.dim}d · {m.kind === "cloud" ? i18n.t.settings.embedApi : sizeLabel(m.sizeMb)}
      </span>
    </button>

    <div class="emb-actions">
      {#if busy === m.id}
        {#if m.kind === "local" && !m.installed}
          <span class="dl-busy"><span class="spin"></span>{i18n.t.settings.embedDownloading}</span>
        {:else}
          <span class="spin" aria-label={i18n.t.settings.embedWorking}></span>
        {/if}
      {:else if m.kind === "local" && !m.installed}
        <button class="dl" disabled={!!busy} onclick={() => download(m.id)}>
          ↓ {i18n.t.settings.embedDownload} · {sizeLabel(m.sizeMb)}
        </button>
      {:else if m.kind === "local"}
        <button
          class="trash"
          disabled={!!busy}
          aria-label={i18n.t.live.deleteModel.trashAria(m.label, sizeLabel(m.sizeMb))}
          title={i18n.t.live.deleteModel.trashTitle(sizeLabel(m.sizeMb))}
          onclick={() => {
            pendingDelete = m;
            deleteOpen = true;
          }}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12" />
          </svg>
        </button>
      {:else if !m.ready}
        <span class="needkey">{i18n.t.settings.embedNeedsKey}</span>
      {/if}
    </div>
  </li>
{/snippet}

<p class="set-intro">{i18n.t.settings.searchIntro}</p>

<div class="row">
  <span class="label">{i18n.t.settings.searchMode}</span>
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

<div class="emb-head">{i18n.t.settings.embedModel}</div>

<ul class="emb-list">
  <li class="emb-row">
    <button class="emb-main" class:on={!active} disabled={!!busy} onclick={() => activate(null)}>
      <span class="emb-name">{i18n.t.settings.embedOff}</span>
      <span class="emb-meta">{i18n.t.settings.embedOffHint}</span>
    </button>
  </li>

  <li class="sec">{i18n.t.settings.embedLocal}</li>
  {#each local as m (m.id)}{@render row(m)}{/each}

  <li class="sec">{i18n.t.settings.embedCloud}</li>
  {#each unkeyed as p (p.provider)}
    <li class="keybox">
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
    </li>
  {/each}
  {#each cloud as m (m.id)}{@render row(m)}{/each}
</ul>

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

  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .label {
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

  .emb-head {
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
    margin-top: 4px;
  }
  .emb-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .sec {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 6px 2px 0;
  }

  .emb-row {
    position: relative;
    display: flex;
    align-items: stretch;
    gap: 8px;
  }
  .emb-main {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
    text-align: left;
    padding: 11px 13px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 11px;
    cursor: pointer;
    font: inherit;
    color: var(--text);
    transition:
      border-color 0.15s,
      background 0.15s;
  }
  .emb-main:hover:not(:disabled) {
    border-color: var(--border-strong);
    background: var(--surface-active);
  }
  .emb-main:disabled {
    cursor: default;
  }
  .emb-main.on {
    border-color: var(--accent);
    background: var(--surface-active);
  }
  .emb-name {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 13.5px;
    font-weight: 600;
  }
  .emb-meta {
    font-size: 12px;
    color: var(--muted);
  }

  .badge {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--muted);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 1px 6px;
  }
  .badge.on {
    color: var(--accent);
    border-color: var(--accent);
  }

  .emb-actions {
    flex: none;
    display: flex;
    align-items: center;
  }
  .dl {
    font-family: inherit;
    font-size: 12px;
    font-weight: 500;
    white-space: nowrap;
    color: var(--accent);
    background: var(--bg);
    border: 1px solid var(--accent);
    border-radius: 9px;
    padding: 0 12px;
    height: 100%;
    cursor: pointer;
    transition:
      background 0.15s,
      opacity 0.15s;
  }
  .dl:hover:not(:disabled) {
    background: var(--surface-active);
  }
  .dl:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .trash {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 100%;
    color: var(--muted);
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 9px;
    cursor: pointer;
    transition:
      color 0.15s,
      border-color 0.15s,
      background 0.15s;
  }
  .trash:hover:not(:disabled) {
    color: var(--stop);
    border-color: var(--stop);
    background: var(--surface-active);
  }
  .trash:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .needkey {
    font-size: 11.5px;
    color: var(--muted);
    padding-right: 4px;
  }

  .dl-busy {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 0 10px;
    font-size: 12px;
    color: var(--muted);
  }
  .spin {
    display: inline-block;
    width: 13px;
    height: 13px;
    border: 2px solid var(--border-strong);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

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
