<script lang="ts">
  // Settings → Notes search: pick the search mode (full-text / semantic / hybrid) and the embedding
  // model that powers semantic + hybrid. The picker is a dropdown holding a LIVE-style two-pane
  // layout: tabs (Device | Cloud) → left a provider/family list (+ Import) → right that provider's
  // models. Clicking a not-downloaded model selects it; an explicit Download button then runs the
  // download in the background (tracked in `embedDownload`, so it survives this dialog closing).
  import { onMount } from "svelte";
  import { slide } from "svelte/transition";
  import { invoke } from "@tauri-apps/api/core";
  import { i18n } from "$lib/i18n.svelte";
  import { dl, dlPct, runDownload } from "$lib/embedDownload.svelte";
  import Modal from "$lib/Modal.svelte";

  const IMPORT = "__import__";

  type EmbModel = {
    id: string;
    label: string;
    dim: number;
    sizeMb: number;
    kind: "local" | "cloud";
    group: string;
    installed: boolean;
    ready: boolean;
    provider: string | null;
    active: boolean;
  };

  let mode = $state("fulltext");
  let models = $state<EmbModel[]>([]);
  let open = $state(false);
  let tab = $state<"device" | "cloud">("device");
  let sel = $state(""); // selected provider/group, or IMPORT
  let chosen = $state<string | null>(null); // a not-yet-downloaded model the user picked (awaits Download)
  let busy = $state(""); // a short op (activate / off) — downloads live in `dl`
  let error = $state("");

  let deleteOpen = $state(false);
  let pendingDelete = $state<EmbModel | null>(null);
  let deleting = $state("");

  let keyDraft = $state<Record<string, string>>({});
  let savingKey = $state("");

  const anyBusy = $derived(!!busy || !!dl.id);
  const active = $derived(models.find((m) => m.active) ?? null);
  const chosenModel = $derived(models.find((m) => m.id === chosen) ?? null);
  const tabModels = $derived(
    models.filter((m) => (tab === "device" ? m.kind === "local" : m.kind === "cloud")),
  );
  const providers = $derived([...new Set(tabModels.map((m) => m.group))]);
  const selModels = $derived(tabModels.filter((m) => m.group === sel));
  const selProviderId = $derived(selModels.find((m) => m.provider)?.provider ?? null);
  const selUnkeyed = $derived(tab === "cloud" && selModels.length > 0 && selModels.some((m) => !m.ready));

  // Keep `sel` valid for the current tab (Import is device-only).
  $effect(() => {
    const valid = tab === "device" ? [...providers, IMPORT] : providers;
    if (open && !valid.includes(sel)) sel = providers[0] ?? "";
  });

  // Drop the pending Download selection when the menu closes or the provider/tab changes.
  $effect(() => {
    void open;
    void sel;
    chosen = null;
  });

  // Refresh the list when a download finishes (dl.id clears) — even if another mount started it.
  let prevDl: string | null = null;
  $effect(() => {
    const cur = dl.id;
    if (prevDl && !cur) refresh();
    prevDl = cur;
  });

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

  function toggle() {
    open = !open;
    if (open) {
      tab = active?.kind === "cloud" ? "cloud" : "device";
      sel = active?.group ?? "";
    }
  }

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

  // Fire-and-forget: the module-level download survives this component unmounting; the dl-effect
  // above refreshes the list when it completes.
  function download(id: string) {
    if (anyBusy) return;
    chosen = null;
    const m = models.find((x) => x.id === id);
    runDownload(id, m?.label ?? id);
  }

  async function activate(id: string | null) {
    if (anyBusy) return;
    chosen = null;
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

  // Clicking a model never downloads. A not-yet-downloaded local model just becomes the "chosen"
  // model (the Download button then appears); installed local / keyed cloud models activate on click.
  function pick(m: EmbModel) {
    if (m.kind === "local" && !m.installed) chosen = chosen === m.id ? null : m.id;
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

{#snippet modelOpt(m: EmbModel)}
  <div class="opt-row">
    <button class="opt" class:sel={m.active} class:chosen={chosen === m.id} disabled={anyBusy || (m.kind === "cloud" && !m.ready)} onclick={() => pick(m)}>
      <span class="opt-name">{m.label}</span>
      {#if busy === m.id}
        <span class="spin"></span>
      {:else if m.active}
        <span class="tag on">{i18n.t.settings.embedActive}</span>
      {:else if m.kind === "local" && m.installed}
        <span class="tag">{i18n.t.settings.embedInstalled}</span>
      {:else if m.kind === "local"}
        <span class="opt-size">{sizeLabel(m.sizeMb)}</span>
      {:else if !m.ready}
        <span class="opt-note">{i18n.t.settings.embedNeedsKey}</span>
      {:else}
        <span class="opt-size">{i18n.t.settings.embedApi}</span>
      {/if}
    </button>
    {#if m.kind === "local" && m.installed && !anyBusy}
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
{/snippet}

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
    <button class="trigger" class:open onclick={toggle}>
      {#if dl.id}
        <span class="spin sm"></span><span class="trig-pct">{dlPct()}%</span>
      {:else if busy}
        <span class="spin sm"></span>
      {/if}
      <span class="trigger-label">{active ? active.label : i18n.t.settings.embedOff}</span>
      <span class="caret"></span>
    </button>

    {#if open}
      <div class="menu" transition:slide={{ duration: 140 }}>
        {#if dl.id}
          <div class="dl-prog">
            <div class="dl-head"><span class="dl-name">↓ {dl.label}</span><span>{dlPct()}%</span></div>
            <div class="dl-bar"><div class="dl-fill" style="width:{dlPct()}%"></div></div>
            <span class="dl-bg">{i18n.t.settings.embedBgHint}</span>
          </div>
        {/if}

        <button class="opt off" class:sel={!active} disabled={anyBusy} onclick={() => activate(null)}>
          <span class="opt-name">{i18n.t.settings.embedOff}</span>
          <span class="opt-note">{i18n.t.settings.embedOffHint}</span>
        </button>

        <div class="tabs">
          <button class="tab" class:on={tab === "device"} onclick={() => (tab = "device")}>
            {i18n.t.settings.embedTabDevice}
          </button>
          <button class="tab" class:on={tab === "cloud"} onclick={() => (tab = "cloud")}>
            {i18n.t.settings.embedTabCloud}
          </button>
        </div>

        <div class="panes">
          <div class="cats">
            {#each providers as g (g)}
              <button class="cat" class:on={sel === g} onclick={() => (sel = g)}>{g}</button>
            {/each}
            {#if tab === "device"}
              <button class="cat import" class:on={sel === IMPORT} onclick={() => (sel = IMPORT)}>
                ＋ {i18n.t.settings.embedImport}
              </button>
            {/if}
          </div>

          <div class="detail">
            {#if sel === IMPORT}
              <p class="import-hint">{i18n.t.settings.embedImportHint}</p>
              <div class="keyrow">
                <input class="keyinput" type="text" placeholder="org/model-onnx" disabled />
                <button class="keysave" disabled>{i18n.t.settings.embedDownload}</button>
              </div>
              <p class="import-soon">{i18n.t.settings.embedImportSoon}</p>
            {:else}
              {#if selUnkeyed && selProviderId}
                {@const pid = selProviderId}
                <div class="keybox">
                  <label class="keylabel" for="emb-key">{i18n.t.settings.embedKeyLabel(sel)}</label>
                  <div class="keyrow">
                    <input
                      id="emb-key"
                      class="keyinput"
                      type="password"
                      autocomplete="off"
                      placeholder={i18n.t.settings.embedKeyPlaceholder}
                      value={keyDraft[pid] ?? ""}
                      oninput={(e) => (keyDraft[pid] = e.currentTarget.value)}
                      onkeydown={(e) => {
                        if (e.key === "Enter") saveKey(pid);
                      }}
                    />
                    <button
                      class="keysave"
                      disabled={savingKey === pid || !(keyDraft[pid] ?? "").trim()}
                      onclick={() => saveKey(pid)}
                    >
                      {savingKey === pid ? i18n.t.settings.embedWorking : i18n.t.settings.embedKeySave}
                    </button>
                  </div>
                </div>
              {/if}
              {#each selModels as m (m.id)}{@render modelOpt(m)}{/each}
            {/if}
          </div>
        </div>

        {#if tab === "cloud"}
          <p class="cloud-note">{i18n.t.settings.embedCloudNote}</p>
        {/if}

        {#if chosenModel && chosenModel.kind === "local" && !chosenModel.installed && !dl.id}
          {@const cm = chosenModel}
          <button class="dl-btn" disabled={anyBusy} onclick={() => download(cm.id)}>
            ↓ {i18n.t.settings.embedDownload} · {sizeLabel(cm.sizeMb)}
          </button>
        {/if}
      </div>
    {/if}
  </div>
</div>

{#if error || dl.error}<p class="set-error">{error || dl.error}</p>{/if}

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
  .set-intro {
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
  .trig-pct {
    flex: none;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--muted);
  }

  .menu {
    margin-top: 6px;
    display: flex;
    flex-direction: column;
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    box-shadow: 0 14px 34px -10px rgba(40, 30, 20, 0.28);
    padding: 6px;
  }

  /* ── In-flight download banner ── */
  .dl-prog {
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding: 8px 10px;
    margin-bottom: 4px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 9px;
  }
  .dl-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: 12.5px;
    font-weight: 500;
    color: var(--text);
  }
  .dl-name {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-bar {
    height: 6px;
    background: var(--surface-active);
    border-radius: 999px;
    overflow: hidden;
  }
  .dl-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 999px;
    transition: width 0.2s ease;
  }
  .dl-bg {
    font-size: 11px;
    color: var(--live);
  }

  .tabs {
    display: flex;
    gap: 2px;
    padding: 4px;
    margin: 2px 0 6px;
    background: var(--bg);
    border-radius: 9px;
  }
  .tab {
    flex: 1;
    font-family: inherit;
    font-size: 12.5px;
    font-weight: 500;
    color: var(--muted);
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 6px 0;
    cursor: pointer;
    transition:
      background 0.12s,
      color 0.12s;
  }
  .tab.on {
    background: var(--surface);
    color: var(--accent);
    box-shadow: 0 1px 2px rgba(40, 30, 20, 0.1);
  }

  .panes {
    display: flex;
    gap: 6px;
    min-height: 120px;
  }
  .cats {
    flex: none;
    width: 132px;
    display: flex;
    flex-direction: column;
    gap: 1px;
    border-right: 1px solid var(--border);
    padding-right: 6px;
  }
  .cat {
    font-family: inherit;
    font-size: 12.5px;
    text-align: left;
    color: var(--text);
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 7px 9px;
    cursor: pointer;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    transition:
      background 0.12s,
      color 0.12s;
  }
  .cat:hover {
    background: var(--surface-active);
  }
  .cat.on {
    background: var(--surface-active);
    color: var(--accent);
    font-weight: 500;
  }
  .cat.import {
    color: var(--muted);
    margin-top: 2px;
  }
  .cat.import.on {
    color: var(--accent);
  }

  .detail {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1px;
    max-height: 232px;
    overflow-y: auto;
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
    font-size: 13px;
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
  /* A not-yet-downloaded model the user picked — highlighted, awaiting the Download button below. */
  .opt.chosen {
    background: var(--surface-active);
    box-shadow: inset 0 0 0 1px var(--accent);
  }
  .opt.off {
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
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

  .dl-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    margin-top: 6px;
    font-family: inherit;
    font-size: 12.5px;
    font-weight: 500;
    color: #fff;
    background: var(--accent);
    border: none;
    border-radius: 9px;
    padding: 9px;
    cursor: pointer;
    transition:
      filter 0.15s,
      opacity 0.15s;
  }
  .dl-btn:hover:not(:disabled) {
    filter: brightness(1.05);
  }
  .dl-btn:disabled {
    opacity: 0.6;
    cursor: default;
  }

  .cloud-note {
    margin: 6px 4px 2px;
    font-size: 11px;
    line-height: 1.45;
    color: var(--muted);
  }

  /* ── Import pane (custom HF repo) ── */
  .import-hint {
    margin: 4px 2px;
    font-size: 12.5px;
    line-height: 1.45;
    color: var(--muted);
  }
  .import-soon {
    margin: 6px 2px 2px;
    font-size: 11.5px;
    color: var(--accent);
  }

  /* ── Cloud key entry ── */
  .keybox {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px;
    margin-bottom: 4px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 9px;
  }
  .keylabel {
    font-size: 12px;
    font-weight: 500;
    color: var(--text);
  }
  .keyrow {
    display: flex;
    gap: 6px;
  }
  .keyinput {
    flex: 1;
    min-width: 0;
    font: inherit;
    font-size: 12.5px;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 6px 9px;
  }
  .keyinput:focus {
    outline: none;
    border-color: var(--accent);
  }
  .keyinput:disabled {
    opacity: 0.55;
  }
  .keysave {
    flex: none;
    font-family: inherit;
    font-size: 12px;
    font-weight: 500;
    color: #fff;
    background: var(--accent);
    border: none;
    border-radius: 8px;
    padding: 0 12px;
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
