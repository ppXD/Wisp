<script lang="ts">
  // Settings → Notes search: choose how the Library searches (full-text / semantic / hybrid) and
  // which local embedding model powers semantic + hybrid. Picking a model downloads it on first use
  // (blocking, so the row shows a "downloading…" state) and installs it as the library's embedder.
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { i18n } from "$lib/i18n.svelte";

  type EmbModel = { id: string; label: string; dim: number; size_mb: number };

  let mode = $state("fulltext");
  let selected = $state<string | null>(null); // active embedding model id, null = off (full-text only)
  let models = $state<EmbModel[]>([]);
  let busy = $state(""); // id currently downloading/loading ("off" while clearing)
  let error = $state("");

  async function load() {
    try {
      const [m, sel, list] = await Promise.all([
        invoke<string>("search_mode"),
        invoke<string | null>("embedding_model"),
        invoke<EmbModel[]>("list_embedding_models"),
      ]);
      mode = m;
      selected = sel;
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

  async function pick(id: string | null) {
    if (busy) return;
    busy = id ?? "off";
    error = "";
    try {
      await invoke("set_embedding_model", { id });
      selected = id;
    } catch (e) {
      error = String(e);
    }
    busy = "";
  }

  function sizeLabel(mb: number): string {
    return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`;
  }
</script>

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
  <li>
    <button class="emb" class:on={selected === null} disabled={!!busy} onclick={() => pick(null)}>
      <span class="emb-name">{i18n.t.settings.embedOff}</span>
      <span class="emb-meta">{i18n.t.settings.embedOffHint}</span>
    </button>
  </li>
  {#each models as m (m.id)}
    <li>
      <button class="emb" class:on={selected === m.id} disabled={!!busy} onclick={() => pick(m.id)}>
        <span class="emb-name">{m.label}</span>
        <span class="emb-meta">
          {m.dim}d · {sizeLabel(m.size_mb)}{selected === m.id
            ? ` · ${i18n.t.settings.embedActive}`
            : ""}{busy === m.id ? ` · ${i18n.t.settings.embedDownloading}` : ""}
        </span>
      </button>
    </li>
  {/each}
</ul>

{#if error}<p class="set-error">{error}</p>{/if}
<p class="set-note">{i18n.t.settings.embedCustomSoon}</p>

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
  .emb {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 2px;
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
  .emb:hover:not(:disabled) {
    border-color: var(--border-strong);
    background: var(--surface-active);
  }
  .emb:disabled {
    cursor: default;
    opacity: 0.7;
  }
  .emb.on {
    border-color: var(--accent);
  }
  .emb-name {
    font-size: 13.5px;
    font-weight: 600;
  }
  .emb-meta {
    font-size: 12px;
    color: var(--muted);
  }
</style>
