<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import { onDestroy, onMount } from "svelte";

  type Segment = {
    id: number;
    text: string;
    startMs: number;
    endMs: number;
    source: string;
    speaker: number | null;
    isFinal: boolean;
  };

  type ModelInfo = {
    id: string;
    name: string;
    sizeBytes: number;
    languages: string[];
    installed: boolean;
    active: boolean;
  };

  let running = $state(false);
  let error = $state("");
  let segments = $state<Segment[]>([]);
  let models = $state<ModelInfo[]>([]);
  let downloading = $state<string | null>(null);
  let unlisten: UnlistenFn | undefined;

  const activeModel = $derived(models.find((m) => m.active));
  const canStart = $derived(!!activeModel?.installed);

  function fmtTime(ms: number): string {
    const total = Math.floor(ms / 1000);
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
  }

  function fmtSize(bytes: number): string {
    const mb = bytes / 1_048_576;
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
  }

  async function refreshModels() {
    try {
      models = await invoke<ModelInfo[]>("list_models");
    } catch (e) {
      error = String(e);
    }
  }

  async function download(id: string) {
    error = "";
    downloading = id;
    try {
      await invoke("download_model", { id });
      await refreshModels();
    } catch (e) {
      error = String(e);
    } finally {
      downloading = null;
    }
  }

  async function selectModel(id: string) {
    try {
      await invoke("select_model", { id });
      await refreshModels();
    } catch (e) {
      error = String(e);
    }
  }

  async function ensureListener() {
    if (unlisten) return;
    unlisten = await listen<Segment>("transcript://segment", (event) => {
      segments = [...segments, event.payload];
    });
  }

  async function start() {
    error = "";
    try {
      await ensureListener();
      await invoke("start_session");
      running = true;
    } catch (e) {
      error = String(e);
    }
  }

  async function stop() {
    try {
      await invoke("stop_session");
    } catch (e) {
      error = String(e);
    }
    running = false;
  }

  function clear() {
    segments = [];
  }

  onMount(refreshModels);
  onDestroy(() => unlisten?.());
</script>

<main class="app">
  <header>
    <h1>Wisp</h1>
    <p class="tagline">Local real-time meeting transcription</p>
  </header>

  <section class="models">
    <h2>Model</h2>
    {#each models as m (m.id)}
      <div class="model" class:active={m.active}>
        <div class="meta">
          <div class="name">{m.name}</div>
          <div class="sub">{fmtSize(m.sizeBytes)} · {m.languages.join(" / ")}</div>
        </div>
        {#if downloading === m.id}
          <button class="btn" disabled>Downloading…</button>
        {:else if !m.installed}
          <button class="btn" onclick={() => download(m.id)} disabled={downloading !== null}>Download</button>
        {:else if m.active}
          <span class="badge">Active</span>
        {:else}
          <button class="btn ghost" onclick={() => selectModel(m.id)}>Use</button>
        {/if}
      </div>
    {:else}
      <p class="hint">Loading models…</p>
    {/each}
  </section>

  <div class="controls">
    {#if running}
      <button class="btn stop" onclick={stop}>■ Stop</button>
    {:else}
      <button class="btn start" onclick={start} disabled={!canStart}>● Start (microphone)</button>
    {/if}
    <button class="btn ghost" onclick={clear} disabled={segments.length === 0}>Clear</button>
    <span class="status" class:live={running}>{running ? "listening…" : canStart ? "idle" : "select a model"}</span>
  </div>

  {#if error}
    <p class="error">{error}</p>
  {/if}

  <ul class="transcript">
    {#each segments as seg (seg.id)}
      <li class:partial={!seg.isFinal}>
        <span class="time">{fmtTime(seg.startMs)}</span>
        <span class="who">{seg.source}</span>
        <span class="text">{seg.text}</span>
      </li>
    {:else}
      <li class="empty">No transcript yet — pick a model, press Start, and speak.</li>
    {/each}
  </ul>
</main>

<style>
  :global(body) {
    margin: 0;
    background: #0f1115;
    color: #e6e7ea;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  .app {
    max-width: 760px;
    margin: 0 auto;
    padding: 28px 24px 40px;
  }

  header h1 {
    margin: 0;
    font-size: 28px;
    letter-spacing: -0.02em;
  }

  .tagline {
    margin: 4px 0 20px;
    color: #9aa0aa;
    font-size: 14px;
  }

  h2 {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #9aa0aa;
    margin: 0 0 8px;
  }

  .models {
    margin-bottom: 22px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .model {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    background: #161a22;
    border: 1px solid transparent;
    border-radius: 8px;
  }

  .model.active {
    border-color: #2563eb;
  }

  .meta {
    flex: 1;
    min-width: 0;
  }

  .name {
    font-size: 14px;
    font-weight: 600;
  }

  .sub {
    font-size: 12px;
    color: #9aa0aa;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
  }

  .btn {
    border: none;
    border-radius: 8px;
    padding: 9px 16px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    color: #fff;
    background: #2a2f3a;
    transition: filter 0.15s, opacity 0.15s;
  }

  .btn:hover:not(:disabled) {
    filter: brightness(1.15);
  }

  .btn:disabled {
    opacity: 0.4;
    cursor: default;
  }

  .btn.start {
    background: #2563eb;
  }

  .btn.stop {
    background: #dc2626;
  }

  .btn.ghost {
    background: transparent;
    border: 1px solid #2a2f3a;
    color: #c7cad1;
  }

  .badge {
    font-size: 12px;
    font-weight: 600;
    color: #60a5fa;
    padding: 6px 10px;
  }

  .status {
    margin-left: auto;
    font-size: 13px;
    color: #9aa0aa;
  }

  .status.live {
    color: #34d399;
  }

  .error {
    background: #2a1316;
    border: 1px solid #5b2327;
    color: #fca5a5;
    padding: 10px 12px;
    border-radius: 8px;
    font-size: 13px;
  }

  .hint {
    color: #6b7280;
    font-size: 13px;
  }

  .transcript {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .transcript li {
    display: grid;
    grid-template-columns: 52px 96px 1fr;
    gap: 12px;
    align-items: baseline;
    padding: 10px 12px;
    background: #161a22;
    border-radius: 8px;
    font-size: 15px;
  }

  .transcript li.partial {
    opacity: 0.6;
    font-style: italic;
  }

  .transcript li.empty {
    display: block;
    color: #6b7280;
    background: transparent;
    text-align: center;
    padding: 40px 0;
  }

  .time {
    color: #6b7280;
    font-variant-numeric: tabular-nums;
    font-size: 13px;
  }

  .who {
    color: #818cf8;
    font-weight: 600;
    font-size: 13px;
  }
</style>
