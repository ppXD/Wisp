<script lang="ts">
  import "@fontsource-variable/geist";
  import "@fontsource-variable/geist-mono";
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
  let devices = $state<string[]>([]);
  let micDevice = $state("");
  let systemDevice = $state("");
  let systemAudioId = $state("");
  let micOffId = $state("");
  let screenAuthorized = $state(true);
  let permissionBusy = $state(false);
  let unlisten: UnlistenFn | undefined;

  const activeModel = $derived(models.find((m) => m.active));
  const canStart = $derived(!!activeModel?.installed);
  // System audio on macOS needs Screen Recording permission; only relevant for the one-click source.
  const needsScreenRecording = $derived(
    !!systemAudioId && systemDevice === systemAudioId && !screenAuthorized,
  );

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

  async function refreshDevices() {
    try {
      devices = await invoke<string[]>("list_input_devices");
      systemAudioId = await invoke<string>("system_audio_id");
      micOffId = await invoke<string>("mic_off_id");
      // Default: capture system audio too, so one click grabs everything (you + all audio).
      if (!systemDevice) systemDevice = systemAudioId;
    } catch (e) {
      error = String(e);
    }
  }

  async function checkPermissions() {
    try {
      screenAuthorized = await invoke<boolean>("screen_recording_authorized");
    } catch (e) {
      error = String(e);
    }
  }

  async function grantScreenRecording() {
    permissionBusy = true;
    try {
      const granted = await invoke<boolean>("request_screen_recording");
      screenAuthorized = granted;
      // After a prior denial macOS won't re-prompt — send the user to System Settings to flip it.
      if (!granted) await invoke("open_privacy_settings", { pane: "screen" });
    } catch (e) {
      error = String(e);
    } finally {
      permissionBusy = false;
    }
  }

  async function applyDevices() {
    try {
      await invoke("set_devices", { mic: micDevice || null, system: systemDevice || null });
    } catch (e) {
      error = String(e);
    }
  }

  function sourceLabel(source: string): string {
    if (source === "Microphone") return "me";
    if (source === "System") return "meeting";
    return source.toLowerCase();
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
      await applyDevices();
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

  onMount(() => {
    refreshModels();
    refreshDevices();
    checkPermissions();
    // Re-check when the window regains focus, so granting in System Settings clears the banner.
    const onFocus = () => checkPermissions();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  });
  onDestroy(() => unlisten?.());
</script>

<main class="app">
  <header>
    <div class="brand">
      <span class="dot"></span>
      <h1>Wisp</h1>
    </div>
    <p class="tagline">Local, real-time meeting transcription · on-device</p>
  </header>

  {#if needsScreenRecording}
    <div class="permission">
      <div class="permission-body">
        <div class="permission-title">Allow Screen Recording to hear the meeting</div>
        <p class="permission-sub">
          Wisp captures system/meeting audio through macOS Screen Recording. Grant access once —
          nothing leaves your device.
        </p>
      </div>
      <button class="btn primary" onclick={grantScreenRecording} disabled={permissionBusy}>
        {permissionBusy ? "Requesting…" : "Grant access"}
      </button>
    </div>
  {/if}

  <section class="card models">
    <div class="section-label">Model</div>
    {#each models as m (m.id)}
      <div class="model" class:active={m.active}>
        <div class="meta">
          <div class="name">{m.name}</div>
          <div class="sub">{fmtSize(m.sizeBytes)} · {m.languages.join(" · ")}</div>
        </div>
        {#if downloading === m.id}
          <button class="btn" disabled>Downloading…</button>
        {:else if !m.installed}
          <button class="btn outline" onclick={() => download(m.id)} disabled={downloading !== null}>Download</button>
        {:else if m.active}
          <span class="pill">Active</span>
        {:else}
          <button class="btn ghost" onclick={() => selectModel(m.id)}>Use</button>
        {/if}
      </div>
    {:else}
      <p class="hint">Loading models…</p>
    {/each}
  </section>

  <details class="advanced">
    <summary>Advanced · audio sources</summary>
    <section class="card sources">
    <label class="source-row">
      <span class="source-name">Microphone <em>(you)</em></span>
      <select bind:value={micDevice} onchange={applyDevices} disabled={running}>
        <option value="">System default</option>
        {#if micOffId}<option value={micOffId}>Off</option>{/if}
        {#each devices as d (d)}<option value={d}>{d}</option>{/each}
      </select>
    </label>
    <label class="source-row">
      <span class="source-name">Meeting audio <em>(others)</em></span>
      <select bind:value={systemDevice} onchange={applyDevices} disabled={running}>
        <option value="">Off</option>
        {#if systemAudioId}<option value={systemAudioId}>System audio — no setup</option>{/if}
        {#each devices as d (d)}<option value={d}>{d}</option>{/each}
      </select>
    </label>
    <p class="hint">By default Wisp captures your <strong>microphone</strong> + <strong>all system audio</strong> (you + anything playing), with <strong>echo cancellation</strong> so the meeting audio your mic re-hears on speakers is removed automatically. Want meeting audio only? Set Microphone to Off. System audio asks for Screen Recording permission once.</p>
    </section>
  </details>

  <div class="controls">
    {#if running}
      <button class="btn primary stop" onclick={stop}>Stop</button>
    {:else}
      <button class="btn primary" onclick={start} disabled={!canStart}>Start listening</button>
    {/if}
    <button class="btn ghost" onclick={clear} disabled={segments.length === 0}>Clear</button>
    <span class="status" class:live={running}>
      <span class="status-dot"></span>{running ? "listening" : canStart ? "ready" : "select a model"}
    </span>
  </div>

  {#if error}
    <p class="error">{error}</p>
  {/if}

  <ul class="transcript">
    {#each segments as seg (seg.source + "-" + seg.id)}
      <li class:partial={!seg.isFinal} class:meeting={seg.source === "System"}>
        <span class="time">{fmtTime(seg.startMs)}</span>
        <span class="who">{sourceLabel(seg.source)}</span>
        <span class="text">{seg.text}</span>
      </li>
    {:else}
      <li class="empty">Pick a model, press <em>Start listening</em>, and speak.</li>
    {/each}
  </ul>
</main>

<style>
  :global(:root) {
    --bg: #f7f4ee;
    --surface: #fdfcfa;
    --surface-active: #f7ece6;
    --text: #1a1915;
    --muted: #78736a;
    --border: #e8e2d5;
    --border-strong: #ddd5c4;
    --accent: #c96442;
    --accent-hover: #b5573a;
    --stop: #b0463a;
    --live: #5f8c6a;
    --font-sans: "Geist Variable", system-ui, -apple-system, sans-serif;
    --font-mono: "Geist Mono Variable", ui-monospace, monospace;
  }

  :global(body) {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }

  .app {
    max-width: 720px;
    margin: 0 auto;
    padding: 40px 28px 56px;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    background: var(--accent);
  }

  header h1 {
    margin: 0;
    font-size: 26px;
    font-weight: 600;
    letter-spacing: -0.02em;
  }

  .tagline {
    margin: 8px 0 28px;
    color: var(--muted);
    font-size: 14px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
  }

  .section-label {
    font-family: var(--font-mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 12px;
  }

  .models {
    margin-bottom: 24px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .model {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 14px;
    border: 1px solid var(--border);
    border-radius: 11px;
    background: var(--bg);
    transition: border-color 0.15s, background 0.15s;
  }

  .model.active {
    border-color: var(--accent);
    background: var(--surface-active);
  }

  .meta {
    flex: 1;
    min-width: 0;
  }

  .name {
    font-size: 14.5px;
    font-weight: 500;
  }

  .sub {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
    margin-top: 3px;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 18px;
  }

  .btn {
    font-family: inherit;
    font-size: 14px;
    font-weight: 500;
    border-radius: 10px;
    padding: 9px 18px;
    border: 1px solid transparent;
    cursor: pointer;
    background: var(--surface);
    color: var(--text);
    transition: background 0.15s, border-color 0.15s, opacity 0.15s;
  }

  .btn:disabled {
    opacity: 0.45;
    cursor: default;
  }

  .btn.primary {
    background: var(--accent);
    color: #fff;
  }

  .btn.primary:hover:not(:disabled) {
    background: var(--accent-hover);
  }

  .btn.primary.stop {
    background: var(--stop);
  }

  .btn.outline {
    background: transparent;
    border-color: var(--accent);
    color: var(--accent);
  }

  .btn.outline:hover:not(:disabled) {
    background: var(--surface-active);
  }

  .btn.ghost {
    background: transparent;
    border-color: var(--border-strong);
    color: var(--muted);
  }

  .btn.ghost:hover:not(:disabled) {
    color: var(--text);
    border-color: var(--muted);
  }

  .pill {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--accent);
    padding: 6px 12px;
    border: 1px solid var(--accent);
    border-radius: 999px;
  }

  .status {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 13px;
    color: var(--muted);
  }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--border-strong);
  }

  .status.live {
    color: var(--live);
  }

  .status.live .status-dot {
    background: var(--live);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--live) 22%, transparent);
  }

  .error {
    background: color-mix(in srgb, var(--stop) 9%, var(--bg));
    border: 1px solid color-mix(in srgb, var(--stop) 35%, var(--border));
    color: var(--stop);
    padding: 11px 14px;
    border-radius: 10px;
    font-size: 13px;
  }

  .permission {
    display: flex;
    align-items: center;
    gap: 16px;
    background: var(--surface-active);
    border: 1px solid color-mix(in srgb, var(--accent) 35%, var(--border));
    border-radius: 14px;
    padding: 15px 18px;
    margin-bottom: 24px;
  }

  .permission-body {
    flex: 1;
    min-width: 0;
  }

  .permission-title {
    font-size: 14.5px;
    font-weight: 600;
  }

  .permission-sub {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
  }

  .permission .btn {
    flex-shrink: 0;
  }

  .hint {
    color: var(--muted);
    font-size: 13px;
    margin: 4px 0;
  }

  .transcript {
    list-style: none;
    margin: 8px 0 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 7px;
  }

  .transcript li {
    display: grid;
    grid-template-columns: 48px 104px 1fr;
    gap: 14px;
    align-items: baseline;
    padding: 13px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 11px;
    font-size: 15.5px;
    line-height: 1.5;
  }

  .transcript li.partial {
    opacity: 0.55;
    font-style: italic;
  }

  .transcript li.empty {
    display: block;
    color: var(--muted);
    background: transparent;
    border: 1px dashed var(--border-strong);
    text-align: center;
    padding: 44px 16px;
    font-size: 14px;
  }

  .transcript li.empty em {
    color: var(--accent);
    font-style: normal;
  }

  .time {
    font-family: var(--font-mono);
    color: var(--muted);
    font-size: 12.5px;
    font-variant-numeric: tabular-nums;
  }

  .who {
    font-family: var(--font-mono);
    color: var(--accent);
    font-size: 12px;
    text-transform: lowercase;
  }

  .transcript li.meeting .who {
    color: #5b7fb0;
  }

  .sources {
    margin-bottom: 24px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .source-row {
    display: flex;
    align-items: center;
    gap: 14px;
  }

  .source-name {
    flex: 1;
    font-size: 14px;
  }

  .source-name em {
    color: var(--muted);
    font-style: normal;
    font-size: 13px;
  }

  .source-row select {
    font-family: inherit;
    font-size: 13px;
    color: var(--text);
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 7px 10px;
    max-width: 280px;
  }

  .source-row select:disabled {
    opacity: 0.5;
  }
  .advanced {
    margin-bottom: 24px;
  }

  .advanced > summary {
    cursor: pointer;
    list-style: none;
    font-family: var(--font-mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    padding: 4px 0;
  }

  .advanced > summary::-webkit-details-marker {
    display: none;
  }

  .advanced[open] > summary {
    margin-bottom: 10px;
  }

  .advanced .sources {
    margin-bottom: 0;
  }
</style>
