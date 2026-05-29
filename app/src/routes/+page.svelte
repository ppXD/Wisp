<script lang="ts">
  import "@fontsource-variable/geist";
  import "@fontsource-variable/geist-mono";
  import { invoke } from "@tauri-apps/api/core";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import { onDestroy, onMount } from "svelte";
  import { slide } from "svelte/transition";

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
    description: string;
    installed: boolean;
    active: boolean;
  };

  let running = $state(false);
  let error = $state("");
  let segments = $state<Segment[]>([]);
  let models = $state<ModelInfo[]>([]);
  let downloading = $state<string | null>(null);
  let downloadProgress = $state<{ downloaded: number; total: number } | null>(null);
  let downloadFailed = $state<string | null>(null);
  let progressUnlisten: UnlistenFn | undefined;
  let devices = $state<string[]>([]);
  let micDevice = $state("");
  let systemDevice = $state("");
  let language = $state("");
  let systemAudioId = $state("");
  let micOffId = $state("");
  let mode = $state<"live" | "file" | "cloud">("live");
  let screenAuthorized = $state(true);
  let micBlocked = $state(false);
  let permissionBusy = $state(false);
  let unlisten: UnlistenFn | undefined;

  const activeModel = $derived(models.find((m) => m.active));
  const canStart = $derived(!!activeModel?.installed);

  // Which model the picker is showing. Defaults to the active one once models load.
  let chosenId = $state("");
  $effect(() => {
    if (!chosenId && models.length) chosenId = (models.find((m) => m.active) ?? models[0]).id;
  });
  const chosenModel = $derived(models.find((m) => m.id === chosenId));

  async function pickModel(id: string) {
    chosenId = id;
    const m = models.find((x) => x.id === id);
    if (m?.installed) await selectModel(id); // installed → make it the active model
  }
  // System audio on macOS needs Screen Recording permission; only relevant for the one-click source.
  const needsScreenRecording = $derived(
    !!systemAudioId && systemDevice === systemAudioId && !screenAuthorized,
  );
  // Microphone is on unless explicitly set to Off; warn only when access is actually blocked.
  const needsMicPermission = $derived(micDevice !== micOffId && micBlocked);

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
      micBlocked = await invoke<boolean>("microphone_blocked");
    } catch (e) {
      error = String(e);
    }
  }

  async function openMicSettings() {
    try {
      // A denied mic can't be re-prompted by macOS — System Settings is the only way to re-enable.
      await invoke("open_privacy_settings", { pane: "microphone" });
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

  // macOS applies a newly granted Screen Recording (and a re-enabled mic) permission only to a
  // freshly launched process, so the running app must relaunch to pick it up — otherwise the
  // permission banner never clears even though access is granted in System Settings.
  async function restartApp() {
    try {
      await invoke("restart_app");
    } catch (e) {
      error = String(e);
    }
  }

  async function applyDevices() {
    try {
      await invoke("set_devices", { mic: micDevice || null, system: systemDevice || null });
    } catch (e) {
      error = String(e);
    }
  }

  async function applyLanguage() {
    try {
      await invoke("set_language", { language });
    } catch (e) {
      error = String(e);
    }
  }

  function sourceLabel(source: string): string {
    if (source === "Microphone") return "mic";
    if (source === "System") return "system";
    return source.toLowerCase();
  }

  async function download(id: string) {
    error = "";
    downloadFailed = null;
    downloading = id;
    const m = models.find((x) => x.id === id);
    downloadProgress = { downloaded: 0, total: m?.sizeBytes ?? 0 };
    try {
      await invoke("download_model", { id });
      await refreshModels();
      await selectModel(id); // a freshly downloaded model becomes the active one
    } catch (e) {
      downloadFailed = id;
      error = String(e);
    } finally {
      downloading = null;
      downloadProgress = null;
    }
  }

  const downloadPct = $derived(
    downloadProgress && downloadProgress.total > 0
      ? Math.min(100, Math.round((downloadProgress.downloaded / downloadProgress.total) * 100))
      : 0,
  );

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

  async function ensureProgressListener() {
    if (progressUnlisten) return;
    progressUnlisten = await listen<{ id: string; downloaded: number; total: number }>(
      "download://progress",
      (event) => {
        if (event.payload.id === downloading) {
          downloadProgress = { downloaded: event.payload.downloaded, total: event.payload.total };
        }
      },
    );
  }

  // Reflect the backend's real session state. The frontend can reload (e.g. dev HMR) while a
  // session keeps running, which would otherwise leave `running` stale and the UI out of sync.
  async function syncRunning() {
    try {
      if (await invoke<boolean>("session_running")) {
        await ensureListener();
        running = true;
      }
    } catch {
      // best-effort; ignore
    }
  }

  async function start() {
    error = "";
    try {
      await applyDevices();
      await applyLanguage();
      await ensureListener();
      await invoke("start_session");
      running = true;
      // Capture started, so the permissions it needed are granted — clear any stale prompts
      // (macOS can report a stale status to a running process after a Settings change).
      screenAuthorized = true;
      micBlocked = false;
    } catch (e) {
      // If a session is actually already running (e.g. after a reload), reflect that instead of
      // showing the error.
      await syncRunning();
      error = running ? "" : String(e);
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

  let transcriptEl = $state<HTMLUListElement>();
  let pinnedToBottom = true;

  function onTranscriptScroll() {
    if (!transcriptEl) return;
    const gap = transcriptEl.scrollHeight - transcriptEl.scrollTop - transcriptEl.clientHeight;
    pinnedToBottom = gap < 48;
  }

  // Auto-follow the newest line (like a live feed) unless the user scrolled up to read back.
  $effect(() => {
    segments.length;
    if (pinnedToBottom && transcriptEl) {
      const el = transcriptEl;
      requestAnimationFrame(() => (el.scrollTop = el.scrollHeight));
    }
  });

  onMount(() => {
    refreshModels();
    refreshDevices();
    checkPermissions();
    ensureProgressListener();
    syncRunning();
    // Re-check when the window regains focus, so granting in System Settings clears the banner.
    const onFocus = () => checkPermissions();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  });
  onDestroy(() => {
    unlisten?.();
    progressUnlisten?.();
  });
</script>

<main class="app">
  <header>
    <div class="topbar">
      <div class="brand">
        <span class="dot"></span>
        <h1>Wisp</h1>
      </div>
      <nav class="modes">
        <button class:active={mode === "live"} onclick={() => (mode = "live")}>Live</button>
        <button class:active={mode === "file"} onclick={() => (mode = "file")}>File</button>
        <button class:active={mode === "cloud"} onclick={() => (mode = "cloud")}>Cloud</button>
      </nav>
    </div>
    <p class="tagline">On-device transcription — live, files, or cloud</p>
  </header>

  {#if mode === "live"}
  {#if needsScreenRecording}
    <div class="permission">
      <div class="permission-body">
        <div class="permission-title">Allow Screen Recording to capture system audio</div>
        <p class="permission-sub">
          Enable Wisp under Screen Recording in System Settings, then <strong>restart Wisp</strong>
          to apply it — macOS only grants screen access to a freshly launched app. Nothing leaves
          your device.
        </p>
      </div>
      <div class="permission-actions">
        <button class="btn primary" onclick={grantScreenRecording} disabled={permissionBusy}>
          {permissionBusy ? "Requesting…" : "Grant access"}
        </button>
        <button class="btn ghost" onclick={restartApp}>Restart Wisp</button>
      </div>
    </div>
  {/if}

  {#if needsMicPermission}
    <div class="permission">
      <div class="permission-body">
        <div class="permission-title">Microphone access is off</div>
        <p class="permission-sub">
          Enable Wisp under Microphone in System Settings, then <strong>restart Wisp</strong> to
          apply it. Or set Microphone to Off under Advanced to capture system audio only.
        </p>
      </div>
      <div class="permission-actions">
        <button class="btn primary" onclick={openMicSettings}>Open settings</button>
        <button class="btn ghost" onclick={restartApp}>Restart Wisp</button>
      </div>
    </div>
  {/if}

  {#if running}
    <div class="active-model" transition:slide={{ duration: 200 }}>
      <span class="live-pip"></span>
      <span class="active-model-name">{activeModel?.name ?? "Model"}</span>
    </div>
  {:else}
  <section class="card models" transition:slide={{ duration: 200 }}>
    <div class="section-label">Model</div>
    {#if models.length}
      <div class="select-wrap">
        <select class="model-select" value={chosenId} onchange={(e) => pickModel(e.currentTarget.value)} disabled={running || downloading !== null}>
          {#each models as m (m.id)}
            <option value={m.id}>{m.name}{m.installed ? "" : " — not downloaded"}</option>
          {/each}
        </select>
      </div>

      {#if chosenModel}
        <div class="model-detail">
          <div class="model-tags">
            <span class="tag">{fmtSize(chosenModel.sizeBytes)}</span>
            {#each chosenModel.languages as l (l)}<span class="tag">{l}</span>{/each}
          </div>
          {#if chosenModel.description}<p class="model-desc">{chosenModel.description}</p>{/if}
          <div class="model-action">
            {#if downloading === chosenModel.id}
              <div class="dl">
                <div class="dl-track"><div class="dl-fill" style="width:{downloadPct}%"></div></div>
                <span class="dl-label">
                  Downloading… {downloadPct}% · {fmtSize(downloadProgress?.downloaded ?? 0)} / {fmtSize(
                    downloadProgress?.total ?? chosenModel.sizeBytes,
                  )}
                </span>
              </div>
            {:else if downloadFailed === chosenModel.id}
              <button class="btn outline" onclick={() => download(chosenModel.id)}>
                Retry download · {fmtSize(chosenModel.sizeBytes)}
              </button>
            {:else if !chosenModel.installed}
              <button class="btn outline" onclick={() => download(chosenModel.id)} disabled={downloading !== null}>
                Download · {fmtSize(chosenModel.sizeBytes)}
              </button>
            {:else if chosenModel.active}
              <span class="pill">Active</span>
            {:else}
              <button class="btn ghost" onclick={() => selectModel(chosenModel.id)}>Use this model</button>
            {/if}
          </div>
        </div>
      {/if}
    {:else}
      <p class="hint">Loading models…</p>
    {/if}
  </section>

  <details class="advanced" transition:slide={{ duration: 200 }}>
    <summary>Advanced · language & audio</summary>
    <section class="card sources">
    <label class="source-row">
      <span class="source-name">Language</span>
      <select bind:value={language} onchange={applyLanguage} disabled={running}>
        <option value="">Auto-detect</option>
        <option value="yue">Cantonese</option>
        <option value="zh">Chinese (Mandarin)</option>
        <option value="en">English</option>
        <option value="ja">Japanese</option>
        <option value="ko">Korean</option>
      </select>
    </label>
    <label class="source-row">
      <span class="source-name">Microphone <em>(you)</em></span>
      <select bind:value={micDevice} onchange={applyDevices} disabled={running}>
        <option value="">System default</option>
        {#if micOffId}<option value={micOffId}>Off</option>{/if}
        {#each devices as d (d)}<option value={d}>{d}</option>{/each}
      </select>
    </label>
    <label class="source-row">
      <span class="source-name">System audio <em>(everything playing)</em></span>
      <select bind:value={systemDevice} onchange={applyDevices} disabled={running}>
        <option value="">Off</option>
        {#if systemAudioId}<option value={systemAudioId}>System audio — no setup</option>{/if}
        {#each devices as d (d)}<option value={d}>{d}</option>{/each}
      </select>
    </label>
    <p class="hint">By default Wisp captures your <strong>microphone</strong> + <strong>all system audio</strong>, with <strong>echo cancellation</strong> so audio your mic re-hears from the speakers is removed automatically. Want system audio only? Set Microphone to Off. System audio asks for Screen Recording permission once. Set a specific <strong>Language</strong> if auto-detect gets it wrong — recommended for Cantonese.</p>
    </section>
  </details>
  {/if}

  <div class="controls">
    {#if running}
      <button class="btn primary stop" onclick={stop}>Stop</button>
    {:else}
      <button class="btn primary" onclick={start} disabled={!canStart || downloading !== null}>
        {downloading !== null ? "Downloading…" : "Start listening"}
      </button>
    {/if}
    <button class="btn ghost" onclick={clear} disabled={segments.length === 0}>Clear</button>
    <span class="status" class:live={running}>
      <span class="status-dot"></span>{running ? "listening" : canStart ? "ready" : "select a model"}
    </span>
  </div>

  {#if error}
    <p class="error">{error}</p>
  {/if}

  <ul class="transcript" bind:this={transcriptEl} onscroll={onTranscriptScroll}>
    {#each segments as seg (seg.source + "-" + seg.id)}
      <li class:partial={!seg.isFinal} class:system={seg.source === "System"}>
        <span class="time">{fmtTime(seg.startMs)}</span>
        <span class="who">{sourceLabel(seg.source)}</span>
        <span class="text">{seg.text}</span>
      </li>
    {:else}
      <li class="empty">Pick a model, press <em>Start listening</em>, and speak.</li>
    {/each}
  </ul>
  {:else if mode === "file"}
    <section class="placeholder card">
      <div class="placeholder-title">Transcribe a file</div>
      <p class="placeholder-sub">
        Drop in an audio or video file and Wisp transcribes it with your local model, then lets you
        export the text (TXT / SRT / VTT). Coming next.
      </p>
    </section>
  {:else}
    <section class="placeholder card">
      <div class="placeholder-title">Cloud realtime</div>
      <p class="placeholder-sub">
        Stream live audio to a realtime transcription API (e.g. OpenAI Realtime) for the highest
        accuracy with no local model. Coming soon.
      </p>
    </section>
  {/if}
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
    padding: 32px 28px 24px;
    height: 100vh;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
  }

  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .modes {
    display: inline-flex;
    gap: 2px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 3px;
  }

  .modes button {
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    color: var(--muted);
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 6px 14px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }

  .modes button:hover {
    color: var(--text);
  }

  .modes button.active {
    background: var(--accent);
    color: #fff;
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
  }

  .select-wrap {
    position: relative;
  }

  .select-wrap::after {
    content: "";
    position: absolute;
    right: 15px;
    top: 50%;
    width: 7px;
    height: 7px;
    margin-top: -5px;
    border-right: 1.5px solid var(--muted);
    border-bottom: 1.5px solid var(--muted);
    transform: rotate(45deg);
    pointer-events: none;
  }

  .model-select {
    width: 100%;
    appearance: none;
    -webkit-appearance: none;
    font-family: inherit;
    font-size: 14px;
    font-weight: 500;
    color: var(--text);
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    padding: 11px 34px 11px 13px;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }

  .model-select:hover {
    border-color: var(--muted);
    background: var(--surface-active);
  }

  .model-select:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .model-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 12px;
  }

  .tag {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 8px;
  }

  .model-desc {
    margin: 10px 0 0;
    font-size: 13px;
    line-height: 1.55;
    color: var(--muted);
  }

  .model-action {
    margin-top: 12px;
  }

  .dl {
    display: flex;
    flex-direction: column;
    gap: 7px;
  }

  .dl-track {
    height: 7px;
    background: var(--border);
    border-radius: 999px;
    overflow: hidden;
  }

  .dl-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 999px;
    transition: width 0.2s ease;
  }

  .dl-label {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .active-model {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    font-size: 14px;
    font-weight: 500;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 9px 14px;
    margin-bottom: 4px;
  }

  .live-pip {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--live);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--live) 22%, transparent);
    flex-shrink: 0;
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

  .permission-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
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
    padding: 0 4px 0 0;
    display: flex;
    flex-direction: column;
    gap: 7px;
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    scroll-behavior: smooth;
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

  .text {
    min-width: 0;
    overflow-wrap: anywhere;
  }

  .transcript li.system .who {
    color: #5b7fb0;
  }

  .placeholder {
    text-align: center;
    padding: 52px 28px;
  }

  .placeholder-title {
    font-size: 17px;
    font-weight: 600;
  }

  .placeholder-sub {
    margin: 10px auto 0;
    max-width: 430px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
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
    appearance: none;
    -webkit-appearance: none;
    font-family: inherit;
    font-size: 13px;
    color: var(--text);
    background-color: var(--bg);
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='9' height='6' viewBox='0 0 9 6' fill='none' stroke='%2378736a' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M1 1l3.5 3.5L8 1'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 7px 26px 7px 10px;
    max-width: 280px;
    cursor: pointer;
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
    display: inline-flex;
    align-items: center;
    gap: 8px;
    width: fit-content;
    font-size: 13px;
    font-weight: 500;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 8px 13px;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
  }

  .advanced > summary::-webkit-details-marker {
    display: none;
  }

  .advanced > summary::before {
    content: "";
    width: 6px;
    height: 6px;
    border-right: 1.5px solid currentColor;
    border-bottom: 1.5px solid currentColor;
    transform: rotate(-45deg);
    transition: transform 0.15s;
  }

  .advanced[open] > summary::before {
    transform: rotate(45deg);
  }

  .advanced > summary:hover {
    color: var(--text);
    border-color: var(--border-strong);
    background: var(--surface-active);
  }

  .advanced[open] > summary {
    margin-bottom: 12px;
  }

  .advanced .sources {
    margin-bottom: 0;
  }
</style>
