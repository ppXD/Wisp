<script lang="ts">
  import "@fontsource-variable/geist";
  import "@fontsource-variable/geist-mono";
  import { invoke } from "@tauri-apps/api/core";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import { open, save } from "@tauri-apps/plugin-dialog";
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

  // Custom model dropdown. Logical order: ready-to-use (installed) models first, then ones to
  // download — each kept in catalog (recommendation) order.
  let pickerOpen = $state(false);
  const installedModels = $derived(models.filter((m) => m.installed));
  const availableModels = $derived(models.filter((m) => !m.installed));

  async function choose(id: string) {
    pickerOpen = false;
    await pickModel(id);
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

  // ── File mode ────────────────────────────────────────────────────────────
  let fileSegments = $state<Segment[]>([]);
  let fileName = $state("");
  let fileTranscribing = $state(false);
  // Timeline (timestamps) is opt-in: off = most accurate plain text; on = timed for SRT/VTT.
  let fileTimestamps = $state(false);
  let fileHasTimestamps = $state(false);
  let dragOver = $state(false);
  let fileListeners: UnlistenFn[] = [];
  let dropUnlisten: UnlistenFn | undefined;

  async function transcribeFile(path: string) {
    if (fileTranscribing) return;
    if (!canStart) {
      error = "Download a model in the Live tab first.";
      return;
    }
    error = "";
    fileSegments = [];
    fileName = path.split(/[\\/]/).pop() ?? path;
    fileHasTimestamps = fileTimestamps;
    fileTranscribing = true;
    try {
      await invoke("transcribe_file", { path, timestamps: fileTimestamps });
    } catch (e) {
      error = String(e);
      fileTranscribing = false;
    }
  }

  function resetFile() {
    fileSegments = [];
    fileName = "";
  }

  async function pickFile() {
    if (!canStart) {
      error = "Download a model in the Live tab first.";
      return;
    }
    try {
      const path = await open({
        multiple: false,
        directory: false,
        filters: [
          {
            name: "Audio / Video",
            extensions: ["mp3", "m4a", "aac", "wav", "flac", "ogg", "opus", "mp4", "mov", "m4v", "webm", "mkv"],
          },
        ],
      });
      if (typeof path === "string") await transcribeFile(path);
    } catch (e) {
      error = String(e);
    }
  }

  async function exportFile(format: string) {
    if (!fileSegments.length) return;
    const base = (fileName || "transcript").replace(/\.[^.]+$/, "");
    try {
      const dest = await save({
        defaultPath: `${base}.${format}`,
        filters: [{ name: format.toUpperCase(), extensions: [format] }],
      });
      if (typeof dest === "string") await invoke("export_transcript", { format, dest });
    } catch (e) {
      error = String(e);
    }
  }

  let fileListenersReady = false;
  async function ensureFileListeners() {
    if (fileListenersReady) return;
    fileListenersReady = true;
    fileListeners.push(
      await listen<{ name: string; totalMs: number }>("file://meta", (e) => {
        fileName = e.payload.name;
      }),
    );
    fileListeners.push(
      await listen<Segment>("file://segment", (e) => {
        fileSegments = [...fileSegments, e.payload];
      }),
    );
    fileListeners.push(
      await listen("file://done", () => {
        fileTranscribing = false;
      }),
    );
    // Window-level drag-and-drop (Tauri core webview event) — only act on it in File mode.
    dropUnlisten = await getCurrentWebview().onDragDropEvent((event) => {
      if (mode !== "file") return;
      const p = event.payload;
      if (p.type === "enter" || p.type === "over") {
        dragOver = true;
      } else if (p.type === "leave") {
        dragOver = false;
      } else if (p.type === "drop") {
        dragOver = false;
        const path = p.paths[0];
        if (path) transcribeFile(path);
      }
    });
  }

  onMount(() => {
    refreshModels();
    refreshDevices();
    checkPermissions();
    ensureProgressListener();
    ensureFileListeners();
    syncRunning();
    // Re-check when the window regains focus, so granting in System Settings clears the banner.
    const onFocus = () => checkPermissions();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  });
  onDestroy(() => {
    unlisten?.();
    progressUnlisten?.();
    fileListeners.forEach((u) => u());
    dropUnlisten?.();
  });
</script>

<main class="app">
  <nav class="tabs">
    <button class:active={mode === "live"} onclick={() => (mode = "live")}>Live</button>
    <button class:active={mode === "file"} onclick={() => (mode = "file")}>File</button>
    <button class:active={mode === "cloud"} onclick={() => (mode = "cloud")}>Cloud</button>
  </nav>

  {#snippet modelPicker()}
    {#if models.length}
      <div class="picker">
        <button
          class="picker-trigger"
          class:open={pickerOpen}
          onclick={() => (pickerOpen = !pickerOpen)}
          disabled={downloading !== null}
        >
          <span class="picker-label">{chosenModel?.name ?? "Select a model"}</span>
          <span class="picker-caret"></span>
        </button>
        {#if pickerOpen}
          <button class="picker-backdrop" aria-label="Close" onclick={() => (pickerOpen = false)}
          ></button>
          <div class="picker-menu" transition:slide={{ duration: 120 }}>
            {#if installedModels.length}
              <div class="picker-section">Installed</div>
              {#each installedModels as m (m.id)}
                <button class="picker-opt" class:sel={m.id === chosenId} onclick={() => choose(m.id)}>
                  <span class="picker-opt-name">{m.name}</span>
                  {#if m.active}<span class="picker-tag">active</span>{/if}
                </button>
              {/each}
            {/if}
            {#if availableModels.length}
              <div class="picker-section">Available to download</div>
              {#each availableModels as m (m.id)}
                <button class="picker-opt" class:sel={m.id === chosenId} onclick={() => choose(m.id)}>
                  <span class="picker-opt-name">{m.name}</span>
                  <span class="picker-opt-size">{fmtSize(m.sizeBytes)}</span>
                </button>
              {/each}
            {/if}
          </div>
        {/if}
      </div>
    {:else}
      <span class="muted">Loading models…</span>
    {/if}
  {/snippet}

  {#if mode === "live"}
    <section class="box">
      <div class="box-head">
        {#if running}
          <span class="active-model"><span class="live-pip"></span>{activeModel?.name ?? "Model"}</span>
        {:else}
          {@render modelPicker()}
        {/if}
        <span class="status" class:live={running}>
          <span class="status-dot"></span>{running ? "listening" : canStart ? "ready" : "no model"}
        </span>
      </div>

      {#if !running && (needsScreenRecording || needsMicPermission || error || (chosenModel && !chosenModel.installed))}
        <div class="box-aux">
          {#if chosenModel && !chosenModel.installed}
            {#if downloading === chosenModel.id && downloadProgress}
              <div class="dl-bar">
                <div class="dl-track"><div class="dl-fill" style="width:{downloadPct}%"></div></div>
                <span class="dl-label">
                  {downloadPct}% · {fmtSize(downloadProgress.downloaded)} / {fmtSize(downloadProgress.total)}
                </span>
              </div>
            {:else}
              <button class="btn outline" onclick={() => download(chosenModel.id)} disabled={downloading !== null}>
                {downloadFailed === chosenModel.id ? "Retry download" : "Download"} · {fmtSize(chosenModel.sizeBytes)}
              </button>
            {/if}
          {/if}

          {#if needsScreenRecording}
            <div class="notice">
              <span class="notice-text">
                <strong>Screen Recording is off.</strong> Enable Wisp under Screen Recording in System
                Settings, then restart to apply it.
              </span>
              <span class="notice-actions">
                <button class="btn outline sm" onclick={grantScreenRecording} disabled={permissionBusy}>
                  {permissionBusy ? "…" : "Grant"}
                </button>
                <button class="btn ghost sm" onclick={restartApp}>Restart</button>
              </span>
            </div>
          {/if}

          {#if needsMicPermission}
            <div class="notice">
              <span class="notice-text">
                <strong>Microphone is off.</strong> Enable Wisp under Microphone in System Settings,
                then restart — or set Microphone to Off in Advanced.
              </span>
              <span class="notice-actions">
                <button class="btn outline sm" onclick={openMicSettings}>Settings</button>
                <button class="btn ghost sm" onclick={restartApp}>Restart</button>
              </span>
            </div>
          {/if}

          {#if error}
            <div class="notice error">{error}</div>
          {/if}
        </div>
      {/if}

      <ul class="feed" bind:this={transcriptEl} onscroll={onTranscriptScroll}>
        {#each segments as seg (seg.source + "-" + seg.id)}
          <li class:partial={!seg.isFinal} class:system={seg.source === "System"}>
            <span class="meta">
              <span class="time">{fmtTime(seg.startMs)}</span>
              <span class="who">{sourceLabel(seg.source)}</span>
            </span>
            <span class="text">{seg.text}</span>
          </li>
        {:else}
          <li class="empty">Pick a model, press <em>Start</em>, and speak.</li>
        {/each}
      </ul>

      <div class="box-foot">
        {#if running}
          <button class="btn primary stop" onclick={stop}>Stop</button>
        {:else}
          <button class="btn primary" onclick={start} disabled={!canStart || downloading !== null}>
            {downloading !== null ? "Downloading…" : "Start"}
          </button>
        {/if}
        <button class="btn ghost" onclick={clear} disabled={segments.length === 0}>Clear</button>
      </div>
    </section>

    {#if !running}
      <details class="advanced-panel">
        <summary>Advanced · language &amp; audio</summary>
        <div class="advanced-grid">
          <label class="source-row">
            <span class="source-name">Language</span>
            <select bind:value={language} onchange={applyLanguage}>
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
            <select bind:value={micDevice} onchange={applyDevices}>
              <option value="">System default</option>
              {#if micOffId}<option value={micOffId}>Off</option>{/if}
              {#each devices as d (d)}<option value={d}>{d}</option>{/each}
            </select>
          </label>
          <label class="source-row">
            <span class="source-name">System audio <em>(everything playing)</em></span>
            <select bind:value={systemDevice} onchange={applyDevices}>
              <option value="">Off</option>
              {#if systemAudioId}<option value={systemAudioId}>System audio — no setup</option>{/if}
              {#each devices as d (d)}<option value={d}>{d}</option>{/each}
            </select>
          </label>
          <p class="hint">
            By default Wisp captures your <strong>microphone</strong> + <strong>all system audio</strong>
            with <strong>echo cancellation</strong>. Want system audio only? Set Microphone to Off.
            Set a <strong>Language</strong> if auto-detect gets it wrong — recommended for Cantonese.
          </p>
        </div>
      </details>
    {/if}
  {:else if mode === "file"}
    <section class="box">
      {#if fileTranscribing || fileSegments.length}
        <div class="box-head">
          <span class="active-model">{fileName || "File"}</span>
          <span class="status" class:live={fileTranscribing}>
            <span class="status-dot"></span>{fileTranscribing ? "transcribing…" : "done"}
          </span>
        </div>
        <ul class="feed">
          {#each fileSegments as seg (seg.id)}
            <li>
              {#if fileHasTimestamps}
                <span class="meta"><span class="time">{fmtTime(seg.startMs)}</span></span>
              {/if}
              <span class="text">{seg.text}</span>
            </li>
          {:else}
            <li class="empty">Transcribing… large files take a little while.</li>
          {/each}
        </ul>
        <div class="box-foot">
          <div class="export-group">
            {#if fileSegments.length && !fileTranscribing}
              <span class="export-label">Export</span>
              <button class="btn outline sm" onclick={() => exportFile("txt")}>TXT</button>
              {#if fileHasTimestamps}
                <button class="btn outline sm" onclick={() => exportFile("srt")}>SRT</button>
                <button class="btn outline sm" onclick={() => exportFile("vtt")}>VTT</button>
              {/if}
            {/if}
          </div>
          <button class="btn ghost" onclick={resetFile} disabled={fileTranscribing}>
            Transcribe another
          </button>
        </div>
      {:else}
        <div class="box-head">
          {@render modelPicker()}
          <label class="toggle">
            <input type="checkbox" bind:checked={fileTimestamps} />
            <span>Timeline</span>
          </label>
        </div>
        <button
          class="dropzone"
          class:over={dragOver}
          onclick={pickFile}
          disabled={!canStart}
          aria-label="Choose a file to transcribe"
        >
          <div class="dropzone-title">Click to choose a file, or drop one here</div>
          <p class="dropzone-sub">
            {#if canStart}
              mp3, m4a, wav, flac, mp4, mov… transcribed locally with
              <strong>{activeModel?.name}</strong>{fileTimestamps
                ? ", with subtitle timing"
                : " — plain text, most accurate"}.
            {:else}
              Download a model in the Live tab first.
            {/if}
          </p>
        </button>
      {/if}
    </section>
  {:else}
    <section class="box box-center">
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

  /* Strict viewport-height column: tabs on top, one content box filling the rest. */
  .app {
    max-width: 820px;
    margin: 0 auto;
    padding: 16px 20px 18px;
    height: 100dvh;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  /* Top row — the three tabs. */
  .tabs {
    flex: none;
    display: inline-flex;
    align-self: center;
    gap: 3px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 3px;
  }

  .tabs button {
    font-family: inherit;
    font-size: 13.5px;
    font-weight: 500;
    color: var(--muted);
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 6px 22px;
    cursor: pointer;
    transition:
      background 0.15s,
      color 0.15s;
  }

  .tabs button:hover {
    color: var(--text);
  }

  .tabs button.active {
    background: var(--accent);
    color: #fff;
  }

  /* The content box — fills all remaining height; only its feed scrolls. */
  .box {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
  }

  .box-center {
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 28px;
  }

  .box-head {
    flex: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 11px 14px;
    border-bottom: 1px solid var(--border);
  }

  .box-aux {
    flex: none;
    display: flex;
    flex-direction: column;
    gap: 9px;
    padding: 12px 14px;
    border-bottom: 1px solid var(--border);
  }

  .box-foot {
    flex: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 11px 14px;
    border-top: 1px solid var(--border);
  }

  /* Custom Claude-style model dropdown. */
  .picker {
    position: relative;
    flex: 1 1 auto;
    min-width: 0;
    max-width: 460px;
  }

  .picker-trigger {
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
    padding: 8px 12px;
    cursor: pointer;
    transition:
      border-color 0.15s,
      background 0.15s;
  }

  .picker-trigger:hover:not(:disabled),
  .picker-trigger.open {
    border-color: var(--muted);
    background: var(--surface-active);
  }

  .picker-trigger:disabled {
    opacity: 0.55;
    cursor: default;
  }

  .picker-label {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    text-align: left;
  }

  .picker-caret {
    flex: none;
    width: 7px;
    height: 7px;
    border-right: 1.5px solid var(--muted);
    border-bottom: 1.5px solid var(--muted);
    transform: rotate(45deg);
    transition: transform 0.15s;
  }

  .picker-trigger.open .picker-caret {
    transform: rotate(-135deg);
  }

  .picker-backdrop {
    position: fixed;
    inset: 0;
    z-index: 20;
    background: transparent;
    border: none;
    cursor: default;
  }

  .picker-menu {
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    right: 0;
    z-index: 21;
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    box-shadow: 0 14px 34px -10px rgba(40, 30, 20, 0.28);
    padding: 6px;
    max-height: 56vh;
    overflow-y: auto;
  }

  .picker-section {
    font-family: var(--font-mono);
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    padding: 8px 10px 4px;
  }

  .picker-opt {
    width: 100%;
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

  .picker-opt:hover {
    background: var(--surface-active);
  }

  .picker-opt.sel {
    color: var(--accent);
    font-weight: 500;
  }

  .picker-opt-name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .picker-opt-size {
    flex: none;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .picker-tag {
    flex: none;
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--live);
  }

  .active-model {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    min-width: 0;
    font-size: 14px;
    font-weight: 500;
    color: var(--text);
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }

  .live-pip {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--live);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--live) 22%, transparent);
    flex-shrink: 0;
  }

  .btn {
    font-family: inherit;
    font-size: 14px;
    font-weight: 500;
    border-radius: 9px;
    padding: 8px 18px;
    border: 1px solid transparent;
    cursor: pointer;
    background: var(--surface);
    color: var(--text);
    white-space: nowrap;
    transition:
      background 0.15s,
      border-color 0.15s,
      opacity 0.15s;
  }

  .btn.sm {
    font-size: 12.5px;
    padding: 6px 11px;
    border-radius: 8px;
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

  .status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 13px;
    color: var(--muted);
    white-space: nowrap;
    flex: none;
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

  .muted {
    color: var(--muted);
    font-size: 13px;
  }

  .dl-bar {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .dl-track {
    flex: 1;
    height: 6px;
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
    flex: none;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .notice {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--surface-active);
    border: 1px solid color-mix(in srgb, var(--accent) 30%, var(--border));
    border-radius: 10px;
    padding: 9px 12px;
    font-size: 13px;
    line-height: 1.45;
  }

  .notice-text {
    flex: 1;
    min-width: 0;
    color: var(--muted);
  }

  .notice-text strong {
    color: var(--text);
    font-weight: 600;
  }

  .notice-actions {
    flex: none;
    display: flex;
    gap: 7px;
  }

  .notice.error {
    background: color-mix(in srgb, var(--stop) 9%, var(--bg));
    border-color: color-mix(in srgb, var(--stop) 35%, var(--border));
    color: var(--stop);
    display: block;
  }

  /* Advanced panel sits below the content box; collapsed by default so it costs ~no height. */
  .advanced-panel {
    flex: none;
  }

  .advanced-panel > summary {
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
    padding: 7px 12px;
    transition:
      color 0.15s,
      border-color 0.15s,
      background 0.15s;
  }

  .advanced-panel > summary::-webkit-details-marker {
    display: none;
  }

  .advanced-panel > summary::before {
    content: "";
    width: 6px;
    height: 6px;
    border-right: 1.5px solid currentColor;
    border-bottom: 1.5px solid currentColor;
    transform: rotate(-45deg);
    transition: transform 0.15s;
  }

  .advanced-panel[open] > summary::before {
    transform: rotate(45deg);
  }

  .advanced-panel > summary:hover {
    color: var(--text);
    border-color: var(--border-strong);
    background: var(--surface-active);
  }

  .advanced-panel[open] > summary {
    margin-bottom: 10px;
  }

  .advanced-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
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

  .hint {
    color: var(--muted);
    font-size: 12.5px;
    line-height: 1.55;
    margin: 2px 0 0;
  }

  /* The feed — the only scroller, fills the box. */
  .feed {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    list-style: none;
    margin: 0;
    padding: 4px 6px;
    display: flex;
    flex-direction: column;
    scroll-behavior: smooth;
  }

  .feed li {
    display: flex;
    align-items: baseline;
    gap: 16px;
    padding: 12px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 16px;
    line-height: 1.55;
  }

  .feed li:last-child {
    border-bottom: none;
  }

  .feed li.partial {
    opacity: 0.55;
    font-style: italic;
  }

  .meta {
    flex: none;
    display: flex;
    flex-direction: column;
    gap: 2px;
    width: 52px;
    padding-top: 1px;
  }

  .time {
    font-family: var(--font-mono);
    color: var(--muted);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }

  .who {
    font-family: var(--font-mono);
    color: var(--accent);
    font-size: 11px;
    text-transform: lowercase;
  }

  .feed li.system .who {
    color: #5b7fb0;
  }

  .text {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
  }

  .feed li.empty {
    display: block;
    margin: auto 0;
    border-bottom: none;
    color: var(--muted);
    text-align: center;
    font-size: 14px;
  }

  .feed li.empty em {
    color: var(--accent);
    font-style: normal;
  }

  .dropzone {
    flex: 1;
    margin: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-family: inherit;
    color: var(--text);
    background: transparent;
    border: 2px dashed var(--border-strong);
    border-radius: 14px;
    cursor: pointer;
    transition:
      border-color 0.15s,
      background 0.15s;
  }

  .dropzone:hover:not(:disabled),
  .dropzone.over {
    border-color: var(--accent);
    background: var(--surface-active);
  }

  .dropzone:disabled {
    cursor: default;
    opacity: 0.6;
  }

  .export-group {
    display: flex;
    align-items: center;
    gap: 7px;
  }

  .export-label {
    font-size: 13px;
    color: var(--muted);
    margin-right: 2px;
  }

  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex: none;
    font-size: 13px;
    color: var(--muted);
    cursor: pointer;
    white-space: nowrap;
  }

  .toggle input {
    accent-color: var(--accent);
    cursor: pointer;
  }

  .dropzone-title {
    font-size: 17px;
    font-weight: 600;
  }

  .dropzone-sub {
    margin: 8px 22px 0;
    max-width: 420px;
    color: var(--muted);
    font-size: 13.5px;
    line-height: 1.6;
  }

  .placeholder-title {
    font-size: 18px;
    font-weight: 600;
  }

  .placeholder-sub {
    margin: 10px auto 0;
    max-width: 420px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
  }
</style>
