<script lang="ts">
  import "@fontsource-variable/geist";
  import "@fontsource-variable/geist-mono";
  import { invoke } from "@tauri-apps/api/core";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import { open, save } from "@tauri-apps/plugin-dialog";
  import { onDestroy, onMount } from "svelte";
  import { fly } from "svelte/transition";
  import Modal from "$lib/Modal.svelte";
  import ParamsPanel from "$lib/ParamsPanel.svelte";
  import AiNotes from "$lib/AiNotes.svelte";
  import Settings from "$lib/Settings.svelte";
  import {
    refreshCloud,
    cloudReady,
    cloudProvider,
    cloudState,
    openEndpointsModal,
    streamingParams,
    batchParams,
    defaultParamValues,
    changedParamValues,
    loadParamValues,
    saveParamValues,
    type ParamSpec,
    type ParamValue,
  } from "$lib/cloud.svelte";

  type Segment = {
    id: number;
    text: string;
    startMs: number;
    endMs: number;
    source: string;
    speaker: number | null;
    isFinal: boolean;
    /** A parallel rendering (a cloud model session's translation), shown under the verbatim text. */
    auxText?: string | null;
  };

  type ModelInfo = {
    id: string;
    name: string;
    sizeBytes: number;
    languages: string[];
    description: string;
    installed: boolean;
    active: boolean;
    family: string;
    recommendedLive: boolean;
    recommendedFile: boolean;
    coremlAvailable: boolean;
    coremlInstalled: boolean;
    coremlSizeBytes: number;
    // How the model fits this machine: "ready" | "heavy" (runs but large for the RAM) | "blocked"
    // (this OS/machine can't run it). Blocked models are shown greyed with the reason, never started.
    fit: string;
    fitReason?: string | null;
  };

  let running = $state(false);
  // True while a session is connecting (the cloud WebSocket handshake blocks ~1-2s) — drives the
  // Start button's disabled + spinner state so the click feels responsive instead of frozen.
  let starting = $state(false);
  // True while a session is tearing down (joining the capture/socket threads) — same smooth
  // disabled + spinner treatment on the Stop button.
  let stopping = $state(false);
  // True once a start has been "Connecting" long enough to be worth telling the user it's loading the
  // model — a slow first run is progress, not a hang, but it looks like one without a hint.
  let slowStart = $state(false);
  // Elapsed recording time (ms), driving the live "Recording · M:SS" readout. Ticks every 250ms
  // while a session runs and resets to 0 when it stops.
  let elapsedMs = $state(0);
  $effect(() => {
    if (!running) {
      elapsedMs = 0;
      return;
    }

    const started = Date.now();
    elapsedMs = 0;
    const id = setInterval(() => (elapsedMs = Date.now() - started), 250);
    return () => clearInterval(id);
  });
  let error = $state("");
  // Non-fatal notice from a started session (e.g. system audio unavailable → mic-only).
  let liveNotice = $state("");
  let segments = $state<Segment[]>([]);
  let models = $state<ModelInfo[]>([]);
  let downloading = $state<string | null>(null);
  let downloadProgress = $state<{ downloaded: number; total: number } | null>(null);
  let downloadFailed = $state<string | null>(null);
  // Core ML (Neural Engine) encoder download, tracked separately from the model download.
  let downloadingCoreml = $state<string | null>(null);
  let coremlProgress = $state<{ downloaded: number; total: number } | null>(null);
  let progressUnlisten: UnlistenFn | undefined;
  let devices = $state<string[]>([]);
  let micDevice = $state("");
  let systemDevice = $state("");
  let language = $state("");
  let liveDenoiser = $state<string | null>(null);
  let liveDiarize = $state(false);
  let liveAccurate = $state(false);
  let livePrompt = $state("");
  let systemAudioId = $state("");
  let micOffId = $state("");
  let mode = $state<"live" | "file">("live");

  // Collapsible left rail: collapsed (icon-only) by default; expands to icon + label rows. Persisted.
  let sidebarExpanded = $state(false);
  function toggleSidebar() {
    sidebarExpanded = !sidebarExpanded;
    try {
      localStorage.setItem("wisp.sidebarExpanded", String(sidebarExpanded));
    } catch {
      /* storage unavailable — keep the choice for this session only */
    }
  }

  // Settings modals (replace the old inline disclosures, so opening them never shifts the layout).
  let liveAdvancedOpen = $state(false);
  let fileOptionsOpen = $state(false);
  let screenAuthorized = $state(true);
  let micBlocked = $state(false);
  let permissionBusy = $state(false);
  let unlisten: UnlistenFn | undefined;
  let liveErrorUnlisten: UnlistenFn | undefined;

  const activeModel = $derived(models.find((m) => m.active));

  // Segments with actual text (a freshly-opened partial can be momentarily empty — never show a blank row).
  const liveSegments = $derived(segments.filter((s) => s.text.trim().length > 0));
  // Only label each row's source (You/Them) when both are present — otherwise the repeated tag is noise.
  const multiSource = $derived(new Set(liveSegments.map((s) => s.source)).size > 1);

  // Each mode remembers its own model — File leans accurate, Live leans real-time — persisted across
  // restarts and seeded from the per-mode recommendation on first run. The picker shows the current
  // mode's pick.
  let liveModelId = $state("");
  let fileModelId = $state("");
  // Engine + cloud selection per mode (the catalog/keys live in $lib/cloud.svelte; these are this
  // screen's current picks). The unified "Transcribe with" dropdown drives all of these.
  let fileEngine = $state<"local" | "cloud">("local");
  let fileCloudProvider = $state("");
  let fileCloudModel = $state("");
  let liveEngine = $state<"local" | "cloud">("local");
  let liveCloudProvider = $state("");
  let liveCloudModel = $state("");
  const chosenId = $derived(mode === "file" ? fileModelId : liveModelId);
  const chosenModel = $derived(models.find((m) => m.id === chosenId));

  function persistModeModels() {
    try {
      localStorage.setItem(
        "wisp.modelByMode",
        JSON.stringify({ live: liveModelId, file: fileModelId }),
      );
    } catch {
      /* storage unavailable (private mode) — keep the choice in memory for this session */
    }
  }

  // Seed an unset mode from its recommendation once models load. Never seed a blocked model (one this
  // machine can't run) — fall back to the first runnable one.
  $effect(() => {
    if (!models.length) return;
    const firstRunnable = models.find((m) => m.fit !== "blocked") ?? models[0];
    if (!liveModelId)
      liveModelId =
        models.find((m) => m.recommendedLive)?.id ??
        models.find((m) => m.active && m.fit !== "blocked")?.id ??
        firstRunnable.id;
    if (!fileModelId) fileModelId = models.find((m) => m.recommendedFile)?.id ?? liveModelId;
  });

  // Keep the backend's active model in step with the current mode's pick (so the "active" tag and
  // any immediate transcribe use the right model when you switch modes).
  $effect(() => {
    const m = models.find((x) => x.id === chosenId);
    if (m?.installed && !m.active) selectModel(chosenId);
  });

  // Ready to start once the *chosen* model is installed and this machine can actually run it. Picking
  // a not-yet-downloaded (or blocked) model must never silently run a different model — Live and File
  // both gate on this.
  const canStart = $derived(!!chosenModel?.installed && chosenModel?.fit !== "blocked");

  async function pickModel(id: string) {
    if (mode === "file") fileModelId = id;
    else liveModelId = id;
    persistModeModels();
    const m = models.find((x) => x.id === id);
    if (m?.installed) await selectModel(id); // installed → apply it as the active model
  }

  /** Make sure the backend's active model is this mode's pick before a transcribe/start. */
  async function ensureActiveModel() {
    const m = models.find((x) => x.id === chosenId);
    if (m?.installed && !m.active) await selectModel(chosenId);
  }

  // Curated model dropdown. One clear "Recommended for this machine" up top (accuracy for File,
  // real-time for Live), the user's other installed models next, and everything else under "More".
  let pickerOpen = $state(false);
  // Which top tab the open dropdown shows — On-device vs Cloud. The left column lists that tab's
  // categories (families / providers); the right column lists the selected category's models.
  let pickerTab = $state<"local" | "cloud">("local");
  // The category selected within the active tab — "local:<Family>" or "cloud:<providerId>".
  let pickerCat = $state<string>("");
  // This machine has a GPU Whisper engine if the catalog surfaced any (the backend hides them off
  // Metal); when it does, the CPU-ONNX Whisper models are strictly worse, so they sort last.
  const hasGpuWhisper = $derived(models.some((m) => m.family === "WhisperCpp"));
  const isRedundant = (m: ModelInfo) => hasGpuWhisper && m.family === "Whisper";
  const recommendedId = $derived(
    (mode === "file"
      ? models.find((m) => m.recommendedFile)
      : models.find((m) => m.recommendedLive)
    )?.id,
  );
  const recommendTag = $derived(mode === "file" ? "best accuracy" : "for this machine");

  // The pinned ★ Recommended category keys (a sentinel "family"/"provider" id the picker special-cases).
  const REC_LOCAL = "local:__rec__";
  const REC_CLOUD = "cloud:__rec__";
  // The on-device ★ Recommended set: a few best-fit models for this machine — the mode's machine pick,
  // the other mode's pick (so both Live + File picks show), and a pinned SenseVoice (the reliable
  // non-autoregressive CPU default). Deduped, runnable only.
  const recommendedLocal = $derived.by(() => {
    const picks: ModelInfo[] = [];
    const add = (m: ModelInfo | undefined) => {
      if (m && m.fit !== "blocked" && !picks.some((p) => p.id === m.id)) picks.push(m);
    };
    add(models.find((m) => (mode === "file" ? m.recommendedFile : m.recommendedLive)));
    add(models.find((m) => (mode === "file" ? m.recommendedLive : m.recommendedFile)));
    add(models.find((m) => m.family === "SenseVoice"));
    return picks;
  });

  // ── On-device categories: group local models by engine family (the left column's "On-device" rows) ─
  const FAMILY_ORDER = ["Apple", "Whisper", "SenseVoice", "Paraformer", "Parakeet", "Streaming"];
  // Map a raw engine family ("AppleSpeech"/"WhisperCpp"/…) to its user-facing category label.
  function familyLabel(family: string): string {
    if (family === "AppleSpeech") return "Apple";
    if (family === "SenseVoice") return "SenseVoice";
    if (family === "Paraformer") return "Paraformer";
    if (family === "Parakeet") return "Parakeet";
    if (family === "StreamingTransducer") return "Streaming";
    return "Whisper";
  }
  const localCategories = $derived([
    ...(recommendedLocal.length ? [{ key: REC_LOCAL, label: "Recommended", star: true }] : []),
    ...FAMILY_ORDER.filter((label) => models.some((m) => familyLabel(m.family) === label)).map(
      (label) => ({ key: `local:${label}`, label, star: false }),
    ),
  ]);
  // Models in one on-device category, best first (recommended → installed → rest), blocked last; the
  // ★ Recommended sentinel returns the curated cross-family pick instead.
  function localModelsFor(label: string): ModelInfo[] {
    if (label === "__rec__") return recommendedLocal;
    const rank = (m: ModelInfo) =>
      (m.fit === "blocked" ? 100 : 0) +
      (m.id === recommendedId ? 0 : m.installed ? 1 : isRedundant(m) ? 3 : 2);
    return models.filter((m) => familyLabel(m.family) === label).sort((a, b) => rank(a) - rank(b));
  }

  // ── Unified "Transcribe with": one dropdown listing on-device + cloud models (mode-aware) ──────────
  // The left column picks a category (family / provider); the right column lists that category's models.
  const currentEngine = $derived(mode === "file" ? fileEngine : liveEngine);
  const cloudCapability = $derived(mode === "file" ? "batch" : "streaming");
  const currentCloudProvider = $derived(mode === "file" ? fileCloudProvider : liveCloudProvider);
  const currentCloudModel = $derived(mode === "file" ? fileCloudModel : liveCloudModel);

  // Cloud providers that have at least one model runnable in this mode (streaming for Live, batch for
  // File) — the left column's "Cloud" rows.
  const runnableCloudModels = (p: (typeof cloudState.providers)[number]) =>
    p.models.filter((m) => (cloudCapability === "streaming" ? m.streaming || m.batch : m.batch));
  const cloudProviders = $derived(cloudState.providers.filter((p) => runnableCloudModels(p).length));
  // The cloud ★ Recommended set: each keyed provider's models flagged `recommended` that run in this
  // mode (streaming for Live, batch for File), across providers — so Live surfaces the realtime
  // transcribers and File the file ones. Each carries its provider for the cross-provider list.
  const recommendedCloud = $derived(
    cloudProviders.flatMap((p) =>
      p.models
        .filter(
          (m) =>
            m.recommended && (cloudCapability === "streaming" ? m.streaming || m.batch : m.batch),
        )
        .map((m) => ({ provider: p, model: m })),
    ),
  );
  const cloudCategories = $derived([
    ...(recommendedCloud.length ? [{ key: REC_CLOUD, label: "Recommended", keySet: true, star: true }] : []),
    ...cloudProviders.map((p) => ({ key: `cloud:${p.id}`, label: p.name, keySet: p.keySet, star: false })),
  ]);

  // The category that owns the current selection — the picker opens focused on it.
  const currentCat = $derived(
    currentEngine === "cloud"
      ? `cloud:${currentCloudProvider}`
      : `local:${familyLabel(chosenModel?.family ?? "")}`,
  );
  // The right column's content for the active category.
  const pickerLocalLabel = $derived(pickerCat.startsWith("local:") ? pickerCat.slice(6) : "");
  const pickerCatProvider = $derived(
    pickerCat.startsWith("cloud:") ? cloudProvider(pickerCat.slice(6)) : undefined,
  );

  // Whether a given local model / cloud option is the current selection (only one engine is active).
  const localSelected = (id: string) => currentEngine === "local" && id === chosenId;
  const cloudSelected = (providerId: string, modelId: string) =>
    currentEngine === "cloud" && providerId === currentCloudProvider && modelId === currentCloudModel;

  // The current selection's display name, for the trigger.
  const sourceName = $derived(
    currentEngine === "cloud"
      ? (cloudProvider(currentCloudProvider)?.models.find((m) => m.id === currentCloudModel)?.name ??
          "Select a model")
      : (chosenModel?.name ?? "Select a model"),
  );

  // Switch the top tab and land on a sensible category: the current selection's if it lives in this
  // tab, otherwise the tab's first category.
  function selectTab(tab: "local" | "cloud") {
    pickerTab = tab;
    const keys = (tab === "local" ? localCategories : cloudCategories).map((c) => c.key);
    pickerCat = keys.includes(currentCat) ? currentCat : (keys[0] ?? "");
  }

  // Toggle the picker; on open, focus the tab + category that own the current selection.
  function openModelPicker() {
    pickerOpen = !pickerOpen;
    if (pickerOpen) selectTab(currentEngine);
  }

  function chooseCloud(providerId: string, modelId: string) {
    pickerOpen = false;
    if (mode === "file") {
      fileEngine = "cloud";
      fileCloudProvider = providerId;
      fileCloudModel = modelId;
    } else {
      liveEngine = "cloud";
      liveCloudProvider = providerId;
      liveCloudModel = modelId;
    }
  }

  async function choose(id: string) {
    pickerOpen = false;
    if (mode === "file") fileEngine = "local";
    else liveEngine = "local";
    await pickModel(id);
  }

  // Import a user-supplied model file: the backend validates + copies it, then it appears in the
  // picker and becomes the current mode's pick. Today it accepts a Whisper GGML/GGUF .bin/.gguf.
  async function importCustom() {
    pickerOpen = false;
    error = "";
    try {
      const path = await open({
        multiple: false,
        directory: false,
        filters: [{ name: "Whisper model", extensions: ["bin", "gguf"] }],
      });
      if (typeof path !== "string") return;
      const info = await invoke<ModelInfo>("import_custom_model", { path });
      await refreshModels();
      await pickModel(info.id);
    } catch (e) {
      error = String(e);
    }
  }

  // Cloud analogue of the on-device "Import custom model" footer: jump to the global AI-models
  // settings (API keys, custom OpenAI-compatible endpoints, custom model ids) instead of importing.
  function manageCloudModels() {
    pickerOpen = false;
    openEndpointsModal();
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

  // Quick mic/system toggles for the Live bar — the "You" (mic) and "Them" (system) chips. The
  // specific-device choice stays in Advanced; these just flip each stream on/off with a sensible
  // default (mic → system default; system → one-click system audio).
  const micOn = $derived(micDevice !== micOffId);
  const systemOn = $derived(!!systemDevice);
  // Both streams are live (You + Them), so each row should say which side it came from — even before
  // the quieter side has produced its first line (when `multiSource` alone wouldn't fire yet).
  const dualStream = $derived(micOn && systemOn);

  function toggleMic() {
    micDevice = micOn ? micOffId : "";
    applyDevices();
  }
  function toggleSystem() {
    systemDevice = systemOn ? "" : systemAudioId;
    applyDevices();
  }

  // While a live session runs, the same chips mute/unmute the streams that were started (capture keeps
  // running; a muted stream is silenced so it stops transcribing). The per-stream mute state is
  // tracked here and reset at Start.
  let liveMicMuted = $state(false);
  let liveSystemMuted = $state(false);

  async function setStreamMuted(kind: "mic" | "system", muted: boolean) {
    try {
      await invoke("set_stream_muted", { kind, muted });
    } catch (e) {
      error = String(e);
    }
  }

  // You/Them are always shown — pre-start each toggles whether that audio is captured; during a live
  // session each mutes/unmutes its stream so you can drop or add a source on the fly.
  const youShown = true;
  const youOn = $derived(running ? !liveMicMuted : micOn);
  function youClick() {
    if (running) {
      liveMicMuted = !liveMicMuted;
      setStreamMuted("mic", liveMicMuted);
    } else {
      toggleMic();
    }
  }

  const themShown = true;
  const themOn = $derived(running ? !liveSystemMuted : systemOn);
  function themClick() {
    if (running) {
      liveSystemMuted = !liveSystemMuted;
      setStreamMuted("system", liveSystemMuted);
    } else {
      toggleSystem();
    }
  }

  async function applyLanguage() {
    try {
      await invoke("set_language", { language });
    } catch (e) {
      error = String(e);
    }
  }

  async function applyDenoise() {
    try {
      await invoke("set_denoise", { denoiser: liveDenoiser });
    } catch (e) {
      error = String(e);
    }
  }

  async function applyLiveDiarize() {
    try {
      await invoke("set_live_diarize", { model: liveDiarize ? diarizeId : null });
    } catch (e) {
      error = String(e);
    }
  }

  async function applyLiveDecode() {
    try {
      await invoke("set_live_decode", { prompt: livePrompt.trim(), accurate: liveAccurate });
    } catch (e) {
      error = String(e);
    }
  }

  function sourceLabel(source: string): string {
    if (source === "Microphone") return "mic";
    if (source === "System") return "system";
    return source.toLowerCase();
  }

  // Who a live row's audio came from, in meeting terms: your mic is "You", system audio is "Them".
  // The per-speaker number (when "Identify speakers" is on) is shown separately, tinted, beside it.
  function whoLabel(source: string): string {
    if (source === "Microphone") return "You";
    if (source === "System") return "Them";
    return sourceLabel(source);
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

  async function downloadCoreml(id: string) {
    error = "";
    downloadingCoreml = id;
    const m = models.find((x) => x.id === id);
    coremlProgress = { downloaded: 0, total: m?.coremlSizeBytes ?? 0 };
    try {
      await invoke("download_coreml", { id });
      await refreshModels();
    } catch (e) {
      error = String(e);
    } finally {
      downloadingCoreml = null;
      coremlProgress = null;
    }
  }

  const coremlPct = $derived(
    coremlProgress && coremlProgress.total > 0
      ? Math.min(100, Math.round((coremlProgress.downloaded / coremlProgress.total) * 100))
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
      // The moment Stop is pressed (or after it lands), ignore any further emissions so a session that
      // is tearing down — slowly, if a native handle wedged — can't keep adding rows. The already-shown
      // transcript stays for export; it just stops growing.
      if (!running || stopping) return;
      // Upsert by (source, id): a provisional partial creates a row, later partials of the same
      // utterance update it in place, and the final replaces it (dropping the .partial styling).
      const incoming = event.payload;
      const i = segments.findIndex((s) => s.id === incoming.id && s.source === incoming.source);
      if (i === -1) {
        segments = [...segments, incoming];
      } else {
        const next = segments.slice();
        next[i] = incoming;
        segments = next;
      }
    });
    // A cloud-streaming error (bad key/model, server error, dropped connection) — surface it as a
    // notice rather than failing silently.
    liveErrorUnlisten = await listen<string>("live://error", (event) => {
      liveNotice = `Cloud error: ${event.payload}`;
    });
  }

  async function ensureProgressListener() {
    if (progressUnlisten) return;
    progressUnlisten = await listen<{ id: string; downloaded: number; total: number }>(
      "download://progress",
      (event) => {
        const { id, downloaded, total } = event.payload;
        if (id === downloading) {
          downloadProgress = { downloaded, total };
        } else if (downloadingCoreml && id === `coreml:${downloadingCoreml}`) {
          coremlProgress = { downloaded, total };
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
    liveNotice = "";
    if (liveEngine === "cloud") {
      if (!liveCloudReady) {
        error = "Pick a cloud model and save its API key first.";
        return;
      }
    } else if (liveDiarize && !diarizeChosen?.installed) {
      error = "Download the speaker model first.";
      return;
    }
    // Fresh session = fresh feed: the backend resets its segment ids to 0 and clears the export buffer
    // per session, so a lingering previous transcript would collide by (source, id) and interleave two
    // different time bases. Clear it here (export the old one first if you need it).
    segments = [];
    starting = true;
    slowStart = false;
    const slowTimer = setTimeout(() => (slowStart = true), 4000);
    try {
      // Local-only prep (the cloud engine self-segments and denoises server-side); the device and
      // language selections apply to both.
      if (liveEngine === "local") await ensureActiveModel();
      await applyDevices();
      await applyLanguage();
      if (liveEngine === "local") {
        await applyDenoise();
        await applyLiveDiarize();
        await applyLiveDecode();
      }
      await ensureListener();
      const notice = await invoke<string | null>("start_session", {
        options: {
          engine: liveEngine,
          cloudProvider: liveEngine === "cloud" ? liveCloudProvider : null,
          cloudModel: liveEngine === "cloud" ? liveCloudModel : null,
          params: liveEngine === "cloud" ? changedParamValues(liveParams, liveParamSpecs) : {},
          // Always tap the live audio for the realtime assist. The tap is a cheap drop-oldest tee that
          // is never drained unless the assist runs, so the realtime assist can be started at any point
          // during a live session — no need to have "armed" it before Start.
          assist: true,
        },
      });
      liveNotice = notice ?? "";
      running = true;
      // Both streams start unmuted; the live You/Them chips flip these mid-session.
      liveMicMuted = false;
      liveSystemMuted = false;
      // Capture started, so the permissions it needed are granted — clear any stale prompts
      // (macOS can report a stale status to a running process after a Settings change).
      screenAuthorized = true;
      micBlocked = false;
    } catch (e) {
      // If a session is actually already running (e.g. after a reload), reflect that instead of
      // showing the error.
      await syncRunning();
      error = running ? "" : String(e);
    } finally {
      clearTimeout(slowTimer);
      starting = false;
      slowStart = false;
    }
  }

  async function stop() {
    stopping = true;
    try {
      await invoke("stop_session");
    } catch (e) {
      error = String(e);
    } finally {
      stopping = false;
    }
    running = false;
    liveNotice = "";
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
  // Engines emit one segment per utterance; a wall of short lines reads poorly. Merge consecutive
  // segments into paragraphs for display (mirrors wisp-core's group_paragraphs: same rules so the
  // on-screen view matches the TXT export).
  type FileParagraph = { id: number; startMs: number; speaker: number | null; text: string };
  const PARAGRAPH_GAP_MS = 1500;
  const MAX_PARAGRAPH_CHARS = 240;
  function joinParagraphText(prev: string, next: string): string {
    return /^[A-Za-z0-9]/.test(next) ? `${prev} ${next}` : `${prev}${next}`;
  }
  function groupParagraphs(segs: Segment[]): FileParagraph[] {
    const paras: FileParagraph[] = [];
    let prevEnd = 0;
    for (const s of segs) {
      const text = s.text.trim();
      if (!text) continue;
      const cur = paras[paras.length - 1];
      const fits =
        cur &&
        cur.speaker === s.speaker &&
        s.startMs - prevEnd <= PARAGRAPH_GAP_MS &&
        [...cur.text].length < MAX_PARAGRAPH_CHARS;
      if (fits) cur.text = joinParagraphText(cur.text, text);
      else paras.push({ id: s.id, startMs: s.startMs, speaker: s.speaker, text });
      prevEnd = s.endMs;
    }
    return paras;
  }
  const fileParagraphs = $derived(groupParagraphs(fileSegments));
  let fileName = $state("");
  // The model this run is transcribing with, captured at submit so the running view shows it.
  let fileModelLabel = $state("");
  let fileTranscribing = $state(false);
  // True from when Cancel is clicked until the backend confirms the run stopped (its file://done).
  let fileCancelling = $state(false);
  // Decode progress 0–100; 0 means the engine hasn't reported yet (bar shows indeterminate).
  let fileProgress = $state(0);
  // Current pipeline phase ("decoding"/"reducing noise"/"transcribing"), shown while no % is
  // available so the bar isn't a content-free sweep.
  let fileStage = $state("");
  // Accurate (beam search) vs Fast (greedy) decoding. Files default to Accurate.
  let fileAccurate = $state(true);
  // Timeline (timestamps) is opt-in: off = most accurate plain text; on = timed for SRT/VTT.
  let fileTimestamps = $state(false);
  let fileHasTimestamps = $state(false);
  // Optional context primer (names, jargon, domain terms) that biases the decoder's spelling.
  let filePrompt = $state("");
  // Skip non-speech (silence/music) before decoding, opt-in: stops hallucinated words in the gaps.
  let fileGate = $state(false);
  // Denoiser backend id (null = off, "rnnoise" = light built-in, else a downloadable model id).
  let fileDenoiser = $state<string | null>(null);
  // Downloadable denoiser models (e.g. GTCRN), loaded on demand like the speaker models.
  let denoiseModels = $state<ModelInfo[]>([]);
  const denoiseModelId = $derived(denoiseModels[0]?.id ?? "denoise-gtcrn");
  const denoiseChosen = $derived(denoiseModels.find((m) => m.id === fileDenoiser));
  // Speaker diarization (who-said-what), opt-in. The models load and download on demand.
  let diarizeModels = $state<ModelInfo[]>([]);
  let diarizeOn = $state(false);
  let diarizeId = $state("");
  const diarizeChosen = $derived(diarizeModels.find((m) => m.id === diarizeId));
  $effect(() => {
    if (!diarizeId && diarizeModels.length) diarizeId = diarizeModels[0].id;
  });

  // Engine choice per mode: the active on-device model, or a cloud provider/model. The provider

  const liveProv = $derived(cloudProvider(liveCloudProvider));
  const liveMod = $derived(liveProv?.models.find((m) => m.id === liveCloudModel));
  const liveCloudReady = $derived(cloudReady(liveCloudProvider, liveCloudModel, "streaming"));
  // The running header's model label: the cloud provider/model in cloud mode, else the on-device one.
  const liveRunningLabel = $derived(
    liveEngine === "cloud"
      ? `${liveProv?.name ?? "Cloud"} · ${liveMod?.name ?? liveCloudModel}`
      : (activeModel?.name ?? "Model"),
  );

  // Generic advanced parameters for the selected streaming provider: fetch its specs, seed values
  // from saved overrides (or smart defaults), and persist edits. Driven entirely by <ParamsPanel>.
  let liveParamSpecs = $state<ParamSpec[]>([]);
  let liveParams = $state<Record<string, ParamValue>>({});
  let liveParamsOpen = $state(false);

  // Live AI assist: a right-side drawer running the same LLM tasks over the live transcript (finals
  // only), on demand. Auto-rolling refresh is a later refinement.
  let liveAssistOpen = $state(false);
  // Whether the transcript's compact "Export ▾" menu is open (collapses MD/TXT/SRT into one control).
  let exportMenuOpen = $state(false);
  let liveBodyEl = $state<HTMLElement | null>(null);
  const ASSIST_MIN = 320;
  const TRANSCRIPT_MIN = 360;
  let assistWidth = $state(Math.max(ASSIST_MIN, Number(localStorage.getItem("wisp.assistWidth")) || 440));
  // The transcript handed to the AI assist (not the on-screen one) — formatted conversationally so the
  // model reasons about turns: mic = "Me", system = "Them", plus the live diarizer's speaker number on
  // the meeting side (where multiple remote participants matter; mic is always you).
  const assistWho = (s: Segment): string => {
    if (s.source === "Microphone") return "Me";
    if (s.source === "System") return s.speaker !== null ? `Them (Speaker ${s.speaker + 1})` : "Them";
    return sourceLabel(s.source);
  };
  const liveTranscriptText = $derived(
    segments
      .filter((s) => s.isFinal)
      // Chronological by start time, not finalization order: mic and system are independent pipelines,
      // so a late-finalizing earlier utterance must still land in its real place — both so the LLM reads
      // turns in order and so the assist's summary-buffer sees a stable, append-only prefix to index into.
      .slice()
      .sort((a, b) => a.startMs - b.startMs)
      .map((s) => `[${fmtTime(s.startMs)}] ${assistWho(s)}: ${s.text}`)
      .join("\n"),
  );

  // Drag the splitter to resize the assist panel: pulling left widens it. Clamped so neither side
  // gets too thin; the chosen width persists. The whole pane grows with the window (.app widens).
  function startAssistResize(e: MouseEvent) {
    e.preventDefault();
    const startX = e.clientX;
    const startW = assistWidth;
    const maxW = liveBodyEl ? Math.max(ASSIST_MIN, liveBodyEl.clientWidth - TRANSCRIPT_MIN) : 9999;
    const onMove = (ev: MouseEvent) => {
      assistWidth = Math.min(maxW, Math.max(ASSIST_MIN, startW - (ev.clientX - startX)));
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      localStorage.setItem("wisp.assistWidth", String(Math.round(assistWidth)));
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  $effect(() => {
    const provider = liveCloudProvider;
    const model = liveCloudModel;
    if (liveEngine !== "cloud" || !provider || !model) {
      liveParamSpecs = [];
      return;
    }
    // A realtime model exposes its streaming knobs; a batch model run segment-batch in Live exposes
    // the batch knobs (temperature, …). Pick by what the selected model can do.
    const streams = cloudProvider(provider)?.models.find((m) => m.id === model)?.streaming ?? false;
    (streams ? streamingParams(provider, model) : batchParams(provider, model)).then((specs) => {
      liveParamSpecs = specs;
      liveParams = { ...defaultParamValues(specs), ...loadParamValues(provider, model) };
    });
  });

  $effect(() => {
    if (liveEngine === "cloud" && liveCloudProvider && liveParamSpecs.length) {
      saveParamValues(liveCloudProvider, liveCloudModel, liveParams);
    }
  });

  const fileProv = $derived(cloudProvider(fileCloudProvider));
  const fileCloudReady = $derived(cloudReady(fileCloudProvider, fileCloudModel, "batch"));
  const fileMod = $derived(fileProv?.models.find((m) => m.id === fileCloudModel));
  // A cloud model that returns its own speaker labels — local diarization must not run on top of it.
  const fileModelSelfDiarizes = $derived(fileEngine === "cloud" && !!fileMod?.diarizes);

  // File (batch) advanced parameters for the selected cloud provider/model — same machinery as live,
  // namespaced "file" so a model used in both surfaces keeps separate values.
  let fileParamSpecs = $state<ParamSpec[]>([]);
  let fileParams = $state<Record<string, ParamValue>>({});
  let fileParamsOpen = $state(false);

  $effect(() => {
    const provider = fileCloudProvider;
    const model = fileCloudModel;
    if (fileEngine !== "cloud" || !provider || !model) {
      fileParamSpecs = [];
      return;
    }
    batchParams(provider, model).then((specs) => {
      fileParamSpecs = specs;
      fileParams = { ...defaultParamValues(specs), ...loadParamValues(provider, model, "file") };
    });
  });

  $effect(() => {
    if (fileEngine === "cloud" && fileCloudProvider && fileParamSpecs.length) {
      saveParamValues(fileCloudProvider, fileCloudModel, fileParams, "file");
    }
  });

  // Whether the current File engine is ready to accept a file (local model installed, or cloud set).
  const fileReady = $derived(fileEngine === "cloud" ? fileCloudReady : canStart);

  let dragOver = $state(false);
  let fileListeners: UnlistenFn[] = [];
  let dropUnlisten: UnlistenFn | undefined;

  // A distinct colour per speaker (cycled), and a 1-based label matching the export.
  const SPEAKER_COLORS = ["#c96442", "#3f7e6b", "#6a5acd", "#b58a2e", "#9c4d6b", "#4a7aa8"];
  const speakerColor = (n: number) => SPEAKER_COLORS[n % SPEAKER_COLORS.length];
  const speakerLabel = (n: number) => `Speaker ${n + 1}`;

  // Which File results tab is showing, and the transcript assembled as plain text for the AI tasks.
  let fileTab = $state<"transcript" | "ai">("transcript");
  const fileTranscriptText = $derived(
    fileParagraphs
      .map((p) => {
        const time = fileHasTimestamps ? `[${fmtTime(p.startMs)}] ` : "";
        const who = p.speaker !== null ? `${speakerLabel(p.speaker)}: ` : "";
        return `${time}${who}${p.text}`;
      })
      .join("\n"),
  );
  // Short segmented-control label for a diarization model (last "·" segment of its display name).
  const diarizeShortName = (m: ModelInfo) => (m.name.split("·").pop() ?? m.name).trim();

  async function refreshDiarizeModels() {
    try {
      diarizeModels = await invoke<ModelInfo[]>("list_diarization_models");
    } catch (e) {
      error = String(e);
    }
  }

  // Download a diarization model. Like `download` but it never becomes the active ASR model.
  async function downloadDiarize(id: string) {
    error = "";
    downloadFailed = null;
    downloading = id;
    downloadProgress = { downloaded: 0, total: diarizeModels.find((m) => m.id === id)?.sizeBytes ?? 0 };
    try {
      await invoke("download_model", { id });
      await refreshDiarizeModels();
    } catch (e) {
      downloadFailed = id;
      error = String(e);
    } finally {
      downloading = null;
      downloadProgress = null;
    }
  }

  async function refreshDenoiseModels() {
    try {
      denoiseModels = await invoke<ModelInfo[]>("list_denoise_models");
    } catch (e) {
      error = String(e);
    }
  }

  // Download a denoiser model (e.g. GTCRN). Like `downloadDiarize`; never the active ASR model.
  async function downloadDenoise(id: string) {
    error = "";
    downloadFailed = null;
    downloading = id;
    downloadProgress = { downloaded: 0, total: denoiseModels.find((m) => m.id === id)?.sizeBytes ?? 0 };
    try {
      await invoke("download_model", { id });
      await refreshDenoiseModels();
    } catch (e) {
      downloadFailed = id;
      error = String(e);
    } finally {
      downloading = null;
      downloadProgress = null;
    }
  }

  async function transcribeFile(path: string) {
    if (fileTranscribing) return;
    if (fileEngine === "local" && !chosenModel?.installed) {
      error = `Download ${chosenModel?.name ?? "the model"} first.`;
      return;
    }
    if (fileEngine === "cloud" && !fileCloudReady) {
      error = fileProv?.keySet
        ? "Choose a cloud model."
        : `Add your ${fileProv?.name ?? "provider"} API key first.`;
      return;
    }
    if (diarizeOn && !fileModelSelfDiarizes && !diarizeChosen?.installed) {
      error = "Download the speaker model first.";
      return;
    }
    if (fileDenoiser === denoiseModelId && !denoiseChosen?.installed) {
      error = "Download the noise-reduction model first.";
      return;
    }
    error = "";
    fileSegments = [];
    fileTab = "transcript";
    fileProgress = 0;
    fileStage = "";
    fileName = path.split(/[\\/]/).pop() ?? path;
    fileModelLabel =
      fileEngine === "cloud"
        ? `${fileProv?.name ?? "Cloud"} · ${fileMod?.name ?? fileCloudModel}`
        : (chosenModel?.name ?? "On-device model");
    fileHasTimestamps = fileTimestamps;
    fileTranscribing = true;
    try {
      if (fileEngine === "local") await ensureActiveModel();
      await invoke("transcribe_file", {
        path,
        options: {
          timestamps: fileTimestamps,
          accurate: fileAccurate,
          prompt: filePrompt.trim(),
          diarizeModel: diarizeOn && !fileModelSelfDiarizes ? diarizeId : null,
          gateSpeech: fileGate,
          denoiser: fileDenoiser,
          engine: fileEngine,
          cloudProvider: fileEngine === "cloud" ? fileCloudProvider : null,
          cloudModel: fileEngine === "cloud" ? fileCloudModel : null,
          params: fileEngine === "cloud" ? changedParamValues(fileParams, fileParamSpecs) : {},
        },
      });
    } catch (e) {
      error = String(e);
      fileTranscribing = false;
    }
  }

  function resetFile() {
    fileSegments = [];
    fileName = "";
  }

  // Stop the running file transcription at the next window boundary; the backend drops the partial and
  // emits file://done, which clears the transcribing + cancelling state.
  async function cancelFile() {
    fileCancelling = true;
    try {
      await invoke("cancel_file_transcription");
    } catch (e) {
      error = String(e);
      fileCancelling = false;
    }
  }

  async function pickFile() {
    if (fileEngine === "local" && !chosenModel?.installed) {
      error = `Download ${chosenModel?.name ?? "the model"} first.`;
      return;
    }
    if (fileEngine === "cloud" && !fileCloudReady) {
      error = fileProv?.keySet
        ? "Choose a cloud model."
        : `Add your ${fileProv?.name ?? "provider"} API key first.`;
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

  // Meeting metadata for a Markdown export — the bits the Rust formatter can't derive from the
  // transcript (date, model, language). The backend appends nothing; the serializer derives duration +
  // participants from the segments themselves.
  function meetingMeta(title: string) {
    return {
      title,
      date: new Date().toISOString().slice(0, 10),
      engine: activeModel?.name,
      language: language || undefined,
    };
  }

  // Saves `source` ("file" or "live") as `format`, prompting for a destination. Markdown also carries
  // the meeting metadata; the subtitle/text formats ignore it.
  async function runExport(format: string, source: "file" | "live", base: string, title: string) {
    try {
      const dest = await save({
        defaultPath: `${base}.${format}`,
        filters: [{ name: format.toUpperCase(), extensions: [format] }],
      });
      if (typeof dest !== "string") return;
      const args: Record<string, unknown> = { format, dest, source };
      if (format === "md") args.meta = meetingMeta(title);
      await invoke("export_transcript", args);
    } catch (e) {
      error = String(e);
    }
  }

  async function exportFile(format: string) {
    if (!fileSegments.length) return;
    const base = (fileName || "transcript").replace(/\.[^.]+$/, "");
    await runExport(format, "file", base, base);
  }

  // Export the live meeting transcript (mic + system finals retained by the backend this session).
  async function exportLive(format: string) {
    if (!segments.length) return;
    const date = new Date().toISOString().slice(0, 10);
    await runExport(format, "live", `meeting-${date}`, `Meeting ${date}`);
  }

  // Close the compact Export menu, then export in the chosen format.
  function exportPick(format: string) {
    exportMenuOpen = false;
    exportLive(format);
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
      await listen<number>("file://progress", (e) => {
        fileProgress = e.payload;
      }),
    );
    fileListeners.push(
      await listen<string>("file://stage", (e) => {
        fileStage = e.payload;
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
        fileCancelling = false;
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
    // Restore each mode's saved model from the previous session (the seed effect fills any gaps).
    try {
      const saved = JSON.parse(localStorage.getItem("wisp.modelByMode") || "{}");
      if (saved.live) liveModelId = saved.live;
      if (saved.file) fileModelId = saved.file;
      sidebarExpanded = localStorage.getItem("wisp.sidebarExpanded") === "true";
    } catch {
      /* ignore unreadable storage */
    }
    refreshModels();
    refreshCloud();
    refreshDiarizeModels();
    refreshDenoiseModels();
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
    liveErrorUnlisten?.();
    progressUnlisten?.();
    fileListeners.forEach((u) => u());
    dropUnlisten?.();
  });
</script>

<main
  class="app"
  class:live={mode === "live"}
  class:wide={mode === "live" && liveAssistOpen && (running || segments.length)}
>
  <nav class="rail" class:expanded={sidebarExpanded}>
    <!-- Collapse/expand handle: sits on the divider line, revealed on hover of the rail. -->
    <button
      class="rail-edge"
      onclick={toggleSidebar}
      title={sidebarExpanded ? "Collapse" : "Expand"}
      aria-label={sidebarExpanded ? "Collapse sidebar" : "Expand sidebar"}
    >
      <svg
        class="rail-chevron"
        viewBox="0 0 16 16"
        fill="none"
        stroke="currentColor"
        stroke-width="1.8"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M6 4l4 4-4 4" />
      </svg>
    </button>

    <div class="rail-nav">
      <button
        class="rail-item"
        class:active={mode === "live"}
        onclick={() => (mode = "live")}
        title="Live"
      >
        <svg class="rail-ico" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <circle cx="12" cy="12" r="6" />
        </svg>
        <span class="rail-label">Live</span>
      </button>

      <button
        class="rail-item"
        class:active={mode === "file"}
        onclick={() => (mode = "file")}
        title="File"
      >
        <svg
          class="rail-ico"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.7"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M7 3h7l4 4v14H7z" /><path d="M14 3v4h4" />
        </svg>
        <span class="rail-label">File</span>
      </button>
    </div>

    <div class="rail-spacer"></div>

    <button
      class="rail-item"
      onclick={() => (cloudState.endpointsOpen = true)}
      title="Settings"
      aria-label="Settings"
    >
      <svg
        class="rail-ico"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.7"
        stroke-linecap="round"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="3.2" />
        <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" />
      </svg>
      <span class="rail-label">Settings</span>
    </button>
  </nav>

  <Settings bind:open={cloudState.endpointsOpen} />

  <div class="workspace">

  {#snippet modelPicker()}
    {#if models.length}
      <div class="picker">
        <button
          class="picker-trigger"
          class:open={pickerOpen}
          onclick={openModelPicker}
          disabled={downloading !== null}
        >
          <span class="picker-label">{sourceName}</span>
          <span class="picker-caret"></span>
        </button>
        {#if pickerOpen}
          <button class="picker-backdrop" aria-label="Close" onclick={() => (pickerOpen = false)}
          ></button>
          <!-- Tabs (On-device | Cloud) up top; below, categories (left) → that category's models (right). -->
          <div class="picker-menu wide" transition:fly={{ y: -6, duration: 120 }}>
            <div class="picker-tabs">
              <button class:active={pickerTab === "local"} onclick={() => selectTab("local")}>
                On-device
              </button>
              <button class:active={pickerTab === "cloud"} onclick={() => selectTab("cloud")}>
                Cloud
              </button>
            </div>
            <div class="picker-panes">
              <div class="picker-cats">
                {#if pickerTab === "local"}
                  {#each localCategories as c (c.key)}
                    <button
                      class="picker-cat"
                      class:active={pickerCat === c.key}
                      onclick={() => (pickerCat = c.key)}
                    >
                      {#if c.star}<span class="picker-cat-star">✦</span>{/if}
                      <span class="picker-cat-name">{c.label}</span>
                    </button>
                  {/each}
                {:else}
                  {#each cloudCategories as c (c.key)}
                    <button
                      class="picker-cat"
                      class:active={pickerCat === c.key}
                      onclick={() => (pickerCat = c.key)}
                    >
                      {#if c.star}<span class="picker-cat-star">✦</span>{/if}
                      <span class="picker-cat-name">{c.label}</span>
                      {#if !c.keySet}<span class="picker-cat-dot" title="API key needed"></span>{/if}
                    </button>
                  {/each}
                {/if}
              </div>
              <div class="picker-detail">
                {#if pickerTab === "local"}
                  {#each localModelsFor(pickerLocalLabel) as m (m.id)}
                    {#if m.fit === "blocked"}
                      <div class="picker-opt blocked" title={m.fitReason ?? ""}>
                        <span class="picker-opt-name">{m.name}</span>
                        <span class="picker-opt-note">{m.fitReason}</span>
                      </div>
                    {:else}
                      <button
                        class="picker-opt"
                        class:sel={localSelected(m.id)}
                        onclick={() => choose(m.id)}
                      >
                        <span class="picker-opt-name">{m.name}</span>
                        {#if m.id === recommendedId}<span class="picker-tag rec">{recommendTag}</span>{/if}
                        {#if m.active}<span class="picker-tag">active</span>
                        {:else if !m.installed}<span class="picker-opt-size">{fmtSize(m.sizeBytes)}</span
                          >{/if}
                        {#if m.fit === "heavy"}<span class="picker-opt-note">{m.fitReason}</span>{/if}
                      </button>
                    {/if}
                  {/each}
                {:else if pickerCat === REC_CLOUD}
                  {#each recommendedCloud as { provider: p, model: m } (p.id + ":" + m.id)}
                    <button
                      class="picker-opt"
                      class:sel={cloudSelected(p.id, m.id)}
                      onclick={() => chooseCloud(p.id, m.id)}
                    >
                      <span class="picker-opt-name">{p.name} · {m.name}</span>
                      {#if !p.keySet}<span class="picker-opt-note">needs key</span>{/if}
                    </button>
                  {/each}
                {:else if pickerCatProvider}
                  {#if !pickerCatProvider.keySet}
                    <div class="picker-detail-hint">
                      Add an API key in Settings → AI models to use {pickerCatProvider.name}.
                    </div>
                  {/if}
                  {#each runnableCloudModels(pickerCatProvider) as m (m.id)}
                    <button
                      class="picker-opt"
                      class:sel={cloudSelected(pickerCatProvider.id, m.id)}
                      onclick={() => chooseCloud(pickerCatProvider.id, m.id)}
                    >
                      <span class="picker-opt-name">{m.name}</span>
                      {#if m.recommended}<span class="picker-tag rec">recommended</span>{/if}
                    </button>
                  {/each}
                {:else}
                  <div class="picker-detail-hint">
                    No cloud models yet — add a provider &amp; key in Settings → AI models.
                  </div>
                {/if}
              </div>
            </div>
            {#if pickerTab === "local"}
              <!-- Import a user model file (Whisper GGML/GGUF) — pinned across the whole dropdown. -->
              <button class="picker-custom" onclick={importCustom}>
                <span class="picker-custom-main">
                  <span class="picker-custom-icon" aria-hidden="true"></span>
                  <span class="picker-custom-label">Import custom model…</span>
                </span>
                <span class="picker-custom-hint">.bin / .gguf · Whisper GGML/GGUF</span>
              </button>
            {:else}
              <!-- Cloud analogue of Import — jump to the AI-models settings (keys, endpoints, models). -->
              <button class="picker-custom" onclick={manageCloudModels}>
                <span class="picker-custom-main">
                  <span class="picker-custom-icon cog" aria-hidden="true"></span>
                  <span class="picker-custom-label">Manage models in Settings…</span>
                </span>
                <span class="picker-custom-hint">API keys · endpoints · custom models</span>
              </button>
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
          <span class="active-model">{liveRunningLabel}</span>
        {:else}
          <div class="engine-group">
            <span class="source-prefix">Transcribe with</span>
            {@render modelPicker()}
          </div>
        {/if}
        <!-- You/Them: each pill toggles whether that audio is captured for transcription. The icon goes
             slashed + grey when off (like a muted mic/speaker), accent + solid when on — read at a glance. -->
        <div class="audio-chips">
          {#if youShown}
            <button
              class="audio-chip"
              class:on={youOn}
              onclick={youClick}
              title={youOn
                ? running
                  ? "You (your mic) is on — click to mute"
                  : "You (your mic) is on — click to exclude from transcription"
                : running
                  ? "You (your mic) is muted — click to unmute"
                  : "You (your mic) is off — click to include in transcription"}
            >
              {#if youOn}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true">
                  <rect x="9" y="3" width="6" height="11" rx="3" />
                  <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
                </svg>
              {:else}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true">
                  <rect x="9" y="3" width="6" height="11" rx="3" />
                  <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
                  <path d="M4 3l16 18" />
                </svg>
              {/if}
              <span class="chip-label">You</span>
            </button>
          {/if}
          {#if themShown}
            <button
              class="audio-chip"
              class:on={themOn}
              onclick={themClick}
              title={themOn
                ? running
                  ? "Them (system audio) is on — click to mute"
                  : "Them (system audio) is on — click to exclude from transcription"
                : running
                  ? "Them (system audio) is muted — click to unmute"
                  : "Them (system audio) is off — click to include in transcription"}
            >
              {#if themOn}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true">
                  <path d="M4 9v6h4l5 4V5L8 9H4z" />
                  <path d="M16 9a4 4 0 0 1 0 6" />
                </svg>
              {:else}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true">
                  <path d="M4 9v6h4l5 4V5L8 9H4z" />
                  <path d="M16 10l4 4M20 10l-4 4" />
                </svg>
              {/if}
              <span class="chip-label">Them</span>
            </button>
          {/if}
        </div>
        <span class="status" class:rec={running}>
          <span class="status-dot"></span>{running
            ? `Recording · ${fmtTime(elapsedMs)}`
            : liveEngine === "cloud"
              ? liveCloudReady
                ? "ready"
                : "key needed"
              : canStart
                ? "ready"
                : "no model"}
        </span>
      </div>

      {#if liveNotice}
        <div class="live-notice" role="status">
          <span>{liveNotice}</span>
          <button class="live-notice-x" aria-label="Dismiss" onclick={() => (liveNotice = "")}>×</button>
        </div>
      {/if}

      {#if !running && (error || (liveEngine === "local" && (needsScreenRecording || needsMicPermission || (chosenModel && !chosenModel.installed) || (chosenModel && chosenModel.installed && chosenModel.coremlAvailable))))}
        <div class="box-aux">
          {#if liveEngine === "local" && chosenModel && chosenModel.fit === "blocked"}
            <span class="blocked-notice">⚠ {chosenModel.fitReason} — pick another model.</span>
          {:else if liveEngine === "local" && chosenModel && !chosenModel.installed}
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

          <!-- Before download, surface that whisper.cpp models also support an optional ANE boost. -->
          {#if liveEngine === "local" && chosenModel && chosenModel.fit !== "blocked" && !chosenModel.installed && chosenModel.coremlAvailable}
            <span class="coreml-hint">
              ⚡ Supports Neural Engine acceleration · optional {fmtSize(chosenModel.coremlSizeBytes)} after install
            </span>
          {/if}

          <!-- Optional Apple Neural Engine encoder for installed whisper.cpp models. -->
          {#if liveEngine === "local" && chosenModel && chosenModel.installed && chosenModel.coremlAvailable}
            {#if chosenModel.coremlInstalled}
              <span class="coreml-on">⚡ Neural Engine acceleration on</span>
            {:else if downloadingCoreml === chosenModel.id && coremlProgress}
              <div class="dl-bar">
                <div class="dl-track"><div class="dl-fill" style="width:{coremlPct}%"></div></div>
                <span class="dl-label">
                  Neural Engine · {coremlPct}% · {fmtSize(coremlProgress.downloaded)} / {fmtSize(coremlProgress.total)}
                </span>
              </div>
            {:else}
              <button
                class="btn outline"
                onclick={() => downloadCoreml(chosenModel.id)}
                disabled={downloadingCoreml !== null}
              >
                ⚡ Neural Engine boost · {fmtSize(chosenModel.coremlSizeBytes)}
              </button>
            {/if}
          {/if}

          {#if liveEngine === "local" && needsScreenRecording}
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

          {#if liveEngine === "local" && needsMicPermission}
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
            <div class="notice error">
              <span class="notice-msg">{error}</span>
              <button class="notice-x" aria-label="Dismiss" onclick={() => (error = "")}>×</button>
            </div>
          {/if}
        </div>
      {/if}

      {#if !running && liveEngine === "cloud"}
        <div class="box-aux">
          {#if !liveCloudReady}
            <div class="notice">
              <span class="notice-text">
                <strong>Add your {liveProv?.name ?? "provider"} API key</strong> to run live cloud
                transcription. Stored on this device only.
              </span>
              <span class="notice-actions">
                <button class="btn outline sm" onclick={openEndpointsModal}>API keys</button>
              </span>
            </div>
          {/if}

          {#if liveMod?.streaming}
            <p class="cloud-live-note">
              Cloud realtime streams audio continuously to {liveProv?.name ?? "the provider"} — it
              bills per minute and needs a stable connection.
              <button class="link-btn" onclick={openEndpointsModal}>Manage API key</button>
            </p>
          {:else}
            <p class="cloud-live-note">
              {liveProv?.name ?? "The provider"} transcribes each finished sentence — near-live, with no
              mid-sentence partials, billed per request.
              <button class="link-btn" onclick={openEndpointsModal}>Manage API key</button>
            </p>
          {/if}

          {#if liveParamSpecs.length}
            <button class="params-trigger" onclick={() => (liveParamsOpen = true)}>
              <span class="params-ico" aria-hidden="true"></span>Advanced parameters
            </button>
          {/if}
        </div>
      {/if}

      <!-- Advanced parameters as a right-side drawer (OpenAI-Studio style). -->
      {#if !running && liveEngine === "cloud" && liveParamsOpen && liveParamSpecs.length}
        <button class="drawer-scrim" aria-label="Close" onclick={() => (liveParamsOpen = false)}
        ></button>
        <aside class="drawer" transition:fly={{ x: 340, duration: 180 }}>
          <div class="drawer-head">
            <span class="drawer-title">{liveProv?.name ?? "Cloud"} parameters</span>
            <button class="drawer-x" aria-label="Close" onclick={() => (liveParamsOpen = false)}
              >×</button
            >
          </div>
          <div class="drawer-body">
            <ParamsPanel specs={liveParamSpecs} bind:values={liveParams} />
          </div>
        </aside>
      {/if}

      <!-- Live body: transcript on the left, the AI assist panel docked on the right when open. -->
      <div class="live-body" bind:this={liveBodyEl}>
        <div class="transcript-pane">
          <div class="pane-head">
            <span class="pane-title">Transcript</span>
            <span class="pane-actions">
              {#if running || liveSegments.length}
                <button
                  class="assist-launch"
                  class:on={liveAssistOpen}
                  onclick={() => (liveAssistOpen = !liveAssistOpen)}
                  title="AI Assist — live hints, notes & summary"
                >
                  <span class="spark">✦</span> Assist
                </button>
              {/if}
            </span>
          </div>
          <ul class="feed" bind:this={transcriptEl} onscroll={onTranscriptScroll}>
        {#each liveSegments as seg (seg.source + "-" + seg.id)}
          <li class:partial={!seg.isFinal} class:system={seg.source === "System"}>
            <span class="meta">
              <span class="time">{fmtTime(seg.startMs)}</span>
              {#if dualStream || multiSource}<span class="who">{whoLabel(seg.source)}</span>{/if}
            </span>
            <span class="body">
              <span class="text"
                >{#if seg.speaker !== null}<span
                    class="speaker"
                    style="--spk: {speakerColor(seg.speaker)}">{speakerLabel(seg.speaker)}</span
                  >{/if}{seg.text}</span>
              {#if seg.auxText}<span class="aux-text">{seg.auxText}</span>{/if}
            </span>
          </li>
        {/each}
        {#if running && !stopping}
          <li class="listening" aria-live="polite">
            <span class="eq" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span>
            <span class="listening-text">Listening…</span>
          </li>
        {:else if !liveSegments.length}
          <li class="empty">Pick a model, press <em>Start</em>, and speak.</li>
        {/if}
          </ul>
          {#if liveSegments.length}
            <!-- Transcript utilities live at the box's bottom-right; the top-right is Assist only. -->
            <div class="pane-foot">
              <div class="export">
                <button
                  class="export-trigger"
                  class:open={exportMenuOpen}
                  onclick={() => (exportMenuOpen = !exportMenuOpen)}
                >
                  Export<span class="export-caret"></span>
                </button>
                {#if exportMenuOpen}
                  <button
                    class="export-backdrop"
                    aria-label="Close"
                    onclick={() => (exportMenuOpen = false)}
                  ></button>
                  <div class="export-menu up" transition:fly={{ y: 4, duration: 100 }}>
                    <button onclick={() => exportPick("md")}>Markdown<span class="export-ext">.md</span></button>
                    <button onclick={() => exportPick("txt")}>Plain text<span class="export-ext">.txt</span></button>
                    <button onclick={() => exportPick("srt")}>Subtitles<span class="export-ext">.srt</span></button>
                  </div>
                {/if}
              </div>
              <button class="pane-clear" onclick={clear}>Clear</button>
            </div>
          {/if}
        </div>
        {#if liveAssistOpen && (running || segments.length)}
          <aside class="assist-panel" style:width="{assistWidth}px">
            <button
              class="assist-resize"
              aria-label="Resize assist panel"
              onmousedown={startAssistResize}
            ></button>
            <div class="assist-head">
              <span class="assist-title">✦ AI Assist</span>
              <button class="assist-x" aria-label="Close" onclick={() => (liveAssistOpen = false)}
                >×</button
              >
            </div>
            <div class="assist-body"><AiNotes transcript={liveTranscriptText} live sessionRunning={running} /></div>
          </aside>
        {/if}
      </div>

      <div class="box-foot center">
        {#if running}
          <!-- Round recorder-style transport: a red circle with a stop square + a pulsing ring. -->
          <button
            class="transport-btn stop"
            class:loading={stopping}
            onclick={stop}
            disabled={stopping}
            title="Stop transcription"
            aria-label="Stop transcription"
          >
            {#if stopping}
              <span class="spinner" aria-hidden="true"></span>
            {:else}
              <span class="ic-stop" aria-hidden="true"></span>
            {/if}
          </button>
        {:else}
          <div class="start-stack">
            <button
              class="transport-btn start"
              class:loading={starting}
              onclick={start}
              disabled={starting ||
                (liveEngine === "cloud" ? !liveCloudReady : !canStart) ||
                downloading !== null}
              title={downloading !== null
                ? "Downloading model…"
                : starting
                  ? "Connecting…"
                  : "Start transcription"}
              aria-label="Start transcription"
            >
              {#if starting}
                <span class="spinner" aria-hidden="true"></span>
              {:else}
                <span class="ic-play" aria-hidden="true"></span>
              {/if}
            </button>
            {#if starting && slowStart}
              <span class="start-hint" role="status">
                {liveEngine === "cloud" ? "Connecting…" : "Loading model — a first run can take a few seconds"}
              </span>
            {/if}
          </div>
        {/if}
      </div>
    </section>

    {#if !running}
      <button class="advanced-trigger" onclick={() => (liveAdvancedOpen = true)}>
        <svg
          class="trigger-icon"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          stroke-linecap="round"
          aria-hidden="true"
        >
          <path d="M2 5h6M11.5 5H14M2 11h2.5M8 11h6" />
          <circle cx="9.5" cy="5" r="1.6" />
          <circle cx="6" cy="11" r="1.6" />
        </svg>
        {liveEngine === "cloud" ? "Audio · devices" : "Advanced · audio, language, speakers"}
      </button>
      <Modal bind:open={liveAdvancedOpen} title={liveEngine === "cloud" ? "Audio" : "Advanced settings"}>
          <section class="modal-section">
            <span class="section-title">Audio</span>
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
            {#if liveEngine !== "cloud"}
              <div class="source-row">
                <span class="source-name">Reduce noise</span>
                <div class="seg">
                  <button
                    class:active={liveDenoiser === null}
                    onclick={() => {
                      liveDenoiser = null;
                      applyDenoise();
                    }}>Off</button
                  >
                  <button
                    class:active={liveDenoiser === "rnnoise"}
                    onclick={() => {
                      liveDenoiser = "rnnoise";
                      applyDenoise();
                    }}>Light</button
                  >
                </div>
              </div>
            {/if}
            <p class="opt-hint">
              Defaults to your mic + all system audio with <strong>echo cancellation</strong>; for
              system audio only, set Microphone to Off.{#if liveEngine !== "cloud"}
                <strong>Light</strong> is the best fit for live.{:else} Cloud denoises server-side —
                tune it under <strong>Advanced parameters</strong>.{/if}
            </p>
          </section>

          {#if liveEngine !== "cloud"}
            <section class="modal-section">
              <span class="section-title">Transcription</span>
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
              <div class="source-row">
                <span class="source-name">Mode</span>
                <div class="seg">
                  <button
                    class:active={liveAccurate}
                    onclick={() => {
                      liveAccurate = true;
                      applyLiveDecode();
                    }}>Accurate</button
                  >
                  <button
                    class:active={!liveAccurate}
                    onclick={() => {
                      liveAccurate = false;
                      applyLiveDecode();
                    }}>Fast</button
                  >
                </div>
              </div>
              <div class="field">
                <span class="field-label">Hints <em>(optional)</em></span>
                <input
                  class="prompt-input"
                  type="text"
                  bind:value={livePrompt}
                  onchange={applyLiveDecode}
                  placeholder="names, jargon, acronyms…"
                />
              </div>
              <p class="opt-hint">
                Set a <strong>Language</strong> if auto-detect is wrong (recommended for Cantonese).
                <strong>Fast</strong> keeps the lowest latency; <strong>Hints</strong> prime names &amp;
                jargon.
              </p>
            </section>
          {/if}

          {#if liveEngine !== "cloud"}
          <section class="modal-section">
            <span class="section-title">Speakers</span>
            <label class="opt-toggle">
              <input type="checkbox" bind:checked={liveDiarize} onchange={applyLiveDiarize} />
              <span>Identify speakers</span>
            </label>
            {#if liveDiarize}
              <div class="source-row">
                <span class="source-name">Model</span>
                <div class="seg">
                  {#each diarizeModels as m (m.id)}
                    <button
                      class:active={diarizeId === m.id}
                      onclick={() => {
                        diarizeId = m.id;
                        applyLiveDiarize();
                      }}>{diarizeShortName(m)}</button
                    >
                  {/each}
                </div>
              </div>
              {#if diarizeChosen && !diarizeChosen.installed}
                <button
                  class="btn outline sm dl-button"
                  onclick={() => downloadDiarize(diarizeId)}
                  disabled={downloading === diarizeId}
                >
                  {downloading === diarizeId
                    ? `Downloading… ${downloadPct}%`
                    : `Download ${fmtSize(diarizeChosen.sizeBytes)}`}
                </button>
              {/if}
            {/if}
            <p class="opt-hint">
              Labels each line by who's talking (Speaker 1, 2…). <strong>Accurate</strong> tells
              similar-sounding voices apart better.
            </p>
          </section>
          {/if}
      </Modal>
    {/if}
  {:else if mode === "file"}
    <section class="box">
      {#if error}
        <div class="notice error">
          <span class="notice-msg">{error}</span>
          <button class="notice-x" aria-label="Dismiss" onclick={() => (error = "")}>×</button>
        </div>
      {/if}
      {#if fileTranscribing || fileSegments.length}
        <div class="box-head">
          <span class="active-model"
            >{fileName || "File"}{#if fileModelLabel}<span class="file-model">
                · {fileModelLabel}</span
              >{/if}</span
          >
          <span class="status" class:live={fileTranscribing}>
            <span class="status-dot"></span>{fileTranscribing
              ? fileProgress > 0
                ? `${fileStage || "transcribing"}… ${fileProgress}%`
                : `${fileStage || "transcribing"}…`
              : "done"}
          </span>
          {#if fileTranscribing}
            <button class="file-cancel" onclick={cancelFile} disabled={fileCancelling}>
              {fileCancelling ? "Cancelling…" : "Cancel"}
            </button>
          {/if}
        </div>
        {#if fileTranscribing}
          <div class="file-progress" class:indeterminate={fileProgress === 0}>
            <div
              class="file-progress-fill"
              style:width={fileProgress > 0 ? `${fileProgress}%` : undefined}
            ></div>
          </div>
        {/if}

        {#if fileSegments.length && !fileTranscribing}
          <div class="seg file-tabs">
            <button class:active={fileTab === "transcript"} onclick={() => (fileTab = "transcript")}>
              Transcript
            </button>
            <button class:active={fileTab === "ai"} onclick={() => (fileTab = "ai")}>✦ AI Notes</button>
          </div>
        {/if}

        {#if fileTab === "ai" && fileSegments.length && !fileTranscribing}
          <div class="ai-pane"><AiNotes transcript={fileTranscriptText} /></div>
        {:else}
        <ul class="feed">
          {#each fileParagraphs as para (para.id)}
            <li>
              {#if fileHasTimestamps}
                <span class="meta"><span class="time">{fmtTime(para.startMs)}</span></span>
              {/if}
              <span class="text"
                >{#if para.speaker !== null}<span
                    class="speaker"
                    style="--spk: {speakerColor(para.speaker)}">{speakerLabel(para.speaker)}</span
                  >{/if}{para.text}</span>
            </li>
          {:else}
            <li class="empty">Transcribing… large files take a little while.</li>
          {/each}
        </ul>
        {/if}
        <div class="box-foot">
          <div class="export-group">
            {#if fileSegments.length && !fileTranscribing}
              <span class="export-label">Export</span>
              <button class="btn outline sm" onclick={() => exportFile("md")}>MD</button>
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
        <div class="box-head file-pick-head">
          <span class="source-prefix">Transcribe with</span>
          {@render modelPicker()}
        </div>
        {#if fileEngine === "cloud"}
          <div class="cloud-key-row">
            {#if fileProv?.keySet}
              <span class="key-ok">✓ {fileProv.name} key saved on this device</span>
              <button class="link-btn" onclick={openEndpointsModal}>Manage keys</button>
            {:else}
              <span class="key-missing">{fileProv?.name ?? "This provider"} needs your API key</span>
              <button class="btn outline sm" onclick={openEndpointsModal}>Add API key</button>
            {/if}
          </div>
        {/if}

        {#if fileEngine === "cloud" && fileParamSpecs.length}
          <button class="params-trigger file-params-trigger" onclick={() => (fileParamsOpen = true)}>
            <span class="params-ico" aria-hidden="true"></span>Advanced parameters
          </button>
        {/if}

        <!-- Advanced parameters as a right-side drawer — same affordance as the live cloud picker. -->
        {#if fileEngine === "cloud" && fileParamsOpen && fileParamSpecs.length}
          <button class="drawer-scrim" aria-label="Close" onclick={() => (fileParamsOpen = false)}
          ></button>
          <aside class="drawer" transition:fly={{ x: 340, duration: 180 }}>
            <div class="drawer-head">
              <span class="drawer-title">{fileProv?.name ?? "Cloud"} parameters</span>
              <button class="drawer-x" aria-label="Close" onclick={() => (fileParamsOpen = false)}
                >×</button
              >
            </div>
            <div class="drawer-body">
              <ParamsPanel specs={fileParamSpecs} bind:values={fileParams} />
            </div>
          </aside>
        {/if}

        {#if fileEngine === "local" && chosenModel && !chosenModel.installed}
          <div class="box-aux">
            {#if downloading === chosenModel.id && downloadProgress}
              <div class="dl-bar">
                <div class="dl-track"><div class="dl-fill" style="width:{downloadPct}%"></div></div>
                <span class="dl-label">
                  {downloadPct}% · {fmtSize(downloadProgress.downloaded)} / {fmtSize(downloadProgress.total)}
                </span>
              </div>
            {:else}
              <button
                class="btn outline"
                onclick={() => download(chosenModel.id)}
                disabled={downloading !== null}
              >
                {downloadFailed === chosenModel.id ? "Retry download" : "Download"} · {chosenModel.name}
                · {fmtSize(chosenModel.sizeBytes)}
              </button>
            {/if}
          </div>
        {/if}
        <button
          class="dropzone"
          class:over={dragOver}
          onclick={pickFile}
          disabled={!fileReady}
          aria-label="Choose a file to transcribe"
        >
          <div class="dropzone-title">Click to choose a file, or drop one here</div>
          <p class="dropzone-sub">
            {#if fileEngine === "cloud"}
              {#if fileCloudReady}
                mp3, m4a, wav, flac, mp4, mov… sent to
                <strong>{fileProv?.name} {fileMod?.name}</strong>.
              {:else if fileProv?.keySet}
                Choose a cloud model above.
              {:else}
                Add your {fileProv?.name ?? "provider"} API key to transcribe in the cloud.
              {/if}
            {:else if chosenModel?.installed}
              mp3, m4a, wav, flac, mp4, mov… transcribed locally with
              <strong>{chosenModel.name}</strong>.
            {:else}
              <strong>{chosenModel?.name}</strong> isn't downloaded yet — get it below to transcribe.
            {/if}
          </p>
        </button>
        <!-- Power options open in a modal, so showing them never reflows the drop zone. -->
        <button
          class="advanced-trigger file-options-trigger"
          onclick={() => (fileOptionsOpen = true)}
        >
          <svg
            class="trigger-icon"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            aria-hidden="true"
          >
            <path d="M2 5h6M11.5 5H14M2 11h2.5M8 11h6" />
            <circle cx="9.5" cy="5" r="1.6" />
            <circle cx="6" cy="11" r="1.6" />
          </svg>
          {fileEngine === "cloud" ? "Options · hints, speakers" : "Options · accuracy, hints, speakers"}
        </button>
        <Modal bind:open={fileOptionsOpen} title="Options">
          <section class="modal-section">
            <span class="section-title">Audio</span>
            <div class="opt-row">
              <span class="opt-label">Reduce noise</span>
              <div class="seg">
                <button class:active={fileDenoiser === null} onclick={() => (fileDenoiser = null)}>Off</button>
                <button
                  class:active={fileDenoiser === "rnnoise"}
                  onclick={() => (fileDenoiser = "rnnoise")}>Light</button
                >
                <button
                  class:active={fileDenoiser === denoiseModelId}
                  onclick={() => (fileDenoiser = denoiseModelId)}>Balanced</button
                >
              </div>
            </div>
            {#if denoiseChosen && !denoiseChosen.installed}
              <button
                class="btn outline sm dl-button"
                onclick={() => downloadDenoise(denoiseModelId)}
                disabled={downloading === denoiseModelId}
              >
                {downloading === denoiseModelId
                  ? `Downloading… ${downloadPct}%`
                  : `Download ${fmtSize(denoiseChosen.sizeBytes)}`}
              </button>
            {/if}
            <label class="opt-toggle">
              <input type="checkbox" bind:checked={fileGate} />
              <span>Skip silence &amp; music</span>
            </label>
            <p class="opt-hint">
              Cleans background noise, and drops long non-speech so the model can't invent words in
              the gaps. Leave off for clean recordings.
            </p>
          </section>

          <section class="modal-section">
            <span class="section-title">Transcription</span>
            {#if fileEngine !== "cloud"}
              <!-- Beam vs greedy is a local-decoder choice; cloud models decode server-side. -->
              <div class="opt-row">
                <span class="opt-label">Mode</span>
                <div class="seg">
                  <button class:active={fileAccurate} onclick={() => (fileAccurate = true)}>Accurate</button>
                  <button class:active={!fileAccurate} onclick={() => (fileAccurate = false)}>Fast</button>
                </div>
              </div>
            {/if}
            <label class="opt-toggle">
              <input type="checkbox" bind:checked={fileTimestamps} />
              <span>Timeline <em>— per-line timestamps for SRT/VTT</em></span>
            </label>
            <div class="field">
              <span class="field-label">Hints <em>(optional)</em></span>
              <input
                id="file-prompt"
                class="prompt-input"
                type="text"
                bind:value={filePrompt}
                placeholder="names, jargon, acronyms…"
              />
            </div>
            <p class="opt-hint">
              {#if fileEngine !== "cloud"}<strong>Accurate</strong> weighs several candidate sentences
                (better wording, slower). {/if}<strong>Hints</strong> prime spellings the model might
              otherwise miss.
            </p>
          </section>

          <section class="modal-section">
            <span class="section-title">Speakers</span>
            {#if fileModelSelfDiarizes}
              <p class="opt-hint">
                <strong>{fileMod?.name}</strong> returns speaker labels itself — local diarization is
                off for this model.
              </p>
            {:else}
              <label class="opt-toggle">
                <input type="checkbox" bind:checked={diarizeOn} />
                <span>Identify speakers</span>
              </label>
              {#if diarizeOn}
              <div class="opt-row">
                <span class="opt-label">Model</span>
                <div class="seg">
                  {#each diarizeModels as m (m.id)}
                    <button class:active={diarizeId === m.id} onclick={() => (diarizeId = m.id)}>
                      {diarizeShortName(m)}
                    </button>
                  {/each}
                </div>
              </div>
              {#if diarizeChosen && !diarizeChosen.installed}
                <button
                  class="btn outline sm dl-button"
                  onclick={() => downloadDiarize(diarizeId)}
                  disabled={downloading === diarizeId}
                >
                  {downloading === diarizeId
                    ? `Downloading… ${downloadPct}%`
                    : `Download ${fmtSize(diarizeChosen.sizeBytes)}`}
                </button>
              {/if}
            {/if}
              <p class="opt-hint">
                Labels each line by who's talking (Speaker 1, 2…). Runs locally after transcribing;
                downloads a small model the first time.
              </p>
            {/if}
          </section>
        </Modal>
      {/if}
    </section>
  {/if}
  </div>
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

  /* App shell: a fixed left nav rail + the workspace that holds the active mode. */
  .app {
    height: 100dvh;
    box-sizing: border-box;
    position: relative;
    display: flex;
  }

  /* Left nav rail — logo on top, Live/File in the middle, Settings gear pinned to the bottom. */
  .rail {
    position: relative;
    flex: none;
    width: 48px;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 3px;
    /* Top/bottom padding matches the workspace so the nav icons line up with the content. */
    padding: 16px 8px 18px;
    border-right: 1px solid var(--border);
    transition: width 0.16s ease;
  }

  .rail.expanded {
    width: 212px;
  }

  /* The collapse/expand handle, floating on the divider line — hidden until the rail is hovered. */
  .rail-edge {
    position: absolute;
    top: 50%;
    right: -11px;
    transform: translateY(-50%);
    z-index: 20;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 50%;
    cursor: pointer;
    opacity: 0;
    box-shadow: 0 1px 5px rgba(0, 0, 0, 0.08);
    transition:
      opacity 0.14s ease,
      background 0.12s,
      color 0.12s;
  }

  .rail:hover .rail-edge {
    opacity: 1;
  }

  .rail-edge:hover {
    background: var(--surface-active);
    color: var(--text);
  }

  .rail-chevron {
    width: 13px;
    height: 13px;
  }

  .rail.expanded .rail-edge .rail-chevron {
    transform: rotate(180deg);
  }

  .rail-nav {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .rail-item {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 36px;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    color: var(--muted);
    background: transparent;
    border: none;
    border-radius: 9px;
    cursor: pointer;
    transition:
      background 0.12s,
      color 0.12s;
  }

  .rail.expanded .rail-item {
    justify-content: flex-start;
    gap: 11px;
    padding: 0 11px;
  }

  .rail-item:hover {
    background: var(--surface-active);
    color: var(--text);
  }

  .rail-item.active {
    color: var(--accent);
    background: var(--surface-active);
  }

  .rail-ico {
    flex: none;
    width: 18px;
    height: 18px;
  }

  .rail-label {
    display: none;
  }

  .rail.expanded .rail-label {
    display: inline;
  }

  .rail-spacer {
    flex: 1;
  }

  /* The workspace: the active mode's content, capped to a readable width and centred in the space
     left of the rail. Grows wider when the assist panel is open so both columns fit. */
  .workspace {
    flex: 1;
    min-width: 0;
    height: 100dvh;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 16px 24px 18px;
    max-width: min(1180px, 100%);
    margin: 0 auto;
  }

  .app.wide .workspace {
    max-width: min(1680px, 100%);
  }

  /* Live is a working surface, not a reading column — let it grow to fill the window when enlarged
     (File stays capped for comfortable review). Defined after .wide so it wins when both apply. */
  .app.live .workspace {
    max-width: 100%;
  }

  /* The "smart" AI Assist launcher — a vibrant, gently shimmering gradient pill that appears top-right
     once a live session starts, so it reads as intelligent and is reachable the moment you press Start. */
  .assist-launch {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
    color: #fff;
    background: linear-gradient(120deg, #8b5cf6, #ec4899, #f97316, #ec4899, #8b5cf6);
    background-size: 240% 100%;
    border: none;
    border-radius: 999px;
    padding: 7px 16px;
    cursor: pointer;
    box-shadow: 0 4px 16px -3px rgba(168, 85, 247, 0.5);
    animation: assist-shimmer 7s ease infinite;
    transition:
      transform 0.15s,
      box-shadow 0.15s,
      filter 0.15s;
  }

  @keyframes assist-shimmer {
    0%,
    100% {
      background-position: 0% 50%;
    }
    50% {
      background-position: 100% 50%;
    }
  }

  .assist-launch:hover {
    transform: translateY(-1px);
    box-shadow: 0 7px 22px -3px rgba(168, 85, 247, 0.6);
    filter: brightness(1.06);
  }

  .assist-launch.on {
    box-shadow:
      0 0 0 2px color-mix(in srgb, #a855f7 55%, transparent),
      0 4px 16px -3px rgba(168, 85, 247, 0.5);
  }

  .assist-launch .spark {
    font-size: 14px;
    animation: spark-twinkle 2.4s ease-in-out infinite;
  }

  @keyframes spark-twinkle {
    0%,
    100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.72;
      transform: scale(0.9);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .assist-launch,
    .assist-launch .spark {
      animation: none;
    }
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

  /* Cloud realtime: cost/connection caveat + the collapsible generic parameters panel. */
  .cloud-live-note {
    margin: 0;
    font-size: 12px;
    line-height: 1.5;
    color: var(--muted);
  }

  /* Trigger for the right-side parameters drawer. */
  .params-trigger {
    align-self: flex-start;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 9px;
    padding: 7px 12px;
    cursor: pointer;
    transition: border-color 0.15s;
  }

  .params-trigger:hover {
    border-color: var(--accent);
  }

  /* Three-line "sliders" glyph, drawn so the panel needs no icon asset. */
  .params-ico {
    width: 13px;
    height: 9px;
    background-image:
      linear-gradient(var(--muted), var(--muted)), linear-gradient(var(--muted), var(--muted)),
      linear-gradient(var(--muted), var(--muted));
    background-size:
      13px 1.5px,
      13px 1.5px,
      13px 1.5px;
    background-position:
      left 0,
      left 4px,
      left 8px;
    background-repeat: no-repeat;
    position: relative;
  }

  /* Dimmed catch behind the drawer; clicking it closes. */
  .drawer-scrim {
    position: fixed;
    inset: 0;
    z-index: 40;
    background: rgba(0, 0, 0, 0.18);
    border: none;
    cursor: default;
  }

  .drawer {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    z-index: 41;
    width: 340px;
    max-width: 86vw;
    display: flex;
    flex-direction: column;
    background: var(--bg);
    border-left: 1px solid var(--border-strong);
    box-shadow: -14px 0 36px rgba(0, 0, 0, 0.14);
  }

  .drawer-head {
    flex: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 18px;
    border-bottom: 1px solid var(--border);
  }

  .drawer-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
  }

  .drawer-x {
    font-size: 20px;
    line-height: 1;
    color: var(--muted);
    background: none;
    border: none;
    cursor: pointer;
    padding: 0 2px;
  }

  .drawer-x:hover {
    color: var(--text);
  }

  .drawer-body {
    flex: 1;
    overflow-y: auto;
    padding: 18px;
  }

  /* Non-fatal info banner (e.g. system audio unavailable → mic-only) shown above the feed. */
  .live-notice {
    flex: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 9px 14px;
    font-size: 13px;
    color: var(--text);
    background: color-mix(in srgb, var(--accent) 9%, transparent);
    border-bottom: 1px solid var(--border);
  }

  .live-notice-x {
    flex: none;
    border: none;
    background: transparent;
    color: var(--muted);
    font-size: 17px;
    line-height: 1;
    cursor: pointer;
    padding: 0 2px;
  }

  .live-notice-x:hover {
    color: var(--text);
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

  /* Live: the lone session control (Start ⇄ Stop) sits centered, same spot. */
  .box-foot.center {
    justify-content: center;
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
    width: 11px;
    height: 7px;
    background-color: var(--muted);
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='7' viewBox='0 0 11 7' fill='none' stroke='%23000' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M1.5 1.5L5.5 5.5L9.5 1.5'/%3E%3C/svg%3E")
      no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='7' viewBox='0 0 11 7' fill='none' stroke='%23000' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M1.5 1.5L5.5 5.5L9.5 1.5'/%3E%3C/svg%3E")
      no-repeat center;
    transition: transform 0.15s;
  }

  .picker-trigger.open .picker-caret {
    transform: rotate(180deg);
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
    display: flex;
    flex-direction: column;
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    box-shadow: 0 14px 34px -10px rgba(40, 30, 20, 0.28);
    padding: 6px;
    max-height: 56vh;
  }

  /* Tabbed two-pane variant: tabs on top, then categories | models. The panes own padding + scroll. */
  .picker-menu.wide {
    right: auto;
    width: 520px;
    max-width: 92vw;
    padding: 0;
    max-height: none;
    overflow: hidden;
  }

  /* On-device / Cloud tabs at the top of the picker menu. */
  .picker-tabs {
    flex: none;
    display: flex;
    gap: 3px;
    padding: 6px;
    border-bottom: 1px solid var(--border);
  }

  .picker-tabs button {
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

  .picker-tabs button:hover {
    color: var(--text);
    background: var(--surface-active);
  }

  .picker-tabs button.active {
    color: #fff;
    background: var(--accent);
  }

  .picker-panes {
    display: flex;
    align-items: stretch;
    min-height: 0;
    max-height: 56vh;
  }

  /* Left column: the active tab's categories (families / providers). */
  .picker-cats {
    flex: none;
    width: 150px;
    border-right: 1px solid var(--border);
    padding: 6px;
    overflow-y: auto;
  }

  /* Claude-style scrollbar: thin, rounded, translucent, only assertive on hover. */
  .picker-cats,
  .picker-detail {
    scrollbar-width: thin;
    scrollbar-color: color-mix(in srgb, var(--muted) 32%, transparent) transparent;
  }

  .picker-cats::-webkit-scrollbar,
  .picker-detail::-webkit-scrollbar {
    width: 10px;
  }

  .picker-cats::-webkit-scrollbar-track,
  .picker-detail::-webkit-scrollbar-track {
    background: transparent;
  }

  .picker-cats::-webkit-scrollbar-thumb,
  .picker-detail::-webkit-scrollbar-thumb {
    background: color-mix(in srgb, var(--muted) 32%, transparent);
    border-radius: 999px;
    border: 3px solid transparent;
    background-clip: padding-box;
  }

  .picker-cats:hover::-webkit-scrollbar-thumb,
  .picker-detail:hover::-webkit-scrollbar-thumb {
    background: color-mix(in srgb, var(--muted) 52%, transparent);
    background-clip: padding-box;
  }

  .picker-cats::-webkit-scrollbar-thumb:hover,
  .picker-detail::-webkit-scrollbar-thumb:hover {
    background: color-mix(in srgb, var(--muted) 72%, transparent);
    background-clip: padding-box;
  }

  .picker-cat {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 7px 9px;
    cursor: pointer;
    text-align: left;
    transition:
      background 0.12s,
      color 0.12s;
  }

  .picker-cat:hover {
    background: var(--surface-active);
  }

  .picker-cat.active {
    background: var(--surface-active);
    color: var(--accent);
  }

  .picker-cat-name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Dot marking a cloud provider with no API key set yet. */
  .picker-cat-dot {
    flex: none;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--muted);
  }

  /* ✦ marker on the pinned "Recommended" category. */
  .picker-cat-star {
    flex: none;
    font-size: 11px;
    line-height: 1;
    color: var(--accent);
  }

  /* Right column: the selected category's models. */
  .picker-detail {
    flex: 1;
    min-width: 0;
    padding: 6px;
    overflow-y: auto;
  }

  .picker-detail-hint {
    padding: 8px 10px;
    font-size: 12px;
    line-height: 1.5;
    color: var(--muted);
  }

  /* Full-width footer across the whole dropdown (below both panes), shown only on the On-device tab. */
  .picker-custom {
    flex: none;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    width: 100%;
    padding: 10px 12px;
    border: none;
    border-top: 1px solid var(--border);
    background: transparent;
    cursor: pointer;
    text-align: center;
    font-family: inherit;
  }

  .picker-custom:hover {
    background: var(--surface-active);
  }

  /* Icon + label, centered together; the hint sits centered below. */
  .picker-custom-main {
    display: inline-flex;
    align-items: center;
    gap: 7px;
  }

  /* A download-into-tray glyph, tinted to the accent like the label (CSS mask, same as .picker-caret). */
  .picker-custom-icon {
    flex: none;
    width: 15px;
    height: 15px;
    background-color: var(--accent);
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16' fill='none' stroke='%23000' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M8 2.5v7'/%3E%3Cpath d='M5 6.5l3 3 3-3'/%3E%3Cpath d='M3 12.5h10'/%3E%3C/svg%3E")
      no-repeat center / contain;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16' fill='none' stroke='%23000' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M8 2.5v7'/%3E%3Cpath d='M5 6.5l3 3 3-3'/%3E%3Cpath d='M3 12.5h10'/%3E%3C/svg%3E")
      no-repeat center / contain;
  }

  /* Cloud footer reuses the import button but with a settings (sliders) glyph, not the import arrow. */
  .picker-custom-icon.cog {
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16' fill='none' stroke='%23000' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='2.5' y1='5' x2='13.5' y2='5'/%3E%3Cline x1='2.5' y1='11' x2='13.5' y2='11'/%3E%3Ccircle cx='6' cy='5' r='1.8'/%3E%3Ccircle cx='10' cy='11' r='1.8'/%3E%3C/svg%3E")
      no-repeat center / contain;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16' fill='none' stroke='%23000' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='2.5' y1='5' x2='13.5' y2='5'/%3E%3Cline x1='2.5' y1='11' x2='13.5' y2='11'/%3E%3Ccircle cx='6' cy='5' r='1.8'/%3E%3Ccircle cx='10' cy='11' r='1.8'/%3E%3C/svg%3E")
      no-repeat center / contain;
  }

  .picker-custom-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--accent);
  }

  .picker-custom-hint {
    font-size: 11px;
    color: var(--muted);
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

  /* Show the full model name: wrap rather than truncate, so even a long name (or a custom id) reads in
     full. The wider menu keeps common names on one line; this is the safety net for the rest. */
  .picker-opt-name {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
    line-height: 1.35;
  }

  .picker-opt-size {
    flex: none;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  /* The "heavy for this RAM" / "needs macOS 26" caveat shown on heavy or blocked rows. */
  .picker-opt-note {
    flex: none;
    font-size: 11.5px;
    color: var(--muted);
    white-space: nowrap;
  }

  /* A model this machine/OS can't run: greyed, not clickable, no hover. */
  .picker-opt.blocked {
    cursor: default;
    opacity: 0.5;
  }

  .picker-opt.blocked:hover {
    background: transparent;
  }

  .picker-tag {
    flex: none;
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--live);
  }

  /* The machine recommendation reads as a pill so it stands apart from the "active" marker. */
  .picker-tag.rec {
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 12%, transparent);
    padding: 1px 6px;
    border-radius: 999px;
  }

  .active-model {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    /* Grow like .engine-group so the audio chips sit at the right in both states (not the middle). */
    flex: 1;
    min-width: 0;
    font-size: 14px;
    font-weight: 500;
    color: var(--text);
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }

  /* The model a File run is using, shown muted next to the file name in the running header. */
  .file-model {
    font-weight: 400;
    color: var(--muted);
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

  .spinner {
    width: 13px;
    height: 13px;
    flex: none;
    border: 2px solid rgba(255, 255, 255, 0.4);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* ── Live transport: one round Start/Stop button, recorder/player style ──────────────────────── */
  .start-stack {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
  }

  /* Shown under the Start button only when a start is taking a while — a slow model load is progress,
     not a hang, so we reassure rather than fail. */
  .start-hint {
    font-size: 12px;
    color: var(--muted);
  }

  .transport-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 58px;
    height: 58px;
    flex: none;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    background: var(--accent);
    color: #fff;
    box-shadow: 0 6px 18px -6px color-mix(in srgb, var(--accent) 55%, transparent);
    transition:
      transform 0.15s,
      box-shadow 0.15s,
      background 0.15s,
      opacity 0.15s;
  }

  .transport-btn:hover:not(:disabled) {
    transform: translateY(-1px) scale(1.04);
    box-shadow: 0 10px 26px -6px color-mix(in srgb, var(--accent) 60%, transparent);
  }

  .transport-btn:active:not(:disabled) {
    transform: scale(0.97);
  }

  .transport-btn:disabled {
    opacity: 0.45;
    cursor: default;
    box-shadow: none;
  }

  .transport-btn.loading:disabled {
    opacity: 0.85;
    cursor: progress;
  }

  /* Recording: red, with a ring that pulses outward — the classic "live recorder" tell. */
  .transport-btn.stop {
    background: var(--stop);
    box-shadow: 0 6px 18px -6px color-mix(in srgb, var(--stop) 55%, transparent);
    animation: rec-ring 1.8s ease-out infinite;
  }

  @keyframes rec-ring {
    0% {
      box-shadow: 0 0 0 0 color-mix(in srgb, var(--stop) 42%, transparent);
    }
    70% {
      box-shadow: 0 0 0 15px color-mix(in srgb, var(--stop) 0%, transparent);
    }
    100% {
      box-shadow: 0 0 0 0 color-mix(in srgb, var(--stop) 0%, transparent);
    }
  }

  /* Play triangle (Start) ↔ stop square (Stop), centered in the circle. */
  .ic-play {
    width: 0;
    height: 0;
    border-style: solid;
    border-width: 11px 0 11px 18px;
    border-color: transparent transparent transparent #fff;
    margin-left: 4px;
  }

  .ic-stop {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    background: #fff;
  }

  @keyframes wave-bounce {
    0%,
    100% {
      transform: scaleY(0.28);
      opacity: 0.7;
    }
    50% {
      transform: scaleY(1);
      opacity: 1;
    }
  }

  /* Respect reduced-motion: drop the recording ring + the status dot pulse. */
  @media (prefers-reduced-motion: reduce) {
    .transport-btn.stop,
    .status.rec .status-dot {
      animation: none;
    }
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

  /* Live recording readout — accent (clay) with a breathing dot, the universal "REC" tell. */
  .status.rec {
    color: var(--accent);
    font-variant-numeric: tabular-nums;
  }

  .status.rec .status-dot {
    background: var(--accent);
    animation: rec-pulse 1.4s ease-in-out infinite;
  }

  @keyframes rec-pulse {
    0%,
    100% {
      box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent) 42%, transparent);
      opacity: 1;
    }
    50% {
      box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 0%, transparent);
      opacity: 0.65;
    }
  }

  /* Thin determinate progress bar under the file header; falls back to an indeterminate sweep
     until the engine reports its first percentage. */
  .file-cancel {
    margin-left: 8px;
    flex: none;
    font-family: inherit;
    font-size: 12px;
    color: var(--muted);
    background: transparent;
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 3px 10px;
    cursor: pointer;
  }

  .file-cancel:hover:not(:disabled) {
    color: var(--text);
    border-color: var(--accent);
  }

  .file-cancel:disabled {
    opacity: 0.6;
    cursor: default;
  }

  .file-progress {
    flex: none;
    height: 3px;
    background: var(--border);
    overflow: hidden;
  }

  .file-progress-fill {
    height: 100%;
    width: 0;
    background: var(--accent);
    transition: width 0.25s ease;
  }

  .file-progress.indeterminate .file-progress-fill {
    width: 32%;
    animation: file-progress-sweep 1.1s ease-in-out infinite;
  }

  @keyframes file-progress-sweep {
    0% {
      margin-left: -32%;
    }
    100% {
      margin-left: 100%;
    }
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

  .coreml-on {
    font-size: 13px;
    color: var(--accent);
    font-weight: 500;
  }

  .coreml-hint {
    font-size: 13px;
    color: var(--muted);
  }

  /* Why the chosen model can't start on this machine (e.g. "Needs macOS 26"). */
  .blocked-notice {
    font-size: 13px;
    color: var(--muted);
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
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }

  .notice-msg {
    flex: 1;
    min-width: 0;
    word-break: break-word;
  }

  .notice-x {
    flex: none;
    font-size: 17px;
    line-height: 1;
    color: var(--stop);
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 0 2px;
    opacity: 0.65;
  }

  .notice-x:hover {
    opacity: 1;
  }

  /* Advanced panel sits below the content box; collapsed by default so it costs ~no height. */
  /* Pill button that opens a settings modal (replaces the old inline disclosure, so the panel
     never reflows the page when shown). */
  .advanced-trigger {
    flex: none;
    align-self: start;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    width: fit-content;
    font-family: inherit;
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

  .advanced-trigger:hover {
    color: var(--text);
    border-color: var(--border-strong);
    background: var(--surface-active);
  }

  .trigger-icon {
    flex: none;
    width: 14px;
    height: 14px;
  }

  /* The File trigger sits inside the drop-zone box, so it needs the same inset the panel had. */
  .file-options-trigger {
    margin: 0 14px 14px;
  }

  /* Transcript ↔ AI Notes tabs + the scrolling AI pane in the File results view. */
  .file-tabs {
    align-self: flex-start;
    margin: 12px 14px 0;
  }

  .ai-pane {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  /* Live: transcript + AI assist side by side — the panel docks on the right, never overlays. */
  .live-body {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
  }

  .transcript-pane {
    flex: 1 1 auto;
    min-width: 0;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  .pane-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 10px 14px 6px;
  }

  .pane-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }

  /* Export (collapsed menu) + Clear, then a divider before Assist — right side of the transcript header. */
  .pane-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .pane-clear {
    font-family: inherit;
    font-size: 12px;
    color: var(--muted);
    background: transparent;
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 3px 10px;
    cursor: pointer;
  }

  .pane-clear:hover {
    color: var(--text);
    border-color: var(--accent);
  }

  /* Export collapsed into one trigger + popover, so the header reads as one control, not four. */
  .export {
    position: relative;
    display: inline-flex;
  }

  .export-trigger {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: inherit;
    font-size: 12px;
    color: var(--muted);
    background: transparent;
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 3px 9px;
    cursor: pointer;
    transition:
      color 0.12s,
      border-color 0.12s;
  }

  .export-trigger:hover,
  .export-trigger.open {
    color: var(--text);
    border-color: var(--accent);
  }

  .export-caret {
    width: 8px;
    height: 5px;
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5' viewBox='0 0 8 5' fill='none' stroke='%23000' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M1 1l3 3 3-3'/%3E%3C/svg%3E")
      no-repeat center;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5' viewBox='0 0 8 5' fill='none' stroke='%23000' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M1 1l3 3 3-3'/%3E%3C/svg%3E")
      no-repeat center;
    transition: transform 0.12s;
  }

  .export-trigger.open .export-caret {
    transform: rotate(180deg);
  }

  .export-backdrop {
    position: fixed;
    inset: 0;
    z-index: 20;
    background: transparent;
    border: none;
    cursor: default;
  }

  .export-menu {
    position: absolute;
    top: calc(100% + 5px);
    right: 0;
    z-index: 21;
    display: flex;
    flex-direction: column;
    min-width: 168px;
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    box-shadow: 0 12px 28px -10px rgba(40, 30, 20, 0.28);
    padding: 5px;
  }

  /* In the bottom-right pane-foot, the menu opens upward so the box edge never clips it. */
  .export-menu.up {
    top: auto;
    bottom: calc(100% + 5px);
  }

  .export-menu button {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    font-family: inherit;
    font-size: 13px;
    color: var(--text);
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 7px 9px;
    cursor: pointer;
    text-align: left;
    transition: background 0.12s;
  }

  .export-menu button:hover {
    background: var(--surface-active);
  }

  .export-ext {
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--muted);
  }

  /* Transcript utilities at the box's bottom-right (Export ▾ + Clear); the top-right holds Assist. */
  .pane-foot {
    flex: none;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
    padding: 4px 14px 9px;
  }

  .assist-panel {
    position: relative;
    flex: none;
    display: flex;
    flex-direction: column;
    min-height: 0;
    border-left: 1px solid var(--border);
    background: var(--surface);
  }

  /* Drag strip on the panel's left edge — pull left to widen, right to narrow. */
  .assist-resize {
    position: absolute;
    top: 0;
    bottom: 0;
    left: -3px;
    width: 7px;
    padding: 0;
    border: none;
    background: transparent;
    cursor: col-resize;
    z-index: 5;
  }

  .assist-resize:hover {
    background: color-mix(in srgb, var(--accent) 22%, transparent);
  }

  .assist-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
  }

  .assist-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
  }

  .assist-x {
    font-size: 18px;
    line-height: 1;
    color: var(--muted);
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 0 4px;
  }

  .assist-x:hover {
    color: var(--text);
  }

  .assist-body {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  /* Inset to align with the key row and drop zone — the shared .params-trigger has no margin (which
     suits the live box-aux it normally sits in, but floats flush-left here). */
  .file-params-trigger {
    margin: 12px 14px 0;
  }

  /* Each modal groups related controls into a titled card so the scopes read as distinct areas. */
  .modal-section {
    display: flex;
    flex-direction: column;
    gap: 11px;
    padding: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
  }

  .section-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--muted);
  }

  /* A labelled free-text field (Hints) — label stacked above the input so its purpose is obvious. */
  .field {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .field-label {
    font-size: 13.5px;
    color: var(--text);
  }

  .field-label em {
    color: var(--muted);
    font-style: normal;
    font-size: 12.5px;
  }

  /* Download button under a model/noise picker — hugs the left edge instead of stretching wide. */
  .dl-button {
    align-self: start;
  }

  .source-row {
    display: flex;
    align-items: center;
    gap: 14px;
  }

  .source-name {
    flex: 1;
    font-size: 13.5px;
  }

  .source-name em {
    color: var(--muted);
    font-style: normal;
    font-size: 12.5px;
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

  /* Stacks the verbatim text and its optional parallel rendering (a translation) in one column. */
  .feed li .body {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  /* The secondary rendering (a cloud model session's translation), under the verbatim line. */
  .aux-text {
    overflow-wrap: anywhere;
    color: var(--accent);
    font-size: 0.92em;
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

  /* The live "mic is open" row pinned at the feed foot while recording — animated, not static. */
  .feed li.listening {
    align-items: center;
    gap: 11px;
    padding: 9px 12px;
    border-bottom: none;
    color: var(--muted);
    font-size: 14px;
  }

  .eq {
    display: inline-flex;
    align-items: center;
    gap: 2.5px;
    height: 15px;
    flex: none;
  }

  .eq i {
    width: 2.5px;
    height: 100%;
    border-radius: 999px;
    background: var(--accent);
    transform: scaleY(0.3);
    transform-origin: center;
    animation: wave-bounce 1.1s ease-in-out infinite;
  }

  .eq i:nth-child(1) {
    animation-delay: -1s;
  }
  .eq i:nth-child(2) {
    animation-delay: -0.4s;
  }
  .eq i:nth-child(3) {
    animation-delay: -0.75s;
  }
  .eq i:nth-child(4) {
    animation-delay: -0.2s;
  }
  .eq i:nth-child(5) {
    animation-delay: -0.55s;
  }

  /* The word itself breathes, reinforcing the "actively listening" feel. */
  .listening-text {
    animation: listening-fade 1.8s ease-in-out infinite;
  }

  @keyframes listening-fade {
    0%,
    100% {
      opacity: 0.5;
    }
    50% {
      opacity: 1;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .eq i {
      animation: none;
      transform: scaleY(0.6);
    }
    .listening-text {
      animation: none;
    }
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

  .opt-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13.5px;
    color: var(--text);
    cursor: pointer;
  }

  .opt-toggle input {
    accent-color: var(--accent);
    cursor: pointer;
  }

  .opt-toggle em {
    color: var(--muted);
    font-style: normal;
    font-size: 12.5px;
  }


  .opt-row {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
  }

  .opt-label {
    flex: 1;
    font-size: 13.5px;
  }

  .seg {
    display: inline-flex;
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 2px;
  }

  .seg button {
    font-family: inherit;
    font-size: 12.5px;
    font-weight: 500;
    color: var(--muted);
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 4px 14px;
    cursor: pointer;
    transition:
      background 0.15s,
      color 0.15s;
  }

  .seg button:hover {
    color: var(--text);
  }

  .seg button.active {
    background: var(--accent);
    color: #fff;
  }

  .opt-hint {
    margin: 0;
    font-size: 12.5px;
    color: var(--muted);
    line-height: 1.5;
  }

  .opt-hint strong {
    color: var(--text);
    font-weight: 600;
  }

  .prompt-input {
    width: 100%;
    box-sizing: border-box;
    font-family: inherit;
    font-size: 13px;
    color: var(--text);
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 8px 11px;
    transition: border-color 0.15s;
  }

  .prompt-input::placeholder {
    color: var(--muted);
  }

  .prompt-input:focus {
    outline: none;
    border-color: var(--accent);
  }


  /* Speaker name prefix on a transcript line, tinted per speaker via the `--spk` variable. */
  .speaker {
    margin-right: 7px;
    font-weight: 600;
    color: var(--spk);
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

  /* File: on-device/cloud engine toggle and the cloud key affordance under the picker. */
  .file-pick-head {
    justify-content: flex-start;
  }

  /* "Transcribe with" label before the unified model dropdown. */
  .source-prefix {
    flex: none;
    font-size: 13px;
    color: var(--muted);
    white-space: nowrap;
  }

  .cloud-key-row {
    flex: none;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }

  .key-ok {
    color: var(--live);
    font-weight: 500;
  }

  .key-missing {
    color: var(--muted);
  }

  .link-btn {
    font-family: inherit;
    font-size: 13px;
    color: var(--accent);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    text-decoration: underline;
  }

  .link-btn:hover {
    color: var(--accent-hover);
  }

  /* Live: groups the engine toggle and its picker on the left of the header (status stays right). */
  .engine-group {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
    flex: 1;
  }

  /* Quick mic/system toggles in the Live bar — "You" (your mic) and "Them" (system/meeting audio). */
  .audio-chips {
    flex: none;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .audio-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: inherit;
    font-size: 12.5px;
    font-weight: 500;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 9px;
    padding: 6px 11px;
    cursor: pointer;
    transition:
      color 0.12s,
      border-color 0.12s,
      background 0.12s;
  }

  .audio-chip svg {
    width: 15px;
    height: 15px;
  }

  .audio-chip:hover {
    color: var(--text);
    border-color: var(--accent);
  }

  .audio-chip.on {
    color: var(--accent);
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 8%, var(--surface));
  }

</style>
