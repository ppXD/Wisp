// English — the canonical catalogue. Its shape defines the `Messages` type (see i18n.svelte.ts),
// so every other locale must provide exactly these keys with the same value types, or it won't
// compile. Adding a string = add a key here; the type then forces every locale to supply it.
//
// Values are plain strings, except where a phrase depends on runtime state — those are small
// functions so each locale controls its own wording and word order.

export const en = {
  // Left navigation rail.
  nav: {
    collapse: "Collapse",
    expand: "Expand",
    collapseSidebar: "Collapse sidebar",
    expandSidebar: "Expand sidebar",
    live: "Live",
    file: "File",
    library: "Library",
    settings: "Settings",
    themeToLight: "Switch to light theme",
    themeToDark: "Switch to dark theme",
    themeToggle: "Toggle colour theme",
    lightMode: "Light mode",
    darkMode: "Dark mode",
    language: "Language",
    languageMenu: "Choose language",
  },

  library: {
    title: "Library",
    newMeetingTitle: (date: string): string => `Meeting · ${date}`,
    searchPlaceholder: "Search meetings…",
    empty: "No meetings yet. Finished sessions are saved here automatically.",
    noResults: "No matches.",
    back: "Back",
    delete: "Delete",
    cancel: "Cancel",
    deleteTitle: "Delete meeting",
    deleteConfirm: "Delete this meeting? This can't be undone.",
    you: "You",
    them: "Them",
  },

  // Live transcription screen.
  live: {
    loadingModels: "Loading models…",

    // Delete-model confirmation dialog (shared by the Live + File pickers).
    deleteModel: {
      trashTitle: (size: string): string => `Delete model · frees ${size}`,
      trashAria: (name: string, size: string): string => `Delete ${name}, frees ${size}`,
      title: "Delete model?",
      body: (name: string, size: string): string => `Delete ${name} and free ${size} of disk space?`,
      sub: "It stays in the catalog — you can re-download it anytime.",
      confirm: "Delete",
      deleting: "Deleting…",
      freed: (size: string, name: string): string => `Freed ${size} — deleted ${name}`,
    },

    you: "You",
    them: "Them",
    // Tooltip for the You/Them capture toggles — depends on whether the source is on and whether a
    // session is running (running → mute/unmute; idle → include/exclude from transcription).
    youTip: (on: boolean, running: boolean): string =>
      on
        ? running
          ? "You (your mic) is on — click to mute"
          : "You (your mic) is on — click to exclude from transcription"
        : running
          ? "You (your mic) is muted — click to unmute"
          : "You (your mic) is off — click to include in transcription",
    themTip: (on: boolean, running: boolean): string =>
      on
        ? running
          ? "Them (system audio) is on — click to mute"
          : "Them (system audio) is on — click to exclude from transcription"
        : running
          ? "Them (system audio) is muted — click to unmute"
          : "Them (system audio) is off — click to include in transcription",

    status: {
      ready: "ready",
      keyNeeded: "key needed",
      noModel: "no model",
      recording: "Recording", // rendered next to the elapsed time, e.g. "Recording · 0:05"
    },

    // Empty-state line. Split so the action word keeps its accent styling while each locale sets
    // its own word order around it.
    empty: {
      before: "Pick a model, press ",
      action: "Start",
      after: ", and speak.",
    },

    start: "Start transcription",
    stop: "Stop transcription",
    startConnecting: "Connecting…",
    startDownloading: "Downloading model…",
    startSlowHint: "Loading model — a first run can take a few seconds",

    advanced: "Advanced · audio, language, speakers",
    advancedCloud: "Audio · devices",
  },

  // File transcription screen (initial pick state; results/options modal follow in a later pass).
  file: {
    dropTitle: "Click to choose a file, or drop one here",
    options: "Options · accuracy, hints, speakers",
    optionsCloud: "Options · hints, speakers",
    // Dropzone sub-line. Variants that name a provider/model keep its <strong> styling, so the
    // sentence is split around the name and each locale sets its own word order.
    subCloudReady: { before: "mp3, m4a, wav, flac, mp4, mov… sent to ", after: "." },
    subCloudPick: "Choose a cloud model above.",
    subCloudNoKey: (provider: string): string => `Add your ${provider} API key to transcribe in the cloud.`,
    subLocalReady: { before: "mp3, m4a, wav, flac, mp4, mov… transcribed locally with ", after: "." },
    subLocalMissing: { before: "", after: " isn't downloaded yet — get it below to transcribe." },
  },

  // Model picker tags (shared by the in-app picker and the cloud picker). Model/provider NAMES are
  // not here — those stay English in every locale.
  picker: {
    active: "active",
    needsKey: "needs key",
    recommended: "recommended",
    custom: "custom",
    removeModel: (name: string): string => `Remove ${name}`,
    displayName: "Display name (optional)",
    provider: "Provider",
    model: "Model",
    noModel: "No model",
    manageModels: "✦ Manage models & endpoints…",
    addCustom: "+ Custom model…",
    onDevice: "On-device",
    cloud: "Cloud",
    recommendedCat: "Recommended",
    bestAccuracy: "best accuracy",
    forThisMachine: "for this machine",
    apiKeyNeeded: "API key needed",
    addKeyHint: (name: string): string => `Add an API key in Settings → AI models to use ${name}.`,
    noCloudModels: "No cloud models yet — add a provider & key in Settings → AI models.",
    importCustom: "Import custom model…",
    manageInSettings: "Manage models in Settings…",
    manageHint: "API keys · endpoints · custom models",
    selectModel: "Select a model",
  },

  // Live box-aux: model download / CoreML hints, permission banners, cloud key notices. "Wisp",
  // "Neural Engine", "System Settings", "Screen Recording"/"Microphone" (macOS panes), and provider
  // names stay English.
  notice: {
    blocked: (reason: string): string => `⚠ ${reason} — pick another model.`,
    download: "Download",
    retryDownload: "Retry download",
    coremlSupports: (size: string): string => `⚡ Supports Neural Engine acceleration · optional ${size} after install`,
    coremlOn: "⚡ Neural Engine acceleration on",
    coremlBoost: (size: string): string => `⚡ Neural Engine boost · ${size}`,
    screenRecOff: "Screen Recording is off.",
    screenRecOffBody: "Enable Wisp under Screen Recording in System Settings, then restart to apply it.",
    micOff: "Microphone is off.",
    micOffBody: "Enable Wisp under Microphone in System Settings, then restart — or set Microphone to Off in Advanced.",
    grant: "Grant",
    restart: "Restart",
    addKeyLead: (name: string): string => `Add your ${name} API key`,
    addKeyBody: "to run live cloud transcription. Stored on this device only.",
    apiKeys: "API keys",
    manageApiKey: "Manage API key",
    realtimeNote: (name: string): string =>
      `Cloud realtime streams audio continuously to ${name} — it bills per minute and needs a stable connection.`,
    sentenceNote: (name: string): string =>
      `${name} transcribes each finished sentence — near-live, with no mid-sentence partials, billed per request.`,
    advancedParams: "Advanced parameters",
  },

  // Transcript pane controls + the assist drawer header. "Markdown" is a format name (stays English).
  transcript: {
    listening: "Listening…",
    export: "Export",
    markdown: "Markdown",
    plainText: "Plain text",
    subtitles: "Subtitles",
    resizeAssist: "Resize assist panel",
    assistTitle: "✦ AI Assist",
  },

  // Live "Advanced settings" + File "Options" modals (audio / transcription / speakers), shared by
  // both. Help hints drop their <strong> emphasis so each is one translatable string. Language names
  // are option labels (translated); "SRT/VTT" and "Speaker 1, 2…" stay as written.
  advanced: {
    title: "Advanced settings",
    audioTitle: "Audio",
    optionsTitle: "Options",
    audio: "Audio",
    microphone: "Microphone",
    youParen: "(you)",
    systemDefault: "System default",
    off: "Off",
    systemAudio: "System audio",
    everythingPlaying: "(everything playing)",
    systemAudioNoSetup: "System audio — no setup",
    reduceNoise: "Reduce noise",
    light: "Light",
    balanced: "Balanced",
    skipSilence: "Skip silence & music",
    downloading: (pct: number): string => `Downloading… ${pct}%`,
    downloadSize: (size: string): string => `Download ${size}`,
    transcription: "Transcription",
    language: "Language",
    autoDetect: "Auto-detect",
    cantonese: "Cantonese",
    mandarin: "Chinese (Mandarin)",
    english: "English",
    japanese: "Japanese",
    korean: "Korean",
    mode: "Mode",
    accurate: "Accurate",
    fast: "Fast",
    timeline: "Timeline",
    timelineNote: "— per-line timestamps for SRT/VTT",
    hints: "Hints",
    optional: "(optional)",
    hintsPlaceholder: "names, jargon, acronyms…",
    speakers: "Speakers",
    identifySpeakers: "Identify speakers",
    model: "Model",
    // Diarization model variant labels (the suffix of the backend display_name). Unknown variants
    // fall back to the original English.
    diarizeLabel: (short: string): string => short,
    audioHint:
      "Defaults to your mic + all system audio with echo cancellation; for system audio only, set Microphone to Off.",
    audioHintLocal: " Light is the best fit for live.",
    audioHintCloud: " Cloud denoises server-side — tune it under Advanced parameters.",
    transcriptionHint:
      "Set a Language if auto-detect is wrong (recommended for Cantonese). Fast keeps the lowest latency; Hints prime names & jargon.",
    speakersHint:
      "Labels each line by who's talking (Speaker 1, 2…). Accurate tells similar-sounding voices apart better.",
    fileAudioHint:
      "Cleans background noise, and drops long non-speech so the model can't invent words in the gaps. Leave off for clean recordings.",
    fileHintAccurate: "Accurate weighs several candidate sentences (better wording, slower). ",
    fileHintHints: "Hints prime spellings the model might otherwise miss.",
    fileSpeakersSelf: (name: string): string =>
      `${name} returns speaker labels itself — local diarization is off for this model.`,
    fileSpeakersHint:
      "Labels each line by who's talking (Speaker 1, 2…). Runs locally after transcribing; downloads a small model the first time.",
  },

  // File transcribing/results state + the cloud key row and params drawer. "MD/TXT/SRT/VTT" and
  // provider names stay English.
  fileResult: {
    transcribing: "transcribing",
    done: "done",
    cancelling: "Cancelling…",
    aiNotes: "AI Notes",
    transcribingLarge: "Transcribing… large files take a little while.",
    transcribeAnother: "Transcribe another",
    keySaved: (name: string): string => `✓ ${name} key saved on this device`,
    manageKeys: "Manage keys",
    needsKey: (name: string): string => `${name} needs your API key`,
    addApiKey: "Add API key",
    paramsTitle: (name: string): string => `${name} parameters`,
  },

  // Advanced parameters panel (ParamsPanel). The per-parameter labels come from the backend specs.
  params: {
    title: "Parameters",
    reset: "Reset to defaults",
  },

  // Dictation settings (Settings.svelte). "Apple" / "macOS" stay English.
  settings: {
    aiModels: "AI models",
    dictation: "Dictation",
    dictationIntro:
      "Hold the hotkey, speak, release — Wisp types it into whatever app has focus, fully on-device (Apple speech).",
    dictationNote: "Dictation needs Apple on-device speech (macOS 26 or newer).",
    pushToTalk: "Push-to-talk",
    on: "On",
    off: "Off",
    hotkey: "Hotkey",
    accessibilityNote: "⚠ Needs Accessibility permission to type into other apps.",
    openSettings: "Open Settings",
    storage: "Storage",
    storageIntro:
      "Where Wisp keeps your models, meeting library, and app data on this device. Click Open to reveal a location in your file manager.",
    autoSaveMeetings: "Auto-save meetings to library",
    storageModels: "Models",
    storageMeetings: "Meeting library",
    storageData: "App data",
    openFolder: "Open",
  },

  // AI assist / notes panel (AiNotes.svelte). "✦ Models" / "AI" / "API" / provider names stay
  // English; the template PROMPTS (LLM instructions) are not here — only their menu labels are.
  assist: {
    emptyText: "Add an AI model for notes and live hints — your gateway, a local Ollama, or OpenAI.",
    manageInModels: "Manage in ✦ Models",
    needsKey: (name: string): string => `⚠ ${name} needs an API key — add it`,
    apiKeyNeeded: "API key needed",
    hint: "Hint",
    hintNow: "Pull a reply now",
    stop: "Stop",
    prompt: "Prompt",
    templates: "Templates",
    advanced: "Advanced",
    promptPlaceholder: "What should the assistant do with the transcript? Pick a template above or write your own.",
    start: "Start",
    connecting: "Connecting…",
    working: "Working…",
    realtimeNote: "⚡ Real-time assist listens to live audio — use it in a running Live session.",
    listening: "Listening — hints will appear here as you talk.",
    pressBefore: "Press ",
    pressRolling: " for rolling hints from the conversation.",
    pressSummary: " to summarize the transcript.",
    tmplSummary: "Summary",
    tmplActionItems: "Action items",
    tmplLiveHints: "Live hints (coach)",
    tmplDecisions: "Decisions & owners",
    tmplTranslate: "Translate to English",
    tmplNotes: "Detailed notes",
    tmplEmail: "Follow-up email",
    tmplQuestions: "Open questions",
    tmplSales: "Sales copilot (live)",
    tmplSupport: "Support copilot (live)",
    tmplSentiment: "Sentiment & tone (live)",
    tmplBlank: "Blank",
  },

  // Custom-endpoint manager (EndpointsManager.svelte). Technical tokens — Base URL, Model id, top_p,
  // the parameter names, code paths — and the dense API-shape explainer paragraphs stay English.
  endpoints: {
    name: "Name",
    namePlaceholder: "e.g. My gateway",
    apiKey: "API key",
    leaveBlank: "(leave blank to keep)",
    advanced: "Advanced — assist parameters & transcription",
    assistHead: "AI notes / assist",
    systemPrompt: "System prompt",
    systemPromptPlaceholder: "Standing instruction prepended to every assist task (persona, language, style).",
    providerDefault: "provider default",
    noLimit: "no limit",
    apiShapeHead: "Transcription API shape",
    builtin: "Built-in",
    customHead: "OpenAI-compatible endpoints",
    noKeyYet: "no key yet",
    keySet: "key set",
    noKey: "no key",
    getKey: "Get a key ↗",
    addKey: "Add key",
    keyPlaceholder: "Paste API key",
    show: "Show",
    hide: "Hide",
    addEndpoint: "+ Add OpenAI Compatible Endpoint",
    intro: "Keys are stored only on this device, and sent only to the provider they belong to.",
    // Add-form note + empty-state line: the <strong>/<code>/proper nouns stay literal in the
    // template, so the prose is split around them.
    formNoteBefore: "An ",
    formNoteAfter: " endpoint — base URL + key, like Cline or Ollama. Backs cloud transcription and AI notes/assist.",
    emptyBefore: "Add your own OpenAI-compatible endpoint — your gateway, a local Ollama (",
    emptyAfter: "), or OpenAI.",
  },

  // User-facing error toasts (set in <script>).
  error: {
    cloudError: (msg: string): string => `Cloud error: ${msg}`,
    pickCloudModel: "Pick a cloud model and save its API key first.",
    downloadSpeakerModel: "Download the speaker model first.",
    downloadModel: (name: string): string => `Download ${name} first.`,
    downloadNoiseModel: "Download the noise-reduction model first.",
  },

  common: {
    transcribeWith: "Transcribe with", // shared by the Live and File headers
    transcript: "Transcript",
    speaker: (n: number): string => `Speaker ${n}`,
    close: "Close",
    cancel: "Cancel",
    save: "Save",
    edit: "Edit",
    remove: "Remove",
    add: "Add",
    clear: "Clear",
    dismiss: "Dismiss",
    copy: "Copy",
  },
};

/** The catalogue contract. Every locale is typed `Messages`, so a missing or mistyped key fails the build. */
export type Messages = typeof en;
