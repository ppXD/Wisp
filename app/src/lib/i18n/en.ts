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
    settings: "Settings",
    themeToLight: "Switch to light theme",
    themeToDark: "Switch to dark theme",
    themeToggle: "Toggle colour theme",
    lightMode: "Light mode",
    darkMode: "Dark mode",
    language: "Language",
    languageMenu: "Choose language",
  },

  // Live transcription screen.
  live: {
    loadingModels: "Loading models…",

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
  },

  common: {
    transcribeWith: "Transcribe with", // shared by the Live and File headers
    transcript: "Transcript",
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
