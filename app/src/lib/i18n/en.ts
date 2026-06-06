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

  common: {
    transcribeWith: "Transcribe with", // shared by the Live and File headers
    transcript: "Transcript",
    close: "Close",
  },
};

/** The catalogue contract. Every locale is typed `Messages`, so a missing or mistyped key fails the build. */
export type Messages = typeof en;
