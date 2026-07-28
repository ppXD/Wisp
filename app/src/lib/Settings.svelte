<script lang="ts">
  // The app's Settings dialog: a left sidebar of categories, a right content panel — Claude-style.
  // Categories are self-contained (the AI-models manager, dictation); add more by extending `sections`.
  import { fade, scale } from "svelte/transition";
  import { cubicOut } from "svelte/easing";
  import { invoke } from "@tauri-apps/api/core";
  import { appDataDir, join } from "@tauri-apps/api/path";
  import { openPath, revealItemInDir } from "@tauri-apps/plugin-opener";
  import EndpointsManager from "$lib/EndpointsManager.svelte";
  import NotesSearch from "$lib/NotesSearch.svelte";
  import { i18n } from "$lib/i18n.svelte";

  let { open = $bindable(false), autoSave = $bindable(false) }: { open?: boolean; autoSave?: boolean } =
    $props();

  type Section = "models" | "search" | "downloads" | "dictation" | "storage";
  const sections = $derived<{ id: Section; label: string }[]>([
    { id: "models", label: i18n.t.settings.aiModels },
    { id: "search", label: i18n.t.settings.search },
    { id: "downloads", label: i18n.t.settings.downloads },
    { id: "dictation", label: i18n.t.settings.dictation },
    { id: "storage", label: i18n.t.settings.storage },
  ]);
  let section = $state<Section>("models");

  function close() {
    open = false;
  }
  function onKeydown(e: KeyboardEvent) {
    if (open && e.key === "Escape") close();
  }
  function onBackdrop(e: MouseEvent) {
    if (e.target === e.currentTarget) close();
  }

  // ── Dictation category ──────────────────────────────────────────────────────────────────────────
  type DictationStatus = {
    available: boolean;
    accessibilityOk: boolean;
    enabled: boolean;
    hotkey: string;
  };
  let dictation = $state<DictationStatus | null>(null);
  let dictationError = $state("");

  async function loadDictation() {
    try {
      dictation = await invoke<DictationStatus>("dictation_status");
    } catch {
      dictation = null;
    }
  }

  async function setDictation(enabled: boolean, hotkey?: string) {
    try {
      dictation = await invoke<DictationStatus>("set_dictation_enabled", {
        enabled,
        hotkey: hotkey ?? null,
      });
      dictationError = "";
    } catch (e) {
      dictationError = String(e);
    }
  }

  async function grantAccessibility() {
    try {
      await invoke("open_accessibility_settings");
    } catch (e) {
      dictationError = String(e);
    }
  }

  // ── Regional downloads ─────────────────────────────────────────────────────────────────────────
  type DownloadSettings = {
    source: "auto" | "official" | "mirror";
    mirrorUrl: string;
    proxyMode: "system" | "direct" | "custom";
    proxyUrl: string;
  };
  let downloads = $state<DownloadSettings>({
    source: "auto",
    mirrorUrl: "https://hf-mirror.com",
    proxyMode: "system",
    proxyUrl: "",
  });
  let downloadSettingsBusy = $state(false);
  let downloadSettingsError = $state("");
  let downloadSettingsSaved = $state(false);

  async function loadDownloadSettings() {
    try {
      downloads = await invoke<DownloadSettings>("get_download_settings");
      downloadSettingsError = "";
    } catch (e) {
      downloadSettingsError = String(e);
    }
  }

  async function saveDownloadSettings() {
    downloadSettingsBusy = true;
    downloadSettingsSaved = false;
    downloadSettingsError = "";
    try {
      downloads = await invoke<DownloadSettings>("set_download_settings", { settings: downloads });
      downloadSettingsSaved = true;
    } catch (e) {
      downloadSettingsError = String(e);
    }
    downloadSettingsBusy = false;
  }

  // ── Storage category ────────────────────────────────────────────────────────────────────────────
  // Where Wisp keeps things on disk, resolved from the Tauri app-data dir — the same dir the Rust side
  // opens the SQLite library and the model store under.
  type StoragePaths = { models: string; database: string; data: string };
  let paths = $state<StoragePaths | null>(null);
  let storageError = $state("");

  async function loadPaths() {
    try {
      const data = await appDataDir();
      paths = { data, models: await join(data, "models"), database: await join(data, "library.db") };
      storageError = "";
    } catch (e) {
      storageError = String(e);
    }
  }

  // Reveal a file in its folder, or open a folder directly — both surface it in the OS file manager.
  async function openLocation(path: string, reveal: boolean) {
    try {
      if (reveal) await revealItemInDir(path);
      else await openPath(path);
    } catch (e) {
      storageError = String(e);
    }
  }

  // Refresh dictation status + the storage paths whenever the dialog opens.
  $effect(() => {
    if (open) {
      loadDictation();
      loadDownloadSettings();
      loadPaths();
    }
  });
</script>

<svelte:window onkeydown={onKeydown} />

{#if open}
  <div
    class="backdrop"
    role="presentation"
    onclick={onBackdrop}
    transition:fade={{ duration: 140, easing: cubicOut }}
  >
    <div
      class="settings"
      role="dialog"
      aria-modal="true"
      aria-label={i18n.t.nav.settings}
      tabindex="-1"
      transition:scale={{ duration: 140, start: 0.96, opacity: 1, easing: cubicOut }}
    >
      <header class="settings-head">
        <h2 class="settings-title">{i18n.t.nav.settings}</h2>
        <button class="settings-close" aria-label={i18n.t.common.close} onclick={close}>
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            stroke-width="1.6"
            stroke-linecap="round"
          >
            <path d="M4 4l8 8M12 4l-8 8" />
          </svg>
        </button>
      </header>

      <div class="settings-body">
        <nav class="settings-nav">
          {#each sections as s (s.id)}
            <button class:active={section === s.id} onclick={() => (section = s.id)}>
              {s.label}
            </button>
          {/each}
        </nav>

        <div class="settings-content">
          {#if section === "models"}
            <EndpointsManager />
          {:else if section === "search"}
            <NotesSearch />
          {:else if section === "downloads"}
            <p class="set-intro">{i18n.t.settings.downloadsIntro}</p>

            <label class="set-row">
              <span class="set-label">{i18n.t.settings.downloadSource}</span>
              <select
                class="set-input set-select"
                value={downloads.source}
                onchange={(e) => {
                  downloads.source = e.currentTarget.value as DownloadSettings["source"];
                  downloadSettingsSaved = false;
                }}
              >
                <option value="auto">{i18n.t.settings.downloadSourceAuto}</option>
                <option value="official">{i18n.t.settings.downloadSourceOfficial}</option>
                <option value="mirror">{i18n.t.settings.downloadSourceMirror}</option>
              </select>
            </label>

            <label class="set-field">
              <span class="set-label">{i18n.t.settings.downloadMirror}</span>
              <input
                class="set-input"
                type="url"
                autocomplete="off"
                disabled={downloads.source === "official"}
                value={downloads.mirrorUrl}
                oninput={(e) => {
                  downloads.mirrorUrl = e.currentTarget.value;
                  downloadSettingsSaved = false;
                }}
                placeholder="https://hf-mirror.com"
              />
              <span class="set-help">{i18n.t.settings.downloadMirrorHint}</span>
            </label>

            <label class="set-row">
              <span class="set-label">{i18n.t.settings.downloadProxy}</span>
              <select
                class="set-input set-select"
                value={downloads.proxyMode}
                onchange={(e) => {
                  downloads.proxyMode = e.currentTarget.value as DownloadSettings["proxyMode"];
                  downloadSettingsSaved = false;
                }}
              >
                <option value="system">{i18n.t.settings.downloadProxySystem}</option>
                <option value="direct">{i18n.t.settings.downloadProxyDirect}</option>
                <option value="custom">{i18n.t.settings.downloadProxyCustom}</option>
              </select>
            </label>

            {#if downloads.proxyMode === "custom"}
              <label class="set-field">
                <span class="set-label">{i18n.t.settings.downloadProxyUrl}</span>
                <input
                  class="set-input"
                  type="text"
                  autocomplete="off"
                  value={downloads.proxyUrl}
                  oninput={(e) => {
                    downloads.proxyUrl = e.currentTarget.value;
                    downloadSettingsSaved = false;
                  }}
                  placeholder="http://127.0.0.1:7890"
                />
                <span class="set-help">{i18n.t.settings.downloadProxyHint}</span>
              </label>
            {/if}

            <div class="set-actions">
              <button class="set-btn primary" disabled={downloadSettingsBusy} onclick={saveDownloadSettings}>
                {downloadSettingsBusy
                  ? i18n.t.settings.downloadSettingsSaving
                  : i18n.t.settings.downloadSettingsSave}
              </button>
              {#if downloadSettingsSaved}<span class="set-saved">{i18n.t.settings.downloadSettingsSaved}</span>{/if}
            </div>
            {#if downloadSettingsError}<p class="set-error">{downloadSettingsError}</p>{/if}
          {:else if section === "dictation"}
            <p class="set-intro">{i18n.t.settings.dictationIntro}</p>

            {#if dictation && !dictation.available}
              <p class="set-note">{i18n.t.settings.dictationNote}</p>
            {:else if dictation}
              <div class="set-row">
                <span class="set-label">{i18n.t.settings.pushToTalk}</span>
                <button
                  class="set-btn"
                  class:on={dictation.enabled}
                  onclick={() => dictation && setDictation(!dictation.enabled)}
                >
                  {dictation.enabled ? i18n.t.settings.on : i18n.t.settings.off}
                </button>
              </div>

              <label class="set-row">
                <span class="set-label">{i18n.t.settings.hotkey}</span>
                <input
                  class="hotkey-input"
                  value={dictation.hotkey}
                  placeholder="CmdOrCtrl+Shift+D"
                  onchange={(e) => setDictation(dictation?.enabled ?? false, e.currentTarget.value)}
                />
              </label>

              {#if !dictation.accessibilityOk}
                <div class="set-perm">
                  <span>{i18n.t.settings.accessibilityNote}</span>
                  <button class="set-btn" onclick={grantAccessibility}>{i18n.t.settings.openSettings}</button>
                </div>
              {/if}
            {/if}

            {#if dictationError}<p class="set-error">{dictationError}</p>{/if}
          {:else if section === "storage"}
            <p class="set-intro">{i18n.t.settings.storageIntro}</p>

            <div class="set-row">
              <span class="set-label">{i18n.t.settings.autoSaveNotes}</span>
              <button class="set-btn" class:on={autoSave} onclick={() => (autoSave = !autoSave)}>
                {autoSave ? i18n.t.settings.on : i18n.t.settings.off}
              </button>
            </div>

            {#if storageError}<p class="set-error">{storageError}</p>{/if}
            {#if paths}
              {#each [{ label: i18n.t.settings.storageModels, path: paths.models, reveal: false }, { label: i18n.t.settings.storageNotes, path: paths.database, reveal: true }, { label: i18n.t.settings.storageData, path: paths.data, reveal: false }] as loc (loc.path)}
                <div class="store-row">
                  <div class="store-info">
                    <span class="set-label">{loc.label}</span>
                    <span class="store-path" title={loc.path}>{loc.path}</span>
                  </div>
                  <button class="set-btn" onclick={() => openLocation(loc.path, loc.reveal)}>
                    {i18n.t.settings.openFolder}
                  </button>
                </div>
              {/each}
            {/if}
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    background: color-mix(in srgb, #000 36%, transparent);
  }

  .settings {
    width: 100%;
    max-width: 720px;
    height: min(82vh, 560px);
    display: flex;
    flex-direction: column;
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: 16px;
    box-shadow: 0 24px 64px rgba(0, 0, 0, 0.28);
    overflow: hidden;
    will-change: transform;
    backface-visibility: hidden;
  }

  .settings-head {
    flex: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 14px 14px 18px;
    border-bottom: 1px solid var(--border);
  }

  .settings-title {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
  }

  .settings-close {
    flex: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border: none;
    border-radius: 8px;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    transition:
      background 0.15s,
      color 0.15s;
  }

  .settings-close:hover {
    background: var(--surface-active);
    color: var(--text);
  }

  .settings-body {
    flex: 1;
    min-height: 0;
    display: flex;
  }

  .settings-nav {
    flex: none;
    width: 168px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 12px 10px;
    border-right: 1px solid var(--border);
    overflow-y: auto;
  }

  .settings-nav button {
    font-family: inherit;
    font-size: 13px;
    text-align: left;
    color: var(--muted);
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 7px 10px;
    cursor: pointer;
    transition:
      background 0.12s,
      color 0.12s;
  }

  .settings-nav button:hover {
    background: var(--surface-active);
    color: var(--text);
  }

  .settings-nav button.active {
    background: var(--surface-active);
    color: var(--accent);
    font-weight: 500;
  }

  .settings-content {
    flex: 1;
    min-width: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 18px;
  }

  .set-intro {
    margin: 0;
    font-size: 13px;
    line-height: 1.5;
    color: var(--muted);
  }

  .set-note {
    margin: 0;
    font-size: 13px;
    color: var(--muted);
  }

  .set-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .set-label {
    font-size: 13px;
    color: var(--text);
  }

  .set-btn {
    font-family: inherit;
    font-size: 12px;
    color: var(--muted);
    background: transparent;
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 4px 12px;
    cursor: pointer;
    transition:
      color 0.12s,
      border-color 0.12s;
  }

  .set-btn:hover {
    color: var(--text);
    border-color: var(--accent);
  }

  .set-btn.on {
    color: var(--accent);
    border-color: var(--accent);
  }

  .set-btn.primary {
    color: var(--accent);
    border-color: var(--accent);
  }

  .set-btn:disabled {
    opacity: 0.55;
    cursor: default;
  }

  .set-field {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .set-input {
    width: 100%;
    box-sizing: border-box;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 7px 9px;
  }

  .set-input:disabled {
    opacity: 0.5;
  }

  .set-select {
    width: 230px;
    font-family: inherit;
  }

  .set-help {
    font-size: 11.5px;
    line-height: 1.45;
    color: var(--muted);
  }

  .set-actions {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .set-saved {
    font-size: 12px;
    color: var(--accent);
  }

  .hotkey-input {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 4px 8px;
    width: 190px;
  }

  .set-perm {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    font-size: 12.5px;
    color: var(--muted);
  }

  .set-error {
    margin: 0;
    font-size: 12.5px;
    color: var(--danger, #c0392b);
  }

  .store-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 0;
    border-top: 1px solid var(--border);
  }

  .store-info {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .store-path {
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
