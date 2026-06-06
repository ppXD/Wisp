<script lang="ts">
  // The app's Settings dialog: a left sidebar of categories, a right content panel — Claude-style.
  // Categories are self-contained (the AI-models manager, dictation); add more by extending `sections`.
  import { fade, scale } from "svelte/transition";
  import { cubicOut } from "svelte/easing";
  import { invoke } from "@tauri-apps/api/core";
  import EndpointsManager from "$lib/EndpointsManager.svelte";
  import { i18n } from "$lib/i18n.svelte";

  let { open = $bindable(false) }: { open?: boolean } = $props();

  type Section = "models" | "dictation";
  const sections = $derived<{ id: Section; label: string }[]>([
    { id: "models", label: i18n.t.settings.aiModels },
    { id: "dictation", label: i18n.t.settings.dictation },
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

  // Refresh dictation status whenever the dialog opens (so a permission granted in System Settings,
  // or the current hotkey, shows correctly).
  $effect(() => {
    if (open) loadDictation();
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
</style>
