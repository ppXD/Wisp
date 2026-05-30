<script lang="ts">
  import type { Snippet } from "svelte";
  import { fade, scale } from "svelte/transition";
  import { cubicOut } from "svelte/easing";

  let {
    open = $bindable(false),
    title = "",
    children,
  }: {
    open?: boolean;
    title?: string;
    children?: Snippet;
  } = $props();

  function close() {
    open = false;
  }

  function onKeydown(e: KeyboardEvent) {
    if (open && e.key === "Escape") close();
  }

  // Close only on a click of the backdrop itself, not one bubbling up from the card.
  function onBackdrop(e: MouseEvent) {
    if (e.target === e.currentTarget) close();
  }
</script>

<svelte:window onkeydown={onKeydown} />

{#if open}
  <!-- Backdrop click (on itself) and Escape close the dialog. -->
  <div
    class="backdrop"
    role="presentation"
    onclick={onBackdrop}
    transition:fade={{ duration: 140, easing: cubicOut }}
  >
    <div
      class="modal"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      tabindex="-1"
      transition:scale={{ duration: 140, start: 0.96, opacity: 1, easing: cubicOut }}
    >
      <header class="modal-head">
        <h2 class="modal-title">{title}</h2>
        <button class="modal-close" aria-label="Close" onclick={close}>
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
      <div class="modal-body">
        {@render children?.()}
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

  .modal {
    width: 100%;
    max-width: 440px;
    max-height: min(86vh, 660px);
    display: flex;
    flex-direction: column;
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: 16px;
    box-shadow: 0 24px 64px rgba(0, 0, 0, 0.28);
    overflow: hidden;
    /* Keep the card on its own compositing layer so the scale/fade outro doesn't repaint-flash
       in WKWebView (the cause of the flicker when the dialog closes). */
    will-change: transform;
    backface-visibility: hidden;
  }

  .modal-head {
    flex: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 14px 14px 18px;
    border-bottom: 1px solid var(--border);
  }

  .modal-title {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
  }

  .modal-close {
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

  .modal-close:hover {
    background: var(--surface-active);
    color: var(--text);
  }

  .modal-body {
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 18px;
  }
</style>
