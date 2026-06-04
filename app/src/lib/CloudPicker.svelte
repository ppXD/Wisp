<script lang="ts">
  import { fly } from "svelte/transition";
  import {
    cloudState,
    cloudProvider,
    addCloudCustomModel,
    removeCloudCustomModel,
    openEndpointsModal,
  } from "$lib/cloud.svelte";

  // Left dropdown picks the provider, right dropdown picks the model. `capability` filters the
  // model list to what this mode can actually run (File = batch, Live = streaming).
  let {
    providerId = $bindable(),
    modelId = $bindable(),
    capability = "batch",
  }: {
    providerId: string;
    modelId: string;
    capability?: "batch" | "streaming";
  } = $props();

  let provOpen = $state(false);
  let modelOpen = $state(false);

  // Inline "add a custom model id" form, shown at the foot of the model menu (File/batch only).
  let adding = $state(false);
  let customId = $state("");
  let customName = $state("");
  let addError = $state("");

  const provider = $derived(cloudProvider(providerId));
  // A model runnable in this mode: File needs batch; Live takes either — a realtime model streams,
  // any batch model (including custom endpoints) runs segment-batch. So Live now lists batch models.
  const runnable = (m: { streaming: boolean; batch: boolean }) =>
    capability === "streaming" ? m.streaming || m.batch : m.batch;
  const visibleProviders = $derived(cloudState.providers.filter((p) => p.models.some(runnable)));
  const models = $derived(provider?.models.filter(runnable) ?? []);
  const model = $derived(models.find((m) => m.id === modelId));

  // Default to the first usable provider, then the provider's recommended capable model (falling
  // back to the first); re-pick when the current provider/model isn't valid for this capability.
  $effect(() => {
    if (visibleProviders.length && !visibleProviders.find((p) => p.id === providerId)) {
      providerId = visibleProviders[0].id;
    }
  });
  $effect(() => {
    if (models.length && !models.find((m) => m.id === modelId)) {
      modelId = (models.find((m) => m.recommended) ?? models[0]).id;
    }
  });
  // Reset the add-model form whenever the menu closes.
  $effect(() => {
    if (!modelOpen) {
      adding = false;
      addError = "";
    }
  });

  function pickProvider(id: string) {
    provOpen = false;
    if (id === providerId) return;
    providerId = id;
    modelId = ""; // let the effect default the model for the new provider
  }

  function pickModel(id: string) {
    modelOpen = false;
    modelId = id;
  }

  // Register the typed id as a custom model (batch-only), then select it. The backend rejects a
  // blank or duplicate id; its message is surfaced inline.
  async function confirmAdd() {
    const id = customId.trim();
    if (!id) return;
    try {
      await addCloudCustomModel(providerId, id, customName);
      customId = "";
      customName = "";
      adding = false;
      pickModel(id);
    } catch (e) {
      addError = String(e);
    }
  }

  function cancelAdd() {
    adding = false;
    addError = "";
    customId = "";
    customName = "";
  }

  async function removeCustom(id: string) {
    await removeCloudCustomModel(providerId, id);
    if (modelId === id) modelId = ""; // the effect re-defaults to a remaining model
  }

  // Endpoint add/edit/remove lives in the global ✦ Models manager (one place for every model
  // source); the picker only selects, and links there.
  function manageModels() {
    provOpen = false;
    openEndpointsModal();
  }
</script>

<div class="cloud-pick">
  <div class="picker">
    <button class="trigger" class:open={provOpen} onclick={() => (provOpen = !provOpen)}>
      <span class="lbl">{provider?.name ?? "Provider"}</span>
      <span class="caret"></span>
    </button>
    {#if provOpen}
      <button class="bg" aria-label="Close" onclick={() => (provOpen = false)}></button>
      <div class="menu" transition:fly={{ y: -6, duration: 120 }}>
        {#each visibleProviders as p (p.id)}
          <div class="opt-row">
            <button class="opt" class:sel={p.id === providerId} onclick={() => pickProvider(p.id)}>
              <span class="opt-name">{p.name}</span>
              {#if !p.keySet}<span class="tag">key needed</span>{/if}
            </button>
          </div>
        {/each}

        <div class="foot">
          <button class="opt add-row" onclick={manageModels}>✦ Manage models &amp; endpoints…</button>
        </div>
      </div>
    {/if}
  </div>

  <div class="picker">
    <button
      class="trigger"
      class:open={modelOpen}
      onclick={() => (modelOpen = !modelOpen)}
      disabled={!models.length && capability !== "batch"}
    >
      <span class="lbl">{model?.name ?? (models.length ? "Model" : "No model")}</span>
      <span class="caret"></span>
    </button>
    {#if modelOpen}
      <button class="bg" aria-label="Close" onclick={() => (modelOpen = false)}></button>
      <div class="menu" transition:fly={{ y: -6, duration: 120 }}>
        {#each models as m (m.id)}
          <div class="opt-row">
            <button class="opt" class:sel={m.id === modelId} onclick={() => pickModel(m.id)}>
              <span class="opt-name">{m.name}</span>
              {#if m.recommended}<span class="tag">recommended</span>{/if}
              {#if m.custom}<span class="tag">custom</span>{/if}
            </button>
            {#if m.custom}
              <button class="rm" aria-label={`Remove ${m.name}`} onclick={() => removeCustom(m.id)}
                >×</button
              >
            {/if}
          </div>
        {/each}

        {#if capability === "batch"}
          <div class="foot">
            {#if adding}
              <div class="add">
                <input
                  class="in"
                  placeholder="Model id (e.g. gpt-4o-transcribe)"
                  bind:value={customId}
                  onkeydown={(e) => e.key === "Enter" && confirmAdd()}
                />
                <input
                  class="in"
                  placeholder="Display name (optional)"
                  bind:value={customName}
                  onkeydown={(e) => e.key === "Enter" && confirmAdd()}
                />
                {#if addError}<span class="err">{addError}</span>{/if}
                <div class="row">
                  <button class="ghost" onclick={cancelAdd}>Cancel</button>
                  <button class="add-btn" disabled={!customId.trim()} onclick={confirmAdd}>Add</button>
                </div>
              </div>
            {:else}
              <button class="opt add-row" onclick={() => (adding = true)}>+ Custom model…</button>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .cloud-pick {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    flex: 1;
  }

  .picker {
    position: relative;
    flex: 1;
    min-width: 0;
  }

  .trigger {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    width: 100%;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 9px;
    padding: 7px 11px;
    cursor: pointer;
    transition:
      border-color 0.15s,
      background 0.15s;
  }

  .trigger:hover {
    border-color: var(--accent);
  }

  .trigger:disabled {
    opacity: 0.55;
    cursor: default;
  }

  .trigger:disabled:hover {
    border-color: var(--border-strong);
  }

  .lbl {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .caret {
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

  .trigger.open .caret {
    transform: rotate(180deg);
  }

  /* Invisible full-screen catch so a click outside closes the menu. */
  .bg {
    position: fixed;
    inset: 0;
    z-index: 20;
    background: transparent;
    border: none;
    cursor: default;
  }

  .menu {
    position: absolute;
    z-index: 21;
    top: calc(100% + 4px);
    left: 0;
    right: 0;
    min-width: 160px;
    max-height: 320px;
    overflow-y: auto;
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.16);
    padding: 4px;
  }

  .opt {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    text-align: left;
    font-family: inherit;
    font-size: 13px;
    color: var(--text);
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 7px 9px;
    cursor: pointer;
  }

  .opt:hover {
    background: var(--surface-active);
  }

  .opt.sel {
    background: var(--surface-active);
    font-weight: 600;
  }

  .opt-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tag {
    flex: none;
    font-size: 10.5px;
    font-weight: 500;
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 12%, transparent);
    border-radius: 999px;
    padding: 1px 7px;
  }

  /* A model row: the selectable button plus an optional remove (×) for custom ids. */
  .opt-row {
    display: flex;
    align-items: center;
    gap: 2px;
  }

  .opt-row .opt {
    flex: 1;
    min-width: 0;
    width: auto;
  }

  .rm {
    flex: none;
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    line-height: 1;
    color: var(--muted);
    background: transparent;
    border: none;
    border-radius: 6px;
    cursor: pointer;
  }

  .rm:hover {
    color: var(--text);
    background: var(--surface-active);
  }

  /* Pinned footer holding the "add a custom model id" affordance. */
  .foot {
    border-top: 1px solid var(--border-strong);
    margin-top: 4px;
    padding-top: 4px;
  }

  .add-row {
    color: var(--accent);
    font-weight: 500;
  }

  .add {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 6px 7px;
  }

  .in {
    width: 100%;
    box-sizing: border-box;
    font-family: inherit;
    font-size: 12.5px;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 6px 8px;
  }

  .in:focus {
    outline: none;
    border-color: var(--accent);
  }

  .err {
    font-size: 11.5px;
    color: #d4493f;
  }

  .row {
    display: flex;
    justify-content: flex-end;
    gap: 6px;
  }

  .ghost,
  .add-btn {
    font-family: inherit;
    font-size: 12.5px;
    font-weight: 500;
    border-radius: 7px;
    padding: 5px 11px;
    cursor: pointer;
    border: 1px solid transparent;
  }

  .ghost {
    color: var(--muted);
    background: transparent;
    border-color: var(--border-strong);
  }

  .add-btn {
    color: white;
    background: var(--accent);
  }

  .add-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }
</style>
