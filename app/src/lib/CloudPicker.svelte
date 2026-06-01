<script lang="ts">
  import { slide } from "svelte/transition";
  import { cloudState, cloudProvider } from "$lib/cloud.svelte";

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

  const provider = $derived(cloudProvider(providerId));
  const models = $derived(
    provider?.models.filter((m) => (capability === "streaming" ? m.streaming : m.batch)) ?? [],
  );
  const model = $derived(models.find((m) => m.id === modelId));

  // Default to the first provider, then the first capable model; re-pick a model if the current one
  // isn't valid for the chosen provider/capability.
  $effect(() => {
    if (!providerId && cloudState.providers.length) providerId = cloudState.providers[0].id;
  });
  $effect(() => {
    if (models.length && !models.find((m) => m.id === modelId)) modelId = models[0].id;
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
</script>

<div class="cloud-pick">
  <div class="picker">
    <button class="trigger" class:open={provOpen} onclick={() => (provOpen = !provOpen)}>
      <span class="lbl">{provider?.name ?? "Provider"}</span>
      <span class="caret"></span>
    </button>
    {#if provOpen}
      <button class="bg" aria-label="Close" onclick={() => (provOpen = false)}></button>
      <div class="menu" transition:slide={{ duration: 120 }}>
        {#each cloudState.providers as p (p.id)}
          <button class="opt" class:sel={p.id === providerId} onclick={() => pickProvider(p.id)}>
            <span class="opt-name">{p.name}</span>
            {#if !p.keySet}<span class="tag">key needed</span>{/if}
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <div class="picker">
    <button
      class="trigger"
      class:open={modelOpen}
      onclick={() => (modelOpen = !modelOpen)}
      disabled={!models.length}
    >
      <span class="lbl">{model?.name ?? (models.length ? "Model" : "No model")}</span>
      <span class="caret"></span>
    </button>
    {#if modelOpen}
      <button class="bg" aria-label="Close" onclick={() => (modelOpen = false)}></button>
      <div class="menu" transition:slide={{ duration: 120 }}>
        {#each models as m (m.id)}
          <button class="opt" class:sel={m.id === modelId} onclick={() => pickModel(m.id)}>
            <span class="opt-name">{m.name}</span>
          </button>
        {/each}
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
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid var(--muted);
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
    max-height: 280px;
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
</style>
