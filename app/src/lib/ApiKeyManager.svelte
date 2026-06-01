<script lang="ts">
  import Modal from "$lib/Modal.svelte";
  import { cloudState, setCloudKey } from "$lib/cloud.svelte";

  // Per-provider input buffers — never persisted here; only handed to setCloudKey on Save.
  let drafts = $state<Record<string, string>>({});

  async function save(id: string) {
    await setCloudKey(id, drafts[id] ?? "");
    drafts[id] = "";
  }
</script>

<Modal bind:open={cloudState.keyModalOpen} title="Cloud API keys">
  <p class="hint">Keys are stored only on this device, and sent only to the provider they belong to.</p>

  {#each cloudState.providers as p (p.id)}
    <section class="prov">
      <div class="prov-head">
        <span class="prov-name">{p.name}</span>
        {#if p.keySet}<span class="ok">saved</span>{/if}
      </div>
      <div class="entry">
        <input
          class="key-input"
          type="password"
          autocomplete="off"
          placeholder={p.keySet ? "Replace saved key…" : "Paste API key"}
          bind:value={drafts[p.id]}
        />
        <button class="btn" onclick={() => save(p.id)} disabled={!drafts[p.id]?.trim()}>Save</button>
        {#if p.keySet}
          <button class="btn ghost" onclick={() => setCloudKey(p.id, "")}>Remove</button>
        {/if}
      </div>
    </section>
  {:else}
    <p class="hint">No cloud providers available.</p>
  {/each}
</Modal>

<style>
  .hint {
    margin: 0;
    font-size: 12.5px;
    color: var(--muted);
    line-height: 1.5;
  }

  .prov {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .prov-head {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .prov-name {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text);
  }

  .ok {
    font-size: 11.5px;
    font-weight: 500;
    color: var(--live);
  }

  .entry {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .key-input {
    flex: 1;
    min-width: 0;
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

  .key-input::placeholder {
    color: var(--muted);
  }

  .key-input:focus {
    outline: none;
    border-color: var(--accent);
  }

  .btn {
    flex: none;
    font-family: inherit;
    font-size: 12.5px;
    font-weight: 500;
    color: #fff;
    background: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 7px 14px;
    cursor: pointer;
    transition:
      background 0.15s,
      opacity 0.15s;
  }

  .btn:hover {
    background: var(--accent-hover);
  }

  .btn:disabled {
    opacity: 0.45;
    cursor: default;
  }

  .btn.ghost {
    color: var(--muted);
    background: transparent;
    border-color: var(--border-strong);
  }

  .btn.ghost:hover {
    background: var(--surface-active);
    color: var(--text);
  }
</style>
