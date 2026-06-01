<script lang="ts">
  import { openUrl } from "@tauri-apps/plugin-opener";
  import Modal from "$lib/Modal.svelte";
  import { cloudState, setCloudKey } from "$lib/cloud.svelte";

  // Which provider's key field is open (adding or editing); null when every row is collapsed.
  let editingId = $state<string | null>(null);
  let draft = $state("");
  let reveal = $state(false);

  function startEdit(id: string) {
    editingId = id;
    draft = "";
    reveal = false;
  }

  function cancel() {
    editingId = null;
    draft = "";
  }

  async function save(id: string) {
    if (!draft.trim()) return;
    await setCloudKey(id, draft);
    cancel();
  }

  async function getKey(url: string) {
    try {
      await openUrl(url);
    } catch {
      /* opening the browser is best-effort */
    }
  }

  // A two-letter monogram: the name's capitals, else its first two letters.
  function monogram(name: string): string {
    const caps = name.replace(/[^A-Z]/g, "");
    return (caps.length >= 2 ? caps.slice(0, 2) : name.slice(0, 2)).toUpperCase();
  }
</script>

<Modal bind:open={cloudState.keyModalOpen} title="Cloud API keys">
  <p class="intro">Keys are stored only on this device, and sent only to the provider they belong to.</p>

  <div class="rows">
    {#each cloudState.providers as p (p.id)}
      <div class="row">
        <div class="chip">{monogram(p.name)}</div>

        <div class="body">
          <div class="line">
            <span class="name">{p.name}</span>
            {#if editingId !== p.id}
              <span class="status" class:on={p.keySet}>
                <span class="dot"></span>{p.keySet ? "Connected" : "Not connected"}
              </span>
            {/if}
          </div>

          {#if editingId === p.id}
            <div class="line">
              <div class="field">
                <!-- svelte-ignore a11y_autofocus -->
                <input
                  class="inp"
                  type={reveal ? "text" : "password"}
                  autocomplete="off"
                  autocapitalize="off"
                  spellcheck="false"
                  autofocus
                  placeholder="Paste API key"
                  bind:value={draft}
                  onkeydown={(e) => {
                    if (e.key === "Enter") save(p.id);
                    if (e.key === "Escape") cancel();
                  }}
                />
                <button class="reveal" type="button" onclick={() => (reveal = !reveal)}>
                  {reveal ? "Hide" : "Show"}
                </button>
              </div>
              <button class="btn" onclick={() => save(p.id)} disabled={!draft.trim()}>Save</button>
              <button class="btn ghost" onclick={cancel}>Cancel</button>
            </div>
          {:else if p.keySet}
            <div class="line">
              <code class="key">{p.keyHint}</code>
              <div class="acts">
                <button class="link" onclick={() => startEdit(p.id)}>Edit</button>
                <button class="link danger" onclick={() => setCloudKey(p.id, "")}>Remove</button>
              </div>
            </div>
          {:else}
            <div class="line">
              <button class="getkey" onclick={() => getKey(p.keysUrl)}>Get a key ↗</button>
              <button class="btn" onclick={() => startEdit(p.id)}>Add key</button>
            </div>
          {/if}
        </div>
      </div>
    {:else}
      <p class="intro">No cloud providers available.</p>
    {/each}
  </div>
</Modal>

<style>
  .intro {
    margin: 0;
    font-size: 12.5px;
    color: var(--muted);
    line-height: 1.5;
  }

  .rows {
    display: flex;
    flex-direction: column;
  }

  .row {
    display: grid;
    grid-template-columns: 34px 1fr;
    gap: 12px;
    padding: 14px 0;
    align-items: start;
  }

  .row:not(:last-child) {
    border-bottom: 1px solid var(--border);
  }

  .chip {
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 9px;
    background: var(--surface-active);
    border: 1px solid var(--border);
    color: var(--text);
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 0.02em;
  }

  .body {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
  }

  .line {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }

  .name {
    flex: 1;
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .status {
    flex: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--muted);
  }

  .status .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--border-strong);
  }

  .status.on {
    color: var(--live);
  }

  .status.on .dot {
    background: var(--live);
  }

  .key {
    flex: 1;
    font-family: var(--font-mono);
    font-size: 12.5px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .acts {
    flex: none;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .field {
    position: relative;
    flex: 1;
    min-width: 0;
  }

  .inp {
    width: 100%;
    box-sizing: border-box;
    font-family: inherit;
    font-size: 13px;
    color: var(--text);
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 8px 52px 8px 11px;
    transition: border-color 0.15s;
  }

  .inp::placeholder {
    color: var(--muted);
  }

  .inp:focus {
    outline: none;
    border-color: var(--accent);
  }

  .reveal {
    position: absolute;
    right: 5px;
    top: 50%;
    transform: translateY(-50%);
    font-family: inherit;
    font-size: 11.5px;
    font-weight: 500;
    color: var(--muted);
    background: transparent;
    border: none;
    padding: 4px 6px;
    cursor: pointer;
  }

  .reveal:hover {
    color: var(--text);
  }

  .getkey {
    flex: 1;
    text-align: left;
    font-family: inherit;
    font-size: 12.5px;
    color: var(--accent);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
  }

  .getkey:hover {
    color: var(--accent-hover);
  }

  .link {
    font-family: inherit;
    font-size: 12.5px;
    font-weight: 500;
    color: var(--muted);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    transition: color 0.15s;
  }

  .link:hover {
    color: var(--text);
  }

  .link.danger:hover {
    color: var(--stop);
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
