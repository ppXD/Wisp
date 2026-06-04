<script lang="ts">
  // The one generic place to manage every model source. A "source" is just a provider: built-in
  // ones (OpenAI, Groq, …) have a fixed URL/protocol and only need a key; custom OpenAI-compatible
  // ones (your gateway, a local Ollama, OpenAI) you define end to end (URL + model + key). Both kinds
  // live in one list here, and both back File/Live transcription and the AI assist/notes.
  import { openUrl } from "@tauri-apps/plugin-opener";
  import {
    cloudState,
    addCloudEndpoint,
    updateCloudEndpoint,
    removeCloudEndpoint,
    setCloudKey,
  } from "$lib/cloud.svelte";

  // Content-only: the host (the Settings dialog) owns the modal chrome; this just renders the
  // AI-models & endpoints manager body.

  const builtins = $derived(cloudState.providers.filter((p) => !p.custom));
  const customs = $derived(cloudState.providers.filter((p) => p.custom));

  // ── Built-in provider: inline key editing (the only thing you configure on a fixed provider) ──
  let keyEditId = $state<string | null>(null);
  let keyDraft = $state("");
  let reveal = $state(false);

  function startKey(id: string) {
    keyEditId = id;
    keyDraft = "";
    reveal = false;
  }

  function cancelKey() {
    keyEditId = null;
    keyDraft = "";
  }

  async function saveKey(id: string) {
    if (!keyDraft.trim()) return;
    await setCloudKey(id, keyDraft);
    cancelKey();
  }

  async function getKey(url: string) {
    try {
      await openUrl(url);
    } catch {
      // opening the browser is best-effort
    }
  }

  // ── Custom endpoint: full add/edit form. null = list view; "new" = adding; an id = editing it. ──
  let editing = $state<string | null>(null);
  let name = $state("");
  let baseUrl = $state("");
  let protocol = $state("openai");
  let model = $state("");
  let apiKey = $state("");
  // Assist params as strings (empty = unset); parsed to numbers on save.
  let temperature = $state("");
  let maxTokens = $state("");
  let contextTokens = $state("");
  let topP = $state("");
  let systemPrompt = $state("");
  let busy = $state(false);
  let error = $state("");

  /** Parse a numeric field; blank or non-numeric → null (the backend treats null as "use default"). */
  function num(s: string): number | null {
    const t = s.trim();
    if (!t) return null;
    const n = Number(t);
    return Number.isFinite(n) ? n : null;
  }

  function reset() {
    editing = null;
    name = "";
    baseUrl = "";
    protocol = "openai";
    model = "";
    apiKey = "";
    temperature = "";
    maxTokens = "";
    contextTokens = "";
    topP = "";
    systemPrompt = "";
    error = "";
  }

  function startAdd() {
    reset();
    editing = "new";
  }

  function startEdit(p: (typeof customs)[number]) {
    reset();
    editing = p.id;
    name = p.name;
    baseUrl = p.baseUrl;
    protocol = p.protocol === "chat" ? "chat" : "openai";
    model = p.models[0]?.id ?? "";
    const a = p.assist;
    temperature = a.temperature != null ? String(a.temperature) : "";
    maxTokens = a.maxTokens != null ? String(a.maxTokens) : "";
    contextTokens = a.contextTokens != null ? String(a.contextTokens) : "";
    topP = a.topP != null ? String(a.topP) : "";
    systemPrompt = a.systemPrompt ?? "";
  }

  async function save() {
    if (busy) return;
    busy = true;
    error = "";
    try {
      const draft = {
        name,
        baseUrl,
        protocol,
        model,
        assist: {
          temperature: num(temperature),
          maxTokens: num(maxTokens),
          contextTokens: num(contextTokens),
          topP: num(topP),
          systemPrompt: systemPrompt.trim(),
        },
      };
      let id: string;
      if (editing === "new") {
        id = await addCloudEndpoint(draft);
      } else {
        id = editing!;
        await updateCloudEndpoint(id, draft);
      }
      // Save the key only when typed; leaving it blank keeps the existing one on edit.
      if (apiKey.trim()) await setCloudKey(id, apiKey.trim());
      reset();
    } catch (e) {
      error = String(e);
    }
    busy = false;
  }

  async function remove(id: string) {
    if (busy) return;
    busy = true;
    try {
      await removeCloudEndpoint(id);
    } finally {
      busy = false;
    }
  }
</script>

{#if editing}
    <div class="ep-form">
      <p class="ep-formnote">
        An <strong>OpenAI-compatible</strong> endpoint — base URL + key, like Cline or Ollama. Backs
        cloud transcription and AI notes/assist.
      </p>
      <label class="ep-field">
        <span class="ep-label">Name</span>
        <input class="ep-in" bind:value={name} placeholder="e.g. My gateway" />
      </label>
      <label class="ep-field">
        <span class="ep-label">Base URL</span>
        <input class="ep-in" bind:value={baseUrl} placeholder="http://host:port/v1" />
      </label>
      <label class="ep-field">
        <span class="ep-label">Model id</span>
        <input class="ep-in" bind:value={model} placeholder="e.g. gpt-4o-mini / metis-coder" />
      </label>
      <label class="ep-field">
        <span class="ep-label">API key {#if editing !== "new"}<em>(leave blank to keep)</em>{/if}</span>
        <input class="ep-in" type="password" bind:value={apiKey} placeholder="sk-…" />
      </label>
      <details class="ep-adv">
        <summary class="ep-adv-sum">Advanced — assist parameters &amp; transcription</summary>

        <span class="ep-subhead">AI notes / assist</span>
        <div class="ep-grid">
          <label class="ep-field">
            <span class="ep-label">Temperature</span>
            <input class="ep-in" type="number" step="0.1" min="0" max="2" bind:value={temperature} placeholder="0.3" />
          </label>
          <label class="ep-field">
            <span class="ep-label">Max reply tokens</span>
            <input class="ep-in" type="number" min="0" bind:value={maxTokens} placeholder="provider default" />
          </label>
          <label class="ep-field">
            <span class="ep-label">Context size (tokens)</span>
            <input class="ep-in" type="number" min="0" bind:value={contextTokens} placeholder="no limit" />
          </label>
          <label class="ep-field">
            <span class="ep-label">top_p</span>
            <input class="ep-in" type="number" step="0.05" min="0" max="1" bind:value={topP} placeholder="provider default" />
          </label>
        </div>
        <label class="ep-field">
          <span class="ep-label">System prompt</span>
          <textarea class="ep-in ep-area" rows="2" bind:value={systemPrompt}
            placeholder="Standing instruction prepended to every assist task (persona, language, style)."
          ></textarea>
        </label>
        <p class="ep-hint">
          Set <strong>Context size</strong> and a long meeting is summarized in chunks (map-reduce)
          instead of overflowing the model — it covers the whole transcript.
        </p>

        <span class="ep-subhead">Transcription API shape</span>
        <div class="ep-seg">
          <button class:active={protocol === "openai"} onclick={() => (protocol = "openai")}>
            Audio transcriptions
          </button>
          <button class:active={protocol === "chat"} onclick={() => (protocol = "chat")}>
            Chat with audio
          </button>
        </div>
        <p class="ep-hint">
          Both are OpenAI-compatible — this only affects <strong>File / Live transcription</strong>:
          <strong>Audio transcriptions</strong> calls <code>/audio/transcriptions</code> (Whisper-style),
          <strong>Chat with audio</strong> calls <code>/chat/completions</code> with an audio part. For
          AI notes/assist (text chat) it doesn't matter.
        </p>
      </details>

      {#if error}<div class="ep-error">{error}</div>{/if}

      <div class="ep-actions">
        <button class="ep-ghost" onclick={reset} disabled={busy}>Cancel</button>
        <button
          class="ep-save"
          disabled={busy || !name.trim() || !baseUrl.trim() || !model.trim()}
          onclick={save}>{editing === "new" ? "Add" : "Save"}</button
        >
      </div>
    </div>
  {:else}
    <p class="ep-intro">Keys are stored only on this device, and sent only to the provider they belong to.</p>

    <div class="ep-section">
      <span class="ep-head">Built-in</span>
      <ul class="ep-list">
        {#each builtins as p (p.id)}
          <li class="ep-row">
            <div class="ep-info">
              <span class="ep-name">{p.name}</span>
              {#if keyEditId !== p.id}
                <span class="ep-meta">
                  {#if p.keySet}<span class="ep-keyed">{p.keyHint}</span>{:else}<span class="ep-nokey"
                      >no key yet</span
                    >{/if}
                </span>
              {/if}
            </div>

            {#if keyEditId === p.id}
              <div class="ep-keyedit">
                <div class="ep-keyfield">
                  <!-- svelte-ignore a11y_autofocus -->
                  <input
                    class="ep-in"
                    type={reveal ? "text" : "password"}
                    autocomplete="off"
                    autocapitalize="off"
                    spellcheck="false"
                    autofocus
                    placeholder="Paste API key"
                    bind:value={keyDraft}
                    onkeydown={(e) => {
                      if (e.key === "Enter") saveKey(p.id);
                      if (e.key === "Escape") cancelKey();
                    }}
                  />
                  <button class="ep-reveal" type="button" onclick={() => (reveal = !reveal)}>
                    {reveal ? "Hide" : "Show"}
                  </button>
                </div>
                <button class="ep-mini accent" onclick={() => saveKey(p.id)} disabled={!keyDraft.trim()}>Save</button>
                <button class="ep-mini" onclick={cancelKey}>Cancel</button>
              </div>
            {:else}
              <div class="ep-row-actions">
                {#if p.keySet}
                  <button class="ep-mini" onclick={() => startKey(p.id)}>Edit</button>
                  <button class="ep-mini danger" onclick={() => setCloudKey(p.id, "")}>Remove</button>
                {:else}
                  <button class="ep-mini" onclick={() => getKey(p.keysUrl)}>Get a key ↗</button>
                  <button class="ep-mini accent" onclick={() => startKey(p.id)}>Add key</button>
                {/if}
              </div>
            {/if}
          </li>
        {/each}
      </ul>
    </div>

    <div class="ep-section">
      <span class="ep-head">OpenAI-compatible endpoints</span>
      {#if customs.length}
        <ul class="ep-list">
          {#each customs as p (p.id)}
            <li class="ep-row">
              <div class="ep-info">
                <span class="ep-name">{p.name}</span>
                <span class="ep-meta">
                  {p.models[0]?.id ?? "—"} · {p.baseUrl}
                  {#if p.keySet}· <span class="ep-keyed">key set</span>{:else}·
                    <span class="ep-nokey">no key</span>{/if}
                </span>
              </div>
              <div class="ep-row-actions">
                <button class="ep-mini" onclick={() => startEdit(p)}>Edit</button>
                <button class="ep-mini danger" onclick={() => remove(p.id)} disabled={busy}>Remove</button>
              </div>
            </li>
          {/each}
        </ul>
      {:else}
        <p class="ep-empty">
          Add your own OpenAI-compatible endpoint — your gateway, a local Ollama
          (<code>http://localhost:11434/v1</code>), or OpenAI.
        </p>
      {/if}
      <button class="ep-add" onclick={startAdd}>+ Add OpenAI Compatible Endpoint</button>
    </div>
  {/if}

<style>
  .ep-form {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .ep-formnote {
    margin: 0;
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--muted);
  }

  .ep-hint code {
    font-family: var(--font-mono, monospace);
    font-size: 11px;
  }

  .ep-adv {
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 0 11px;
  }

  .ep-adv[open] {
    padding-bottom: 12px;
  }

  .ep-adv-sum {
    cursor: pointer;
    padding: 10px 2px;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--muted);
    list-style: none;
  }

  .ep-adv-sum::-webkit-details-marker {
    display: none;
  }

  .ep-adv-sum::before {
    content: "▸ ";
  }

  .ep-adv[open] .ep-adv-sum::before {
    content: "▾ ";
  }

  .ep-subhead {
    display: block;
    margin: 12px 0 8px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }

  .ep-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 10px;
  }

  .ep-area {
    resize: vertical;
    line-height: 1.45;
  }

  .ep-adv .ep-hint {
    margin-top: 8px;
  }

  .ep-field {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .ep-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
  }

  .ep-label em {
    font-weight: 400;
    font-style: normal;
    opacity: 0.8;
  }

  .ep-in {
    width: 100%;
    box-sizing: border-box;
    font-family: inherit;
    font-size: 13px;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 7px 10px;
  }

  .ep-in:focus {
    outline: none;
    border-color: var(--accent);
  }

  .ep-seg {
    display: flex;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    overflow: hidden;
  }

  .ep-seg button {
    flex: 1;
    font-family: inherit;
    font-size: 12.5px;
    color: var(--muted);
    background: var(--surface);
    border: none;
    padding: 7px 8px;
    cursor: pointer;
  }

  .ep-seg button.active {
    color: white;
    background: var(--accent);
  }

  .ep-hint {
    margin: 0;
    font-size: 11.5px;
    line-height: 1.45;
    color: var(--muted);
  }

  .ep-error {
    font-size: 12.5px;
    color: var(--stop);
    background: color-mix(in srgb, var(--stop) 9%, var(--bg));
    border: 1px solid color-mix(in srgb, var(--stop) 35%, var(--border));
    border-radius: 8px;
    padding: 8px 11px;
    word-break: break-word;
  }

  .ep-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }

  .ep-ghost,
  .ep-save {
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    border-radius: 8px;
    padding: 7px 15px;
    cursor: pointer;
    border: 1px solid transparent;
  }

  .ep-ghost {
    color: var(--muted);
    background: transparent;
    border-color: var(--border-strong);
  }

  .ep-save {
    color: white;
    background: var(--accent);
  }

  .ep-save:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .ep-intro {
    margin: 0 0 4px;
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--muted);
  }

  .ep-section {
    margin-top: 14px;
  }

  .ep-head {
    display: block;
    margin-bottom: 8px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }

  .ep-list {
    list-style: none;
    margin: 0 0 10px;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .ep-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 9px 11px;
    border: 1px solid var(--border);
    border-radius: 9px;
  }

  .ep-info {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .ep-name {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text);
  }

  .ep-meta {
    font-size: 11.5px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .ep-keyed {
    color: var(--accent);
    font-family: var(--font-mono);
  }

  .ep-nokey {
    color: var(--stop);
  }

  .ep-row-actions {
    flex: none;
    display: flex;
    gap: 6px;
  }

  .ep-keyedit {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .ep-keyfield {
    position: relative;
    flex: 1;
    min-width: 0;
  }

  .ep-keyfield .ep-in {
    padding-right: 52px;
    background: var(--bg);
  }

  .ep-reveal {
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

  .ep-reveal:hover {
    color: var(--text);
  }

  .ep-mini {
    font-family: inherit;
    font-size: 12px;
    color: var(--muted);
    background: transparent;
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 4px 10px;
    cursor: pointer;
  }

  .ep-mini:hover {
    color: var(--text);
    border-color: var(--accent);
  }

  .ep-mini.accent {
    color: white;
    background: var(--accent);
    border-color: var(--accent);
  }

  .ep-mini.accent:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .ep-mini.danger:hover {
    color: var(--stop);
    border-color: var(--stop);
  }

  .ep-empty {
    margin: 0 0 10px;
    font-size: 13px;
    line-height: 1.5;
    color: var(--muted);
  }

  .ep-empty code {
    font-family: var(--font-mono, monospace);
    font-size: 12px;
  }

  .ep-add {
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    color: var(--accent);
    background: transparent;
    border: 1px dashed var(--border-strong);
    border-radius: 9px;
    padding: 9px 12px;
    cursor: pointer;
    width: 100%;
  }

  .ep-add:hover {
    border-color: var(--accent);
  }
</style>
