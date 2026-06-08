<script lang="ts">
  // The Library — a browsable, searchable archive of finished notes. Reads from the SQLite-backed
  // store via the note commands; search uses the backend's full-text index (with a short-CJK
  // substring fallback). List ⇄ detail in one view; deletes go through a confirm modal.
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { i18n } from "$lib/i18n.svelte";
  import Modal from "$lib/Modal.svelte";

  type NoteSummary = {
    id: string;
    title: string;
    started_at_ms: number;
    duration_ms: number;
    language: string | null;
    engine: string | null;
    preview: string;
  };
  type SearchHit = {
    meeting_id: string;
    title: string;
    started_at_ms: number;
    snippet: string;
    score: number;
  };
  type Note = {
    id: string;
    title: string;
    started_at_ms: number;
    duration_ms: number;
    language: string | null;
    engine: string | null;
    summary: string | null;
    segment_count: number;
  };
  type Segment = {
    idx: number;
    start_ms: number;
    end_ms: number;
    speaker: number | null;
    source: string;
    text: string;
  };
  type Detail = { meeting: Note; segments: Segment[] };

  let notes = $state<NoteSummary[]>([]);
  let query = $state("");
  let hits = $state<SearchHit[] | null>(null); // null = not searching; [] = searched with no results
  let detail = $state<Detail | null>(null);
  let error = $state("");
  let pendingDelete = $state<string | null>(null);
  let confirmOpen = $state(false);
  let searchTimer: ReturnType<typeof setTimeout> | undefined;

  async function loadList() {
    try {
      notes = await invoke<NoteSummary[]>("list_library_notes");
      error = "";
    } catch (e) {
      error = String(e);
    }
  }

  onMount(loadList);

  // Debounced so each keystroke doesn't hit the database.
  function onSearchInput() {
    clearTimeout(searchTimer);
    const q = query.trim();
    if (!q) {
      hits = null;
      return;
    }
    searchTimer = setTimeout(async () => {
      try {
        hits = await invoke<SearchHit[]>("search_library", { query: q, limit: 50 });
        error = "";
      } catch (e) {
        error = String(e);
      }
    }, 200);
  }

  async function openNote(id: string) {
    try {
      detail = await invoke<Detail | null>("get_library_note", { id });
      error = "";
    } catch (e) {
      error = String(e);
    }
  }

  function askDelete(id: string) {
    pendingDelete = id;
    confirmOpen = true;
  }

  async function doDelete() {
    const id = pendingDelete;
    confirmOpen = false;
    pendingDelete = null;
    if (!id) return;
    try {
      await invoke<boolean>("delete_library_note", { id });
      if (detail?.meeting.id === id) detail = null;
      await loadList();
      if (query.trim()) onSearchInput();
    } catch (e) {
      error = String(e);
    }
  }

  function fmtDate(ms: number): string {
    return new Date(ms).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function fmtDuration(ms: number): string {
    const total = Math.round(ms / 1000);
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  function speakerLabel(source: string): string {
    if (source === "mic") return i18n.t.library.you;
    if (source === "system") return i18n.t.library.them;
    return "";
  }

  // The FTS snippet wraps matches in «…»; escape the (user-content) text, then render those as <mark>.
  function renderSnippet(s: string): string {
    const escaped = s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    return escaped.replaceAll("«", "<mark>").replaceAll("»", "</mark>");
  }
</script>

<section class="library">
  {#if detail}
    <header class="lib-head detail-head">
      <button class="back" onclick={() => (detail = null)}>← {i18n.t.library.back}</button>
      <div class="lib-titles">
        <h2 class="lib-h2">{detail.meeting.title}</h2>
        <span class="lib-meta">
          {fmtDate(detail.meeting.started_at_ms)} · {fmtDuration(detail.meeting.duration_ms)}{detail
            .meeting.engine
            ? ` · ${detail.meeting.engine}`
            : ""}
        </span>
      </div>
      <button class="del-btn" onclick={() => askDelete(detail!.meeting.id)}>
        {i18n.t.library.delete}
      </button>
    </header>

    {#if detail.meeting.summary}
      <div class="summary">{detail.meeting.summary}</div>
    {/if}

    <div class="transcript">
      {#each detail.segments as seg (seg.idx)}
        <p class="seg">
          {#if speakerLabel(seg.source)}
            <span class="spk" class:them={seg.source === "system"}>{speakerLabel(seg.source)}</span>
          {/if}
          <span class="txt">{seg.text}</span>
        </p>
      {/each}
    </div>
  {:else}
    <header class="lib-head">
      <h2 class="lib-h2">{i18n.t.library.title}</h2>
      <input
        class="search"
        type="search"
        placeholder={i18n.t.library.searchPlaceholder}
        bind:value={query}
        oninput={onSearchInput}
      />
    </header>

    {#if error}
      <div class="err">{error}</div>
    {/if}

    {#if hits !== null}
      {#if hits.length === 0}
        <div class="empty">{i18n.t.library.noResults}</div>
      {:else}
        <ul class="cards">
          {#each hits as hit (hit.meeting_id)}
            <li>
              <button class="card hit-card" onclick={() => openNote(hit.meeting_id)}>
                <div class="card-top">
                  <span class="card-title">{hit.title}</span>
                  <span class="card-date">{fmtDate(hit.started_at_ms)}</span>
                </div>
                <!-- eslint-disable-next-line svelte/no-at-html-tags -->
                <div class="snippet">{@html renderSnippet(hit.snippet)}</div>
              </button>
            </li>
          {/each}
        </ul>
      {/if}
    {:else if notes.length === 0}
      <div class="empty">{i18n.t.library.empty}</div>
    {:else}
      <ul class="cards">
        {#each notes as m (m.id)}
          <li class="row">
            <button class="card card-main" onclick={() => openNote(m.id)}>
              <div class="card-top">
                <span class="card-title">{m.title}</span>
                <span class="card-date">{fmtDate(m.started_at_ms)}</span>
              </div>
              <div class="card-sub">
                {fmtDuration(m.duration_ms)}{m.engine ? ` · ${m.engine}` : ""}
              </div>
              {#if m.preview}<div class="preview">{m.preview}</div>{/if}
            </button>
            <button class="del-x" onclick={() => askDelete(m.id)} title={i18n.t.library.delete}>×</button>
          </li>
        {/each}
      </ul>
    {/if}
  {/if}

  <Modal bind:open={confirmOpen} title={i18n.t.library.deleteTitle}>
    <p class="confirm-text">{i18n.t.library.deleteConfirm}</p>
    <div class="confirm-actions">
      <button class="btn" onclick={() => (confirmOpen = false)}>{i18n.t.library.cancel}</button>
      <button class="btn danger" onclick={doDelete}>{i18n.t.library.delete}</button>
    </div>
  </Modal>
</section>

<style>
  .library {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 24px clamp(16px, 4vw, 48px);
    overflow-y: auto;
  }

  .lib-head {
    display: flex;
    align-items: center;
    gap: 14px;
    flex: none;
  }

  .lib-h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
  }

  .search {
    margin-left: auto;
    width: min(360px, 50%);
    padding: 9px 13px;
    font: inherit;
    font-size: 14px;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    outline: none;
  }
  .search:focus {
    border-color: var(--accent);
  }

  .cards {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .row {
    display: flex;
    align-items: stretch;
    gap: 6px;
  }

  .card {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
    text-align: left;
    padding: 13px 15px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    cursor: pointer;
    font: inherit;
    color: var(--text);
    transition:
      border-color 0.15s,
      background 0.15s;
  }
  .card:hover {
    border-color: var(--border-strong);
    background: var(--surface-active);
  }

  .card-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
  }
  .card-title {
    font-size: 14.5px;
    font-weight: 600;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .card-date {
    flex: none;
    font-size: 12px;
    color: var(--muted);
  }
  .card-sub {
    font-size: 12px;
    color: var(--muted);
  }
  .preview,
  .snippet {
    font-size: 13px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .snippet :global(mark) {
    background: color-mix(in srgb, var(--accent) 26%, transparent);
    color: var(--text);
    border-radius: 3px;
    padding: 0 1px;
  }

  .del-x {
    flex: none;
    width: 34px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    color: var(--muted);
    font-size: 18px;
    line-height: 1;
    cursor: pointer;
    transition:
      color 0.15s,
      border-color 0.15s,
      background 0.15s;
  }
  .del-x:hover {
    color: var(--stop);
    border-color: var(--stop);
  }

  .empty {
    margin: 32px auto;
    max-width: 360px;
    text-align: center;
    font-size: 14px;
    color: var(--muted);
  }
  .err {
    font-size: 13px;
    color: var(--stop);
  }

  .detail-head {
    align-items: flex-start;
  }
  .back {
    flex: none;
    padding: 7px 11px;
    font: inherit;
    font-size: 13px;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 9px;
    cursor: pointer;
  }
  .back:hover {
    color: var(--text);
    border-color: var(--border-strong);
  }
  .lib-titles {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .lib-meta {
    font-size: 12px;
    color: var(--muted);
  }
  .del-btn {
    margin-left: auto;
    flex: none;
    padding: 7px 13px;
    font: inherit;
    font-size: 13px;
    color: var(--stop);
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 9px;
    cursor: pointer;
  }
  .del-btn:hover {
    border-color: var(--stop);
  }

  .summary {
    padding: 13px 15px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    font-size: 13.5px;
    line-height: 1.6;
    color: var(--text);
    white-space: pre-wrap;
  }

  .transcript {
    display: flex;
    flex-direction: column;
    gap: 9px;
  }
  .seg {
    margin: 0;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text);
  }
  .spk {
    display: inline-block;
    margin-right: 8px;
    font-size: 11px;
    font-weight: 600;
    color: var(--live);
  }
  .spk.them {
    color: var(--accent);
  }

  .confirm-text {
    margin: 0;
    font-size: 14px;
    line-height: 1.5;
    color: var(--text);
  }
  .confirm-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }
  .btn {
    padding: 8px 15px;
    font: inherit;
    font-size: 13px;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 9px;
    cursor: pointer;
  }
  .btn:hover {
    border-color: var(--border-strong);
  }
  .btn.danger {
    color: #fff;
    background: var(--stop);
    border-color: var(--stop);
  }
  .btn.danger:hover {
    filter: brightness(1.05);
  }
</style>
