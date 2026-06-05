<script lang="ts">
  // AI assist over the transcript via a chosen provider + model — the same catalog the cloud
  // transcription picker uses (OpenAI, Groq, …, and custom endpoints) — through the shared
  // `runLlmTask` (chat) path. Independent of whatever engine is transcribing (device or cloud).
  //
  // One generic flow: pick a model, write/insert a prompt (the built-in tasks are just templates),
  // then Run once over the whole transcript, or — in `live` mode — turn on Live for a rolling prompter.
  // Every result is appended to a scrolling feed (newest at the bottom, like live subtitles). The
  // assist has its own Stop (halt the rolling) and Clear (empty the feed), separate from the session.
  import {
    cloudState,
    runLlmTask,
    runAssistStream,
    openEndpointsModal,
    assistParams,
    defaultParamValues,
    changedParamValues,
    loadParamValues,
    saveParamValues,
    type ParamSpec,
    type ParamValue,
  } from "$lib/cloud.svelte";
  import ParamsPanel from "$lib/ParamsPanel.svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";

  // `transcript`: one line per turn. `live`: enable the rolling prompter + Stop control.
  // `sessionRunning`: the live session is active — when it stops, the assist rolling stops with it.
  let {
    transcript,
    live = false,
    sessionRunning = true,
  }: { transcript: string; live?: boolean; sessionRunning?: boolean } = $props();

  // The assist's OWN model catalog — separate from transcription. `chat` models run via the polling
  // `/chat/completions` path (option A); `realtime` models would run via the official realtime
  // WebSocket (option B, coming). A custom endpoint contributes its own (chat) model. The provider's
  // base URL + key come from the cloud catalog; only these providers speak `/chat/completions`.
  type AssistModel = { id: string; label: string; kind: "chat" | "realtime" };
  const ASSIST_CATALOG: Record<string, AssistModel[]> = {
    openai: [
      { id: "gpt-4o-mini", label: "GPT-4o mini", kind: "chat" },
      { id: "gpt-4o", label: "GPT-4o", kind: "chat" },
      { id: "gpt-4.1-mini", label: "GPT-4.1 mini", kind: "chat" },
      { id: "gpt-4.1", label: "GPT-4.1", kind: "chat" },
      { id: "gpt-5-mini", label: "GPT-5 mini", kind: "chat" },
      { id: "gpt-5", label: "GPT-5", kind: "chat" },
      { id: "o4-mini", label: "o4-mini", kind: "chat" },
      // GA realtime models (the `session: { type: realtime }` shape the assist WebSocket speaks).
      { id: "gpt-realtime-mini", label: "GPT Realtime mini", kind: "realtime" },
      { id: "gpt-realtime", label: "GPT Realtime", kind: "realtime" },
    ],
    groq: [
      { id: "llama-3.3-70b-versatile", label: "Llama 3.3 70B", kind: "chat" },
      { id: "llama-3.1-8b-instant", label: "Llama 3.1 8B", kind: "chat" },
    ],
    qwen: [
      { id: "qwen-max", label: "Qwen Max", kind: "chat" },
      { id: "qwen-plus", label: "Qwen Plus", kind: "chat" },
    ],
    // Google/Gemini is intentionally absent: it speaks neither `/chat/completions` (chat assist) nor
    // the OpenAI realtime protocol (realtime assist), so no assist path is wired for it yet.
  };

  const PROVIDER_KEY = "wisp.assistProvider";
  const MODEL_KEY = "wisp.assistModel";
  let providerId = $state(localStorage.getItem(PROVIDER_KEY) ?? "");
  let modelId = $state(localStorage.getItem(MODEL_KEY) ?? "");

  // Providers that can host the assist, each with its assist models (catalog chat/realtime, or a
  // custom endpoint's own model).
  const providerAssist = $derived(
    cloudState.providers
      .map((p) => ({
        provider: p,
        models:
          ASSIST_CATALOG[p.id] ??
          (p.custom
            ? p.models.map((m) => ({ id: m.id, label: m.name, kind: "chat" as const }))
            : []),
      }))
      .filter((g) => g.models.length > 0),
  );
  const provider = $derived(cloudState.providers.find((p) => p.id === providerId));
  const assistModels = $derived(providerAssist.find((g) => g.provider.id === providerId)?.models ?? []);
  const model = $derived(assistModels.find((m) => m.id === modelId));
  const selectedKind = $derived(model?.kind ?? "chat");

  // Default to a sensible pick (prefer a custom endpoint), keep the model valid, persist.
  $effect(() => {
    if (!provider || !assistModels.length) {
      const fb = providerAssist.find((g) => g.provider.custom) ?? providerAssist[0];
      if (fb) {
        providerId = fb.provider.id;
        modelId = fb.models[0]?.id ?? "";
      }
      return;
    }
    if (!model) {
      modelId = assistModels[0]?.id ?? "";
      return;
    }
    localStorage.setItem(PROVIDER_KEY, providerId);
    localStorage.setItem(MODEL_KEY, modelId);
  });

  // Built-in prompts are just starting points — pick one, then edit freely. "Summary" is one of many.
  const TEMPLATES = [
    {
      icon: "✦",
      label: "Summary",
      prompt:
        "Summarize the conversation so far: a one or two sentence overview, then 3-8 bullet points of \
the key topics, decisions, and outcomes. Reply in the spoken language. Markdown only, no preamble.",
    },
    {
      icon: "☑",
      label: "Action items",
      prompt:
        "Extract concrete action items as a Markdown checklist in the spoken language. Format each as \
'- [ ] <task> — <owner if named> (<due if mentioned>)'. Only real, actionable tasks; if none, reply \
'No action items found.' No preamble.",
    },
    {
      icon: "◎",
      label: "Live hints (coach)",
      prompt:
        "You are a live meeting copilot. From the recent conversation, give 2-4 very brief, actionable \
hints for the speaker right now — a question to ask next, a fact to verify, a point to clarify, or a \
risk to flag. Reply in the spoken language as a short bullet list, no preamble.",
    },
    {
      icon: "⊞",
      label: "Decisions & owners",
      prompt:
        "List every decision made and who owns it as a Markdown table with columns Decision | Owner | \
Notes, in the spoken language. If there are no decisions, say so. No preamble.",
    },
    {
      icon: "⇄",
      label: "Translate to English",
      prompt: "Translate the transcript into clear, natural English, preserving speaker turns. No preamble.",
    },
    { icon: "✎", label: "Blank", prompt: "" },
  ];

  // The realtime prompt ships with the full grounding rules visible + editable right here — there is NO
  // hidden backend instruction. A realtime voice model defaults to chatting (it will greet you and
  // invent smalltalk), so this hard wall has to live where you can read and tune every word of it.
  const REALTIME_DEFAULT =
    "You are a silent meeting-notes tool, not a chat assistant. The people speaking (\"Me\" and \"Them\") \
are talking to each other, NOT to you — never greet them, reply to them, answer their questions, or \
continue their conversation. Use ONLY the transcript lines you are given; never invent, guess, or pad. \
If nothing meaningful has been said yet, reply with just \"—\".\n\nTask: from the conversation so far, \
give 2-4 very short, useful live notes — a key point, a decision, an open question, or a fact worth \
checking. Bullet list, in the spoken language, no preamble.";

  // Chat and realtime keep separate prompts (the grounded realtime wall vs. a plain chat task), each
  // persisted under its own key, so switching model kind never clobbers the other.
  const CHAT_PROMPT_KEY = "wisp.assistPrompt";
  const RT_PROMPT_KEY = "wisp.assistPromptRealtime";
  const promptKey = $derived(selectedKind === "realtime" ? RT_PROMPT_KEY : CHAT_PROMPT_KEY);
  const promptDefault = $derived(selectedKind === "realtime" ? REALTIME_DEFAULT : TEMPLATES[0].prompt);

  let prompt = $state("");
  let loadedPromptKey = $state("");
  // Load that kind's saved prompt when the kind changes (also populates it on mount); persist edits
  // under the active key, but not during a key swap, so the two prompts never cross-contaminate.
  $effect(() => {
    if (promptKey !== loadedPromptKey) {
      loadedPromptKey = promptKey;
      prompt = localStorage.getItem(promptKey) ?? promptDefault;
    }
  });
  $effect(() => {
    if (promptKey === loadedPromptKey) localStorage.setItem(promptKey, prompt);
  });

  // Advanced model params (temperature / top_p / max reply tokens) — the same generic schema + panel
  // the transcription picker uses, so a new knob is backend-only. Specs are vendor-agnostic; values
  // persist per (provider, model). Chat only — realtime tuning rides its own path.
  let assistParamSpecs = $state<ParamSpec[]>([]);
  let assistParamValues = $state<Record<string, ParamValue>>({});
  let advancedOpen = $state(false);

  // Load the specs + this (provider, model)'s saved values whenever the model changes; empty specs
  // (non-chat, or none configured) hide the Advanced disclosure.
  $effect(() => {
    const p = providerId,
      m = modelId;
    if (selectedKind !== "chat" || !p || !m) {
      assistParamSpecs = [];
      assistParamValues = {};
      return;
    }
    assistParams().then((specs) => {
      assistParamSpecs = specs;
      assistParamValues = { ...defaultParamValues(specs), ...loadParamValues(p, m, "assist") };
    });
  });
  // Persist edits for the active (provider, model).
  $effect(() => {
    if (assistParamSpecs.length && providerId && modelId) {
      saveParamValues(providerId, modelId, assistParamValues, "assist");
    }
  });
  // Only the knobs the user moved off their default — an untouched knob is omitted so the model
  // applies its own optimum (and a reasoning model that rejects a non-default value isn't sent one).
  const assistOverrides = $derived(changedParamValues(assistParamValues, assistParamSpecs));

  let modelOpen = $state(false);
  let templatesOpen = $state(false);
  let running = $state(false);
  let error = $state("");

  // The assist output feed — every run appends an entry; newest scrolls into view at the bottom. A
  // `streaming` entry is still being filled token-by-token (realtime deltas) and shows a live cursor.
  let feed = $state<Array<{ id: number; at: number; text: string; streaming?: boolean }>>([]);
  let nextId = 0;
  let streamId = $state<number | null>(null); // the entry currently being streamed into, if any
  let feedEl = $state<HTMLElement>();
  let stick = $state(true); // auto-scroll only while the user is at the bottom

  // Live rolling.
  let liveOn = $state(false);
  let connecting = $state(false); // Start pressed, establishing the first response (not yet "live")
  // True while a realtime (WebSocket) assist is the live one — so Stop closes the socket (not just
  // the chat rolling), and the rolling timer below stays off (realtime pushes via events instead).
  let runningRealtime = $state(false);
  const INTERVAL = 12_000; // ms between rolling refreshes

  // Rolling context = a summary-buffer memory (the SOTA pattern for unbounded conversations): the most
  // recent turns are fed verbatim, while anything older is folded into a running summary. So the assist
  // never loses early context (the old blind last-N-chars window dropped it), the input stays bounded,
  // and a multi-hour meeting costs the same per tick as a short one. `summarizedChars` tracks how much
  // of the (append-only) transcript prefix is already folded in.
  let runningSummary = $state("");
  let summarizedChars = $state(0);
  const RECENT_BUDGET = 6000; // chars kept verbatim; older turns get compressed into the summary
  const SUMMARY_PROMPT =
    "You maintain a running summary of a live conversation. Merge the existing summary with the new \
turns into one updated summary that preserves decisions, action items, names, numbers, and open \
questions; drop chit-chat. Be concise. Reply in the conversation's language. Output only the summary.";

  const collapsed = $derived(live && liveOn);
  // A realtime model listens to live audio, so it only works inside a running Live session (never in
  // File mode, whose transcript is static). Otherwise Start is disabled and a hint explains — a chat
  // model still runs over the transcript anywhere.
  const realtimeNeedsSession = $derived(selectedKind === "realtime" && (!live || !sessionRunning));

  function pick(pId: string, mId: string) {
    providerId = pId;
    modelId = mId;
    modelOpen = false;
  }

  function useTemplate(p: string) {
    prompt = p;
    templatesOpen = false;
  }

  function clock(at: number): string {
    const d = new Date(at);
    return `${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  function onFeedScroll() {
    if (!feedEl) return;
    stick = feedEl.scrollHeight - feedEl.scrollTop - feedEl.clientHeight < 48;
  }

  // Keep the newest entry in view while the user is parked at the bottom (don't yank them if they
  // scrolled up to read). Runs after the DOM updates whenever the feed grows.
  $effect(() => {
    feed.length;
    if (stick && feedEl) {
      requestAnimationFrame(() => {
        if (feedEl) feedEl.scrollTop = feedEl.scrollHeight;
      });
    }
  });

  // One assist call over `text`; the reply streams into the feed (via the assist:// events). Skips
  // when busy or unconfigured. On error, closes any half-streamed entry so it doesn't dangle.
  async function call(text: string) {
    if (!provider || !model || running) return;
    if (!provider.keySet) {
      error = `Add an API key for ${provider.name} first.`;
      liveOn = false;
      return;
    }
    if (!text.trim() || !prompt.trim()) return;
    running = true;
    error = "";
    try {
      await runAssistStream(provider.id, model.id, prompt, text, assistOverrides);
    } catch (e) {
      error = String(e);
      closeDanglingStream();
    }
    running = false;
  }

  // Finalise a streaming entry that never got its closing `assist://text` (an errored stream), so it
  // stops showing the live cursor. No-op when nothing is streaming.
  function closeDanglingStream() {
    if (streamId === null) return;
    const entry = feed.find((f) => f.id === streamId);
    if (entry) entry.streaming = false;
    streamId = null;
  }

  // The largest line-aligned cut at or before `maxLen`, so whole transcript turns (one per line) are
  // folded into the summary, never half an utterance. 0 if no boundary fits (wait for more).
  function lineAlignedCut(text: string, maxLen: number): number {
    if (maxLen >= text.length) return text.length;
    const nl = text.lastIndexOf("\n", maxLen);
    return nl <= 0 ? 0 : nl + 1;
  }

  // Fold `newText` (older turns aging out of the verbatim window) into the running summary. Best-effort:
  // on failure return the prior summary unchanged so the tick still produces a hint and we retry later.
  async function summarize(prior: string, newText: string): Promise<string | null> {
    if (!provider || !model) return null;
    const input = prior ? `Existing summary:\n${prior}\n\nNew turns to fold in:\n${newText}` : newText;
    try {
      return (await runLlmTask(provider.id, model.id, SUMMARY_PROMPT, input, assistOverrides)).trim();
    } catch {
      return null;
    }
  }

  // One rolling pass with summary-buffer memory: fold any overflow beyond the recent window into the
  // running summary, then run the user's task over (summary + recent) and append the result. Owns the
  // `running` busy flag for the whole pass so the timer can't re-enter it mid-flight.
  async function rollOnce() {
    if (!provider || !model || running || !prompt.trim()) return;
    if (!provider.keySet) {
      error = `Add an API key for ${provider.name} first.`;
      liveOn = false;
      return;
    }

    running = true;
    error = "";
    try {
      const full = transcript;
      // A new/cleared session shrinks the transcript → reset the memory so stale context can't leak in.
      if (full.length < summarizedChars) {
        runningSummary = "";
        summarizedChars = 0;
      }

      let recent = full.slice(summarizedChars);
      const cut = recent.length > RECENT_BUDGET ? lineAlignedCut(recent, recent.length - RECENT_BUDGET) : 0;
      if (cut > 0) {
        const folded = await summarize(runningSummary, recent.slice(0, cut));
        if (folded !== null) {
          runningSummary = folded;
          summarizedChars += cut;
          recent = full.slice(summarizedChars);
        }
      }

      const context = runningSummary
        ? `Summary of earlier conversation:\n${runningSummary}\n\nRecent conversation:\n${recent}`
        : recent;
      if (!context.trim()) return;

      await runAssistStream(provider.id, model.id, prompt, context, assistOverrides);
    } catch (e) {
      error = String(e);
      closeDanglingStream();
    } finally {
      running = false;
    }
  }

  // One generic Start — the calling method is chosen by the model + session: a real-time model opens
  // the official WebSocket and listens live (option B); a chat model rolls by polling the transcript
  // during a live session (option A), or runs a single pass over a finished transcript. While Start is
  // establishing the first response it shows "Connecting…"; only on success does it switch to Stop.
  async function start() {
    if (connecting || running || !provider?.keySet || !model || !prompt.trim()) return;
    error = "";
    // A real-time model opens the official WebSocket and listens to the live audio (option B); a chat
    // model rolls by polling the transcript (option A). One button, dispatched by the model's kind.
    if (selectedKind === "realtime") {
      await startRealtime();
      return;
    }
    if (!sessionRunning) {
      await call(transcript); // static transcript → a single pass (the feed shows Working…)
      return;
    }
    connecting = true;
    await rollOnce(); // first hint — the "connect" — also seeds the summary-buffer memory
    connecting = false;
    if (!error) liveOn = true; // succeeded → go live (collapses; the bar shows Stop)
  }

  // Open the realtime assist: connect the WebSocket (the spinner shows "Connecting…"), and only once
  // the handshake succeeds switch to live (collapsed, the bar shows Stop). The backend taps the live
  // session's mic+system audio and emits each finalised reply as an `assist://text` event.
  async function startRealtime() {
    if (!provider || !model || !live || !sessionRunning) return;
    connecting = true;
    try {
      await invoke("start_assist_realtime", {
        provider: provider.id,
        model: model.id,
        instructions: prompt,
      });
      runningRealtime = true;
      liveOn = true;
    } catch (e) {
      error = String(e);
    }
    connecting = false;
  }

  // Close the realtime assist's WebSocket. Best-effort — even if the call fails, drop the live state.
  async function stopRealtime() {
    runningRealtime = false;
    try {
      await invoke("stop_assist_realtime");
    } catch {
      // already torn down (e.g. the session ended) — nothing to do.
    }
  }

  // The assist's own Stop: close the realtime socket if that's what's running, then drop the live
  // state (which also stops the chat rolling). Shared by the collapsed Stop button.
  function stopAssist() {
    if (runningRealtime) void stopRealtime();
    liveOn = false;
  }

  // Pull a realtime reply on demand — the assist otherwise volunteers hints on a throttle, so this is
  // the "answer right now" button (e.g. "what should I ask next?"). Best-effort; ignore if not running.
  async function hintNow() {
    try {
      await invoke("assist_hint_now");
    } catch {
      // assist not running — nothing to pull.
    }
  }

  function clearFeed() {
    feed = [];
    error = "";
    streamId = null;
  }

  async function copyEntry(text: string) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // clipboard unavailable — silently ignore.
    }
  }

  // The global Stop closes the session — wind the assist down with it. The backend already tore down
  // the realtime worker (Stop stops the assist too), so just clear the live flags; no further chat
  // calls go out either. The panel stays open for an after-the-fact summary.
  $effect(() => {
    if (!sessionRunning && (liveOn || runningRealtime)) {
      liveOn = false;
      runningRealtime = false;
    }
  });

  // While Live (live mode only) and the session is running, re-run on a timer whenever the transcript
  // has grown since the last hint. Realtime assist is excluded — it streams replies via events, not by
  // polling the transcript. Only `live`/`liveOn`/`sessionRunning`/`selectedKind` are tracked here;
  // `transcript` is read inside the timer.
  $effect(() => {
    if (!live || !liveOn || !sessionRunning || selectedKind === "realtime") return;
    let lastLen = -1;
    let stopped = false;
    const tick = () => {
      if (stopped || running) return;
      const t = transcript.trim();
      if (!t || t.length === lastLen) return;
      lastLen = t.length;
      void rollOnce(); // summary-buffer memory over the grown transcript
    };
    const id = setInterval(tick, INTERVAL); // Start already did the first pass (the "connect")
    return () => {
      stopped = true;
      clearInterval(id);
    };
  });

  // Subscribe to the realtime assist's backend events for this component's lifetime: a reply streams in
  // as `assist://delta` chunks and closes on `assist://text`; an async failure (`assist://error`, e.g.
  // a dropped socket mid-session) surfaces in the dismissible error box and drops the live state.
  onMount(() => {
    let offDelta: (() => void) | undefined;
    let offText: (() => void) | undefined;
    let offError: (() => void) | undefined;

    // A chunk of the in-progress reply — open a streaming entry on the first chunk, then append.
    void listen<string>("assist://delta", (e) => {
      const chunk = e.payload ?? "";
      if (!chunk) return;
      if (streamId === null) {
        streamId = nextId++;
        feed.push({ id: streamId, at: Date.now(), text: chunk, streaming: true });
      } else {
        const entry = feed.find((f) => f.id === streamId);
        if (entry) entry.text += chunk;
      }
    }).then((off) => (offDelta = off));

    // The reply closed — finalise the streamed entry with the authoritative text, or (if nothing
    // streamed) append it whole.
    void listen<string>("assist://text", (e) => {
      const text = (e.payload ?? "").trim();
      const entry = streamId !== null ? feed.find((f) => f.id === streamId) : undefined;
      if (entry) {
        if (text) entry.text = text;
        entry.streaming = false;
      } else if (text) {
        feed.push({ id: nextId++, at: Date.now(), text });
      }
      streamId = null;
    }).then((off) => (offText = off));

    void listen<string>("assist://error", (e) => {
      error = String(e.payload ?? "assist error");
      streamId = null;
      runningRealtime = false;
      liveOn = false;
    }).then((off) => (offError = off));

    return () => {
      offDelta?.();
      offText?.();
      offError?.();
    };
  });
</script>

{#if !providerAssist.length}
  <div class="empty">
    <span class="empty-icon">✦</span>
    <p>
      Add an AI model for notes and live hints — your gateway, a local Ollama, or OpenAI.
      <button class="link" onclick={openEndpointsModal}>Manage in ✦ Models</button>.
    </p>
  </div>
{:else}
  <!-- Control bar: the model, plus the assist's own Stop (rolling) and Clear (feed). -->
  <div class="ctl">
    <div class="model">
      <button class="model-trigger" class:open={modelOpen} onclick={() => (modelOpen = !modelOpen)}>
        <span class="model-name">{model ? `${provider?.name} · ${model.label}` : "Pick a model"}</span>
        <span class="caret"></span>
      </button>
      {#if modelOpen}
        <button class="scrim" aria-label="Close" onclick={() => (modelOpen = false)}></button>
        <div class="menu">
          {#each providerAssist as g (g.provider.id)}
            <span class="menu-group">
              {g.provider.name}{#if !g.provider.keySet}<span class="grp-warn"> · ⚠ needs key</span>{/if}
            </span>
            {#each g.models as m (m.id)}
              <button
                class="menu-opt"
                class:sel={g.provider.id === providerId && m.id === modelId}
                onclick={() => pick(g.provider.id, m.id)}
              >
                <span class="opt-name">
                  {#if m.kind === "realtime"}<span class="rt-tag">⚡</span>{/if}{m.label}
                </span>
                <span class="opt-model">{m.id}</span>
              </button>
            {/each}
          {/each}
          <button class="menu-manage" onclick={() => ((modelOpen = false), openEndpointsModal())}>
            ✦ Manage models &amp; endpoints…
          </button>
        </div>
      {/if}
    </div>

    {#if collapsed}
      <div class="ctl-right">
        {#if runningRealtime}
          <button class="hint" onclick={hintNow} title="Pull a reply now">✨ Hint</button>
        {/if}
        <button class="stop" onclick={stopAssist}>◼ Stop</button>
        <button class="clear" onclick={clearFeed} disabled={!feed.length}>Clear</button>
      </div>
    {/if}
  </div>

  {#if !collapsed}
    {#if provider && !provider.keySet}
      <button class="keyrow" onclick={openEndpointsModal}>⚠ {provider.name} needs an API key — add it</button>
    {/if}

    <div class="phead">
      <span class="plabel">Prompt</span>
      <div class="tmpl">
        <button class="tmpl-trigger" class:open={templatesOpen} onclick={() => (templatesOpen = !templatesOpen)}>
          Templates <span class="caret"></span>
        </button>
        {#if templatesOpen}
          <button class="scrim" aria-label="Close" onclick={() => (templatesOpen = false)}></button>
          <div class="menu tmpl-menu">
            {#each TEMPLATES as t (t.label)}
              <button class="menu-opt" onclick={() => useTemplate(t.prompt)}>
                <span class="opt-mark">{t.icon}</span><span class="opt-name">{t.label}</span>
              </button>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <textarea
      class="prompt"
      rows="3"
      bind:value={prompt}
      placeholder="What should the assistant do with the transcript? Pick a template above or write your own."
    ></textarea>

    <!-- Advanced model params (temperature / top_p / max tokens), rendered from the generic schema —
         grouped with the prompt as configuration, above the action. Chat models only; a knob left at
         its default isn't sent (the model uses its own optimum). -->
    {#if selectedKind === "chat" && assistParamSpecs.length}
      <div class="adv">
        <button class="adv-trigger" class:open={advancedOpen} onclick={() => (advancedOpen = !advancedOpen)}>
          <span class="caret"></span> Advanced
        </button>
        {#if advancedOpen}
          <div class="adv-body">
            <ParamsPanel specs={assistParamSpecs} bind:values={assistParamValues} />
          </div>
        {/if}
      </div>
    {/if}

    <!-- One generic Start, dispatched by the model's kind: a chat model runs/rolls over the transcript,
         a real-time model opens the live-audio WebSocket. Real-time needs a running session to listen to. -->
    <div class="actions">
      <button
        class="run"
        disabled={connecting || running || !provider?.keySet || !model || !prompt.trim() || realtimeNeedsSession}
        onclick={start}
      >
        {#if connecting || running}<span class="btn-spin"></span>{connecting ? "Connecting…" : "Working…"}{:else}{selectedKind === "realtime" ? "⚡ Start" : "▸ Start"}{/if}
      </button>
      <button class="clear act-clear" onclick={clearFeed} disabled={!feed.length}>Clear</button>
    </div>

    {#if realtimeNeedsSession}
      <p class="rt-note">⚡ Real-time assist listens to live audio — use it in a running Live session.</p>
    {/if}
  {/if}

  {#if error}
    <div class="out-error">
      <span class="out-error-msg">{error}</span>
      <button class="out-error-x" aria-label="Dismiss" onclick={() => (error = "")}>×</button>
    </div>
  {/if}

  <!-- The scrolling feed of assist outputs (newest at the bottom). -->
  <div class="feed" bind:this={feedEl} onscroll={onFeedScroll}>
    {#each feed as e (e.id)}
      <div class="entry">
        <div class="entry-meta">
          <span class="entry-time">{clock(e.at)}</span>
          <button class="entry-copy" aria-label="Copy" onclick={() => copyEntry(e.text)}>⧉</button>
        </div>
        <div class="entry-text">{e.text}{#if e.streaming}<span class="cursor"></span>{/if}</div>
      </div>
    {:else}
      {#if !running}
        <p class="feed-hint">
          {#if collapsed}Listening — hints will appear here as you talk.{:else}Press <em>Start</em>
            {sessionRunning ? "for rolling hints from the conversation" : "to summarize the transcript"}.{/if}
        </p>
      {/if}
    {/each}
    {#if running && streamId === null}<div class="working"><span class="spin"></span>Working…</div>{/if}
  </div>
{/if}

<style>
  .empty {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 18px 16px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
  }

  .empty-icon {
    font-size: 16px;
    color: var(--accent);
  }

  .empty p {
    margin: 0;
  }

  .link {
    font: inherit;
    color: var(--accent);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
  }

  .link:hover {
    text-decoration: underline;
  }

  /* ── Control bar: model + Stop + Clear ── */
  .ctl {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 14px 0;
  }

  .ctl-right {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .stop {
    font-family: inherit;
    font-size: 12px;
    font-weight: 600;
    color: var(--stop);
    background: transparent;
    border: 1px solid var(--stop);
    border-radius: 7px;
    padding: 4px 11px;
    cursor: pointer;
  }

  .hint {
    font-family: inherit;
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 10%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent) 35%, var(--border));
    border-radius: 7px;
    padding: 4px 11px;
    cursor: pointer;
  }

  .hint:hover {
    background: color-mix(in srgb, var(--accent) 18%, transparent);
  }

  .clear {
    font-family: inherit;
    font-size: 12px;
    color: var(--muted);
    background: transparent;
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 4px 10px;
    cursor: pointer;
  }

  .clear:hover:not(:disabled) {
    color: var(--text);
    border-color: var(--accent);
  }

  .clear:disabled {
    opacity: 0.45;
    cursor: default;
  }

  /* ── Model dropdown ── */
  .model {
    position: relative;
  }

  .model-trigger {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    max-width: 190px;
    font-family: inherit;
    font-size: 12.5px;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 999px;
    padding: 5px 12px;
    cursor: pointer;
  }

  .model-name {
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

  .model-trigger.open .caret,
  .tmpl-trigger.open .caret {
    transform: rotate(180deg);
  }

  .scrim {
    position: fixed;
    inset: 0;
    z-index: 10;
    background: transparent;
    border: none;
    cursor: default;
  }

  .menu {
    position: absolute;
    z-index: 11;
    top: calc(100% + 6px);
    left: 0;
    min-width: 230px;
    max-width: min(340px, 84vw);
    max-height: 52vh;
    overflow-x: hidden;
    overflow-y: auto;
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    padding: 6px;
    box-shadow: 0 12px 28px rgb(0 0 0 / 18%);
  }

  .menu-group {
    display: block;
    padding: 8px 8px 4px;
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }

  .grp-warn {
    color: var(--stop);
    text-transform: none;
    letter-spacing: 0;
    font-weight: 500;
  }

  .menu-opt {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    min-width: 0;
    text-align: left;
    font-family: inherit;
    font-size: 13px;
    color: var(--text);
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 7px 8px;
    cursor: pointer;
  }

  .menu-opt:hover {
    background: var(--surface-active);
  }

  .menu-opt.sel {
    background: color-mix(in srgb, var(--accent) 12%, transparent);
  }

  .opt-mark {
    flex: none;
    width: 14px;
    font-size: 11px;
    color: var(--muted);
  }

  .opt-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .opt-model {
    flex: 0 1 auto;
    min-width: 0;
    max-width: 46%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
  }

  .menu-manage {
    width: 100%;
    text-align: left;
    margin-top: 4px;
    border-top: 1px solid var(--border);
    padding: 8px;
    font-family: inherit;
    font-size: 12.5px;
    color: var(--accent);
    background: transparent;
    border-left: none;
    border-right: none;
    border-bottom: none;
    cursor: pointer;
  }

  .menu-manage:hover {
    text-decoration: underline;
  }

  /* ── Key-missing inline ── */
  .keyrow {
    display: block;
    width: calc(100% - 28px);
    text-align: left;
    margin: 10px 14px 0;
    font-family: inherit;
    font-size: 12.5px;
    color: var(--stop);
    background: color-mix(in srgb, var(--stop) 8%, var(--bg));
    border: 1px solid color-mix(in srgb, var(--stop) 30%, var(--border));
    border-radius: 8px;
    padding: 8px 11px;
    cursor: pointer;
  }

  /* ── Prompt ── */
  .phead {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 14px 6px;
  }

  .plabel {
    flex: 1;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }

  .tmpl {
    position: relative;
  }

  .tmpl-trigger {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: inherit;
    font-size: 12px;
    color: var(--accent);
    background: transparent;
    border: none;
    padding: 2px 4px;
    cursor: pointer;
  }

  .tmpl-trigger .caret {
    background-color: var(--accent);
  }

  .tmpl-menu {
    min-width: 190px;
    left: auto;
    right: 0;
  }

  .prompt {
    width: calc(100% - 28px);
    box-sizing: border-box;
    margin: 0 14px;
    resize: vertical;
    font-family: inherit;
    font-size: 13px;
    line-height: 1.5;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 9px;
    padding: 10px 12px;
  }

  .prompt:focus {
    outline: none;
    border-color: var(--accent);
  }

  .actions {
    display: flex;
    gap: 8px;
    padding: 12px 14px 0;
  }

  .run {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    border-radius: 8px;
    padding: 7px 15px;
    cursor: pointer;
    color: white;
    background: var(--accent);
    border: 1px solid var(--accent);
  }

  .run:disabled {
    opacity: 0.7;
    cursor: default;
  }

  .btn-spin {
    width: 12px;
    height: 12px;
    border: 2px solid rgb(255 255 255 / 45%);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  .out-error {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin: 10px 14px 0;
    color: var(--stop);
    background: color-mix(in srgb, var(--stop) 9%, var(--bg));
    border: 1px solid color-mix(in srgb, var(--stop) 35%, var(--border));
    border-radius: 8px;
    padding: 9px 12px;
  }

  .out-error-msg {
    flex: 1;
    min-width: 0;
    max-height: 116px;
    overflow-y: auto;
    font-size: 12.5px;
    line-height: 1.5;
    word-break: break-word;
  }

  .out-error-x {
    flex: none;
    font-size: 16px;
    line-height: 1;
    color: var(--stop);
    background: transparent;
    border: none;
    padding: 0 2px;
    cursor: pointer;
    opacity: 0.7;
  }

  .out-error-x:hover {
    opacity: 1;
  }

  .act-clear {
    margin-left: auto;
  }

  .rt-tag {
    margin-right: 5px;
    color: var(--accent);
  }

  .rt-note {
    margin: 8px 14px 0;
    font-size: 12px;
    line-height: 1.4;
    color: var(--muted);
  }

  /* ── Advanced model params disclosure ── */
  .adv {
    margin: 10px 14px 0;
  }

  .adv-trigger {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: inherit;
    font-size: 12px;
    color: var(--muted);
    background: transparent;
    border: none;
    padding: 2px 0;
    cursor: pointer;
  }

  .adv-trigger:hover {
    color: var(--text);
  }

  .adv-trigger.open .caret {
    transform: rotate(180deg);
  }

  .adv-body {
    margin-top: 10px;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: var(--surface);
  }

  /* ── Scrolling output feed ── */
  .feed {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 12px 14px 14px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .entry {
    border-left: 2px solid color-mix(in srgb, var(--accent) 45%, var(--border));
    padding-left: 11px;
  }

  .entry-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }

  .entry-time {
    font-size: 11px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .entry-copy {
    font-family: inherit;
    font-size: 11px;
    color: var(--muted);
    background: transparent;
    border: none;
    padding: 0 2px;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.15s;
  }

  .entry:hover .entry-copy {
    opacity: 1;
  }

  .entry-copy:hover {
    color: var(--accent);
  }

  .entry-text {
    font-size: 13.5px;
    line-height: 1.6;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* A blinking caret on a reply that's still streaming in. */
  .cursor {
    display: inline-block;
    width: 2px;
    height: 1em;
    margin-left: 1px;
    vertical-align: text-bottom;
    background: var(--accent);
    animation: blink 1s steps(2, start) infinite;
  }

  @keyframes blink {
    50% {
      opacity: 0;
    }
  }

  .feed-hint {
    margin: 0;
    font-size: 13px;
    color: var(--muted);
  }

  .working {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 13px;
    color: var(--muted);
  }

  .spin {
    width: 13px;
    height: 13px;
    border: 2px solid color-mix(in srgb, var(--accent) 35%, transparent);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
