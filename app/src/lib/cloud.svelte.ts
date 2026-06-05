// Shared, reactive cloud state — one source of truth for the provider/model catalog, each
// provider's key status, and the single global "API keys" modal. Live, File, the cloud picker, and
// the key manager all read and mutate this, so saving a key in one place updates every other.

import { invoke } from "@tauri-apps/api/core";

export type CloudModel = {
  id: string;
  name: string;
  streaming: boolean;
  batch: boolean;
  description: string;
  /** The provider's recommended default — the picker pre-selects it over list order. */
  recommended: boolean;
  /** The model returns speaker labels itself — the UI hides local diarization when it's picked. */
  diarizes: boolean;
  /** User-added (not in the built-in catalog) — the picker tags it and offers removal. */
  custom: boolean;
};

/** AI notes/assist tuning for a custom endpoint's chat model. `null`/empty = use the built-in
 *  default; never sent to the provider. Mirrors the backend `AssistParams`. */
export type AssistParams = {
  temperature: number | null;
  maxTokens: number | null;
  /** Context window in tokens; when set, a transcript over it is map-reduced instead of truncated. */
  contextTokens: number | null;
  topP: number | null;
  systemPrompt: string;
};

/** A blank assist config (all defaults) — the starting point for a new endpoint. */
export function emptyAssist(): AssistParams {
  return { temperature: null, maxTokens: null, contextTokens: null, topP: null, systemPrompt: "" };
}

export type CloudProvider = {
  id: string;
  name: string;
  keySet: boolean;
  /** Masked form of the saved key (e.g. `sk-…a1b2`), or null when no key is saved. */
  keyHint: string | null;
  /** The provider's "API keys" console page, for the "Get a key" link. */
  keysUrl: string;
  /** User-added custom OpenAI-compatible endpoint (not a built-in catalog provider). */
  custom: boolean;
  /** Base URL + normalized protocol (`"openai"`/`"chat"`/`"gemini"`) — pre-fills the edit form. */
  baseUrl: string;
  protocol: string;
  /** AI notes/assist tuning — meaningful for custom endpoints; default for catalog providers. */
  assist: AssistParams;
  models: CloudModel[];
};

// A `$state` object (not a bare primitive) so the proxy is shared across every importer and stays
// reactive — and so `cloudState.endpointsOpen` is bindable from the Modal.
export const cloudState = $state<{ providers: CloudProvider[]; endpointsOpen: boolean }>({
  providers: [],
  endpointsOpen: false,
});

/** Reload the provider catalog and key-status flags from the backend. */
export async function refreshCloud(): Promise<void> {
  cloudState.providers = await invoke<CloudProvider[]>("list_cloud_providers");
}

/** Save (non-empty `key`) or clear (empty `key`) a provider's API key on this device, then refresh. */
export async function setCloudKey(providerId: string, key: string): Promise<void> {
  await invoke("set_cloud_key", { provider: providerId, key: key.trim() });
  await refreshCloud();
}

/**
 * Add a custom model id for a provider so it's usable immediately, with no app update — the cloud
 * adapter routes by the provider's protocol, so any new id of a known provider just works. Throws
 * (with the backend's message) on a blank or duplicate id; refreshes the catalog on success.
 */
export async function addCloudCustomModel(providerId: string, modelId: string, name: string): Promise<void> {
  await invoke("add_cloud_custom_model", { provider: providerId, modelId: modelId.trim(), name: name.trim() });
  await refreshCloud();
}

/** Remove a previously added custom cloud model id, then refresh the catalog. */
export async function removeCloudCustomModel(providerId: string, modelId: string): Promise<void> {
  await invoke("remove_cloud_custom_model", { provider: providerId, modelId });
  await refreshCloud();
}

/** The editable fields of a custom endpoint. `protocol` is `"openai"` (the `/audio/transcriptions`
 *  shape) or `"chat"` (`/chat/completions` with audio); `assist` tunes the AI notes/assist calls. */
export type EndpointDraft = {
  name: string;
  baseUrl: string;
  protocol: string;
  model: string;
  assist: AssistParams;
};

/**
 * Add a custom OpenAI-compatible endpoint and return its generated id. Throws (with the backend's
 * message) on invalid input; refreshes the catalog on success.
 */
export async function addCloudEndpoint(input: EndpointDraft): Promise<string> {
  const id = await invoke<string>("add_cloud_endpoint", { input });
  await refreshCloud();
  return id;
}

/** Update a custom endpoint in place (keeping its id and stored key), then refresh the catalog. */
export async function updateCloudEndpoint(id: string, input: EndpointDraft): Promise<void> {
  await invoke("update_cloud_endpoint", { id, input });
  await refreshCloud();
}

/** Remove a custom endpoint (and its stored key), then refresh the catalog. */
export async function removeCloudEndpoint(id: string): Promise<void> {
  await invoke("remove_cloud_endpoint", { id });
  await refreshCloud();
}

/**
 * Run a one-shot LLM task over `transcript` using the chat model of a provider (a custom endpoint —
 * the user's gateway, a local Ollama, or OpenAI). `systemPrompt` steers the task (summary, action
 * items, or a custom prompt). Returns the assistant's reply; throws (with the backend's message) on
 * a missing key or an HTTP error.
 */
export async function runLlmTask(provider: string, model: string, systemPrompt: string, transcript: string, params: Record<string, ParamValue> = {}): Promise<string> {
  return await invoke<string>("run_llm_task", { provider, model, systemPrompt, transcript, params });
}

/**
 * Streaming variant of {@link runLlmTask}: the reply streams back via `assist://delta` events (and
 * closes on `assist://text`), so the caller's feed fills token-by-token instead of waiting for the
 * whole reply. Resolves when the reply completes; throws (with the backend's message) on error.
 */
export async function runAssistStream(provider: string, model: string, systemPrompt: string, transcript: string, params: Record<string, ParamValue> = {}): Promise<void> {
  await invoke("run_assist_stream", { provider, model, systemPrompt, transcript, params });
}

/**
 * Open the one global "AI models & endpoints" manager from anywhere. It lists every model source —
 * built-in providers (set a key) and custom OpenAI-compatible endpoints (full config) — so keys and
 * endpoints are managed in a single generic place.
 */
export function openEndpointsModal(): void {
  cloudState.endpointsOpen = true;
}

// ── Generic engine parameters (the "advanced settings" schema) ──────────────────────────────────
// An engine declares its tunables as ParamSpecs; the generic <ParamsPanel> renders them and the
// user's values ride to the backend as overrides. The same panel serves any engine — new knobs are
// backend-only.

export type ParamValue = number | boolean | string;

export type EnumOption = { value: string; label: string };

export type ParamSpec = {
  key: string;
  label: string;
  help: string;
  kind: "float" | "int" | "bool" | "enum" | "text" | "textarea";
  min: number;
  max: number;
  step: number;
  options: EnumOption[];
  default: ParamValue;
  advanced: boolean;
};

/** The advanced live-streaming parameter specs a provider's `model` exposes (empty if it can't stream). */
export async function streamingParams(providerId: string, model: string): Promise<ParamSpec[]> {
  if (!providerId || !model) return [];
  try {
    return await invoke<ParamSpec[]>("streaming_params", { provider: providerId, model });
  } catch {
    return [];
  }
}

/** The advanced File (batch) parameter specs a provider's `model` exposes (empty if none). */
export async function batchParams(providerId: string, model: string): Promise<ParamSpec[]> {
  if (!providerId || !model) return [];
  try {
    return await invoke<ParamSpec[]>("batch_params", { provider: providerId, model });
  } catch {
    return [];
  }
}

/** The advanced parameter specs the AI assist exposes (temperature / top_p / max reply tokens). The
 *  same generic <ParamsPanel> renders them; empty on error so the panel just shows nothing. Vendor-
 *  agnostic — every assist provider speaks the same OpenAI-compatible chat tuning. */
export async function assistParams(): Promise<ParamSpec[]> {
  try {
    return await invoke<ParamSpec[]>("assist_params");
  } catch {
    return [];
  }
}

/** The advanced parameter specs the **realtime** assist exposes (turn-detection + noise reduction) —
 *  the realtime counterpart of {@link assistParams}, rendered by the same generic <ParamsPanel>. */
export async function assistRealtimeParams(): Promise<ParamSpec[]> {
  try {
    return await invoke<ParamSpec[]>("assist_realtime_params");
  } catch {
    return [];
  }
}

/** Each spec's smart default, as a flat key→value map. */
export function defaultParamValues(specs: ParamSpec[]): Record<string, ParamValue> {
  return Object.fromEntries(specs.map((s) => [s.key, s.default]));
}

/** Only the params whose value differs from its spec default — what we actually send to the provider.
 *  A param left at its default is omitted, so the model applies its own (optimal) default rather than
 *  a fabricated value it might reject (e.g. a model that only accepts temperature = 1). */
export function changedParamValues(
  values: Record<string, ParamValue>,
  specs: ParamSpec[],
): Record<string, ParamValue> {
  const defaults = new Map(specs.map((s) => [s.key, s.default]));
  return Object.fromEntries(
    Object.entries(values).filter(([k, v]) => !defaults.has(k) || v !== defaults.get(k)),
  );
}

const PARAMS_KEY = "wisp.params";

/** Storage key for a (provider, model); `scope` ("live"/"file") namespaces it so the two surfaces
 *  don't overwrite each other's values for a model that does both. Empty scope = the legacy key. */
function paramsKey(provider: string, model: string, scope: string): string {
  return scope ? `${scope}:${provider}/${model}` : `${provider}/${model}`;
}

/** Saved parameter values for a (provider, model), or `{}` — overlaid on the spec defaults. */
export function loadParamValues(provider: string, model: string, scope = ""): Record<string, ParamValue> {
  try {
    const all = JSON.parse(localStorage.getItem(PARAMS_KEY) ?? "{}");
    return all?.[paramsKey(provider, model, scope)] ?? {};
  } catch {
    return {};
  }
}

/** Persists parameter values for a (provider, model) on this device. */
export function saveParamValues(
  provider: string,
  model: string,
  values: Record<string, ParamValue>,
  scope = "",
): void {
  try {
    const all = JSON.parse(localStorage.getItem(PARAMS_KEY) ?? "{}");
    all[paramsKey(provider, model, scope)] = values;
    localStorage.setItem(PARAMS_KEY, JSON.stringify(all));
  } catch {
    // localStorage unavailable — params just won't persist; not fatal.
  }
}

/** The provider with `id`, if present in the catalog. */
export function cloudProvider(id: string): CloudProvider | undefined {
  return cloudState.providers.find((p) => p.id === id);
}

/**
 * Whether a cloud transcription is ready to run: the provider has a saved key and the chosen model
 * exists with the needed capability. File needs `batch`. Live (`streaming`) takes either — a realtime
 * model streams, and any batch model runs segment-batch (local VAD + a per-utterance cloud call).
 */
export function cloudReady(providerId: string, modelId: string, capability: "batch" | "streaming"): boolean {
  const provider = cloudProvider(providerId);
  if (!provider?.keySet) return false;
  const model = provider.models.find((m) => m.id === modelId);
  if (!model) return false;
  return capability === "streaming" ? model.streaming || model.batch : model.batch;
}
