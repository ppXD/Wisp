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
  /** User-added (not in the built-in catalog) — the picker tags it and offers removal. */
  custom: boolean;
};

export type CloudProvider = {
  id: string;
  name: string;
  keySet: boolean;
  /** Masked form of the saved key (e.g. `sk-…a1b2`), or null when no key is saved. */
  keyHint: string | null;
  /** The provider's "API keys" console page, for the "Get a key" link. */
  keysUrl: string;
  models: CloudModel[];
};

// A `$state` object (not a bare primitive) so the proxy is shared across every importer and stays
// reactive — and so `cloudState.keyModalOpen` is bindable from the Modal.
export const cloudState = $state<{ providers: CloudProvider[]; keyModalOpen: boolean }>({
  providers: [],
  keyModalOpen: false,
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

/** Open the single global "Cloud API keys" modal from anywhere. */
export function openKeyModal(): void {
  cloudState.keyModalOpen = true;
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
  kind: "float" | "int" | "bool" | "enum" | "text";
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

/** Each spec's smart default, as a flat key→value map. */
export function defaultParamValues(specs: ParamSpec[]): Record<string, ParamValue> {
  return Object.fromEntries(specs.map((s) => [s.key, s.default]));
}

const PARAMS_KEY = "wisp.params";

/** Saved parameter values for a (provider, model), or `{}` — overlaid on the spec defaults. */
export function loadParamValues(provider: string, model: string): Record<string, ParamValue> {
  try {
    const all = JSON.parse(localStorage.getItem(PARAMS_KEY) ?? "{}");
    return all?.[`${provider}/${model}`] ?? {};
  } catch {
    return {};
  }
}

/** Persists parameter values for a (provider, model) on this device. */
export function saveParamValues(provider: string, model: string, values: Record<string, ParamValue>): void {
  try {
    const all = JSON.parse(localStorage.getItem(PARAMS_KEY) ?? "{}");
    all[`${provider}/${model}`] = values;
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
 * exists with the needed capability (`batch` for File, `streaming` for Live).
 */
export function cloudReady(providerId: string, modelId: string, capability: "batch" | "streaming"): boolean {
  const provider = cloudProvider(providerId);
  if (!provider?.keySet) return false;
  const model = provider.models.find((m) => m.id === modelId);
  return !!model && (capability === "streaming" ? model.streaming : model.batch);
}
