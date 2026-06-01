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

/** Open the single global "Cloud API keys" modal from anywhere. */
export function openKeyModal(): void {
  cloudState.keyModalOpen = true;
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
