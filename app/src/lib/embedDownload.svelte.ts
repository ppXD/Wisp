// A single in-flight embedding-model download, tracked at module scope so it survives the Settings
// dialog (and the picker component) closing and reopening. The progress listener is subscribed once
// and never torn down, so events keep updating `dl` even while the picker isn't mounted.
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export const dl = $state<{
  id: string | null;
  label: string;
  downloaded: number;
  total: number;
  error: string;
}>({ id: null, label: "", downloaded: 0, total: 0, error: "" });

export function dlPct(): number {
  return dl.total > 0 ? Math.min(100, Math.round((dl.downloaded / dl.total) * 100)) : 0;
}

let listening = false;
async function ensureListener() {
  if (listening) return;
  listening = true;
  await listen<{ id: string; downloaded: number; total: number }>("embed-download://progress", (e) => {
    if (dl.id === e.payload.id) {
      dl.downloaded = e.payload.downloaded;
      dl.total = e.payload.total;
    }
  });
}

// Downloads then activates a model. Resolves true on success. Module-scoped, so it runs to
// completion (and keeps reporting progress) even if the picker unmounts mid-download.
export async function runDownload(id: string, label: string): Promise<boolean> {
  if (dl.id) return false;
  await ensureListener();
  dl.id = id;
  dl.label = label;
  dl.downloaded = 0;
  dl.total = 0;
  dl.error = "";
  try {
    await invoke("download_embedding_model", { id });
    await invoke("set_embedding_model", { id });
    return true;
  } catch (e) {
    dl.error = String(e);
    return false;
  } finally {
    dl.id = null;
  }
}
