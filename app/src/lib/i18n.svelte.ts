// Shared, reactive UI language — one source of truth for the active locale and its message
// catalogue. Mirrors the `cloud.svelte.ts` pattern: a module-level rune singleton every component
// reads. Zero dependencies; the catalogues are plain objects (see ./i18n/*.ts).
//
// ── To add a language: create ./i18n/<id>.ts (copy en.ts, translate, keep every key — it's typed
//    `Messages`, so a missing key won't compile), then import it and add ONE row to REGISTRY below.
//    The Locale type, the switcher menu, and validation all derive from REGISTRY — nothing else
//    changes, not even app.html. ──

import { en, type Messages } from "./i18n/en";
import { zhHans } from "./i18n/zh-Hans";
import { zhHant } from "./i18n/zh-Hant";

/** Every locale: its own name (shown in the menu) + its catalogue. Insertion order = menu order. */
const REGISTRY = {
  "en": { label: "English", messages: en },
  "zh-Hans": { label: "简体中文", messages: zhHans },
  "zh-Hant": { label: "繁體中文", messages: zhHant },
} satisfies Record<string, { label: string; messages: Messages }>;

export type Locale = keyof typeof REGISTRY;

/** Switcher menu, derived from REGISTRY. */
export const LOCALES: ReadonlyArray<{ id: Locale; label: string }> = (Object.keys(REGISTRY) as Locale[]).map(
  (id) => ({ id, label: REGISTRY[id].label }),
);

/** localStorage key for the saved locale. The inline script in app.html reads the same key. */
export const LOCALE_STORAGE_KEY = "wisp.locale";

function isLocale(value: string | null | undefined): value is Locale {
  return value != null && value in REGISTRY;
}

// The inline script in app.html already put the saved locale on <html lang> before first paint;
// mirror it so the first render matches. Falls back to English for anything unrecognised.
function initialLocale(): Locale {
  if (typeof document !== "undefined" && isLocale(document.documentElement.lang)) {
    return document.documentElement.lang;
  }
  return "en";
}

class I18n {
  locale = $state<Locale>(initialLocale());
  /** The active catalogue. Components read `i18n.t.…`; it re-resolves when the locale changes. */
  t = $derived(REGISTRY[this.locale].messages);

  set(next: Locale) {
    this.locale = next;

    if (typeof document !== "undefined") document.documentElement.lang = next;

    try {
      localStorage.setItem(LOCALE_STORAGE_KEY, next);
    } catch {
      /* storage unavailable — keep the choice for this session only */
    }
  }
}

export const i18n = new I18n();
