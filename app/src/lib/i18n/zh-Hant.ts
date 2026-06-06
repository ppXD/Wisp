// 繁體中文 (Traditional Chinese). Typed `Messages`, so it must mirror en.ts exactly — a missing or
// mistyped key is a compile error. To add a language, copy this file and translate the values.

import type { Messages } from "./en";

export const zhHant: Messages = {
  nav: {
    collapse: "收合",
    expand: "展開",
    collapseSidebar: "收合側欄",
    expandSidebar: "展開側欄",
    live: "即時",
    file: "檔案",
    settings: "設定",
    themeToLight: "切換到淺色主題",
    themeToDark: "切換到深色主題",
    themeToggle: "切換主題",
    lightMode: "淺色模式",
    darkMode: "深色模式",
    language: "語言",
    languageMenu: "選擇語言",
  },

  live: {
    loadingModels: "正在載入模型…",

    // 刪除模型確認對話框（Live + File 選擇器共用）。
    deleteModel: {
      trashTitle: (size: string): string => `刪除模型 · 釋放 ${size}`,
      trashAria: (name: string, size: string): string => `刪除 ${name}，釋放 ${size}`,
      title: "刪除模型？",
      body: (name: string, size: string): string => `刪除 ${name} 並釋放 ${size} 磁碟空間？`,
      sub: "模型仍保留在目錄中 —— 隨時可重新下載。",
      confirm: "刪除",
      deleting: "刪除中…",
      freed: (size: string, name: string): string => `已釋放 ${size} —— 已刪除 ${name}`,
    },

    you: "我",
    them: "對方",
    youTip: (on, running) =>
      on
        ? running
          ? "我（麥克風）已開啟 —— 點擊靜音"
          : "我（麥克風）已開啟 —— 點擊以排除轉寫"
        : running
          ? "我（麥克風）已靜音 —— 點擊取消靜音"
          : "我（麥克風）已關閉 —— 點擊以納入轉寫",
    themTip: (on, running) =>
      on
        ? running
          ? "對方（系統音訊）已開啟 —— 點擊靜音"
          : "對方（系統音訊）已開啟 —— 點擊以排除轉寫"
        : running
          ? "對方（系統音訊）已靜音 —— 點擊取消靜音"
          : "對方（系統音訊）已關閉 —— 點擊以納入轉寫",

    status: {
      ready: "就緒",
      keyNeeded: "需要金鑰",
      noModel: "無模型",
      recording: "錄音中",
    },

    empty: {
      before: "選個模型，按 ",
      action: "開始",
      after: "，然後開口說話。",
    },

    start: "開始轉寫",
    stop: "停止轉寫",
    startConnecting: "連線中…",
    startDownloading: "正在下載模型…",
    startSlowHint: "正在載入模型 —— 首次執行可能需要幾秒",

    advanced: "進階 · 音訊、語言、說話人",
    advancedCloud: "音訊 · 裝置",
  },

  file: {
    dropTitle: "點擊選擇檔案，或拖入此處",
    options: "選項 · 精度、提示、說話人",
    optionsCloud: "選項 · 提示、說話人",
    subCloudReady: { before: "mp3、m4a、wav、flac、mp4、mov… 傳送至 ", after: "。" },
    subCloudPick: "請在上方選擇一個雲端模型。",
    subCloudNoKey: (provider) => `新增你的 ${provider} API 金鑰以在雲端轉寫。`,
    subLocalReady: { before: "mp3、m4a、wav、flac、mp4、mov… 由 ", after: " 在本機轉寫。" },
    subLocalMissing: { before: "", after: " 尚未下載 —— 在下方取得後即可轉寫。" },
  },

  picker: {
    active: "使用中",
    needsKey: "需要金鑰",
    recommended: "推薦",
    custom: "自訂",
    removeModel: (name) => `移除 ${name}`,
    displayName: "顯示名稱（可選）",
    provider: "提供商",
    model: "模型",
    noModel: "無模型",
    manageModels: "✦ 管理模型與端點…",
    addCustom: "+ 自訂模型…",
  },

  params: {
    title: "參數",
    reset: "恢復預設",
  },

  settings: {
    aiModels: "AI 模型",
    dictation: "聽寫",
    dictationIntro: "按住快捷鍵說話、放開 —— Wisp 會把它輸入到目前聚焦的 app，全程在裝置端（Apple 語音）。",
    dictationNote: "聽寫需要 Apple 裝置端語音（macOS 26 或更新版本）。",
    pushToTalk: "按住說話",
    on: "開",
    off: "關",
    hotkey: "快捷鍵",
    accessibilityNote: "⚠ 需要「輔助使用」權限才能輸入到其他 app。",
    openSettings: "打開系統設定",
  },

  assist: {
    emptyText: "新增一個 AI 模型用於筆記和即時提示 —— 你的閘道、本地 Ollama，或 OpenAI。",
    manageInModels: "在 ✦ Models 中管理",
    needsKey: (name) => `⚠ ${name} 需要 API 金鑰 —— 點此新增`,
    apiKeyNeeded: "需要 API 金鑰",
    hint: "提示",
    hintNow: "立即產生回覆",
    stop: "停止",
    prompt: "提示詞",
    templates: "範本",
    advanced: "進階",
    promptPlaceholder: "希望助手對轉寫內容做什麼？在上方選個範本，或自己寫。",
    start: "開始",
    connecting: "連線中…",
    working: "處理中…",
    realtimeNote: "⚡ 即時輔助會聆聽現場音訊 —— 請在執行中的 Live 工作階段裡使用。",
    listening: "正在聆聽 —— 提示會隨著你說話出現在這裡。",
    pressBefore: "按 ",
    pressRolling: " 即可從對話中滾動產生提示。",
    pressSummary: " 即可總結轉寫內容。",
    tmplSummary: "摘要",
    tmplActionItems: "行動項",
    tmplLiveHints: "即時提示（教練）",
    tmplDecisions: "決策與負責人",
    tmplTranslate: "翻譯成英文",
    tmplBlank: "空白",
  },

  endpoints: {
    name: "名稱",
    namePlaceholder: "例如：My gateway",
    apiKey: "API 金鑰",
    leaveBlank: "（留空則保持不變）",
    advanced: "進階 —— 輔助參數與轉寫",
    assistHead: "AI 筆記 / 輔助",
    systemPrompt: "系統提示詞",
    systemPromptPlaceholder: "前置到每個輔助任務的常駐指令（人設、語言、風格）。",
    providerDefault: "提供商預設值",
    noLimit: "無限制",
    apiShapeHead: "轉寫 API 形式",
    builtin: "內建",
    customHead: "OpenAI-compatible 端點",
    noKeyYet: "尚無金鑰",
    keySet: "已設金鑰",
    noKey: "無金鑰",
    getKey: "取得金鑰 ↗",
    addKey: "新增金鑰",
    keyPlaceholder: "貼上 API 金鑰",
    show: "顯示",
    hide: "隱藏",
    addEndpoint: "+ 新增 OpenAI Compatible 端點",
    intro: "金鑰只儲存在本機，且只傳送給所屬的提供商。",
  },

  common: {
    transcribeWith: "轉寫模型",
    transcript: "轉寫文字",
    close: "關閉",
    cancel: "取消",
    save: "儲存",
    edit: "編輯",
    remove: "移除",
    add: "新增",
    clear: "清空",
    dismiss: "關閉",
    copy: "複製",
  },
};
