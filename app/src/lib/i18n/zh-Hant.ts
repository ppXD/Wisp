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

  common: {
    transcribeWith: "轉寫模型",
    transcript: "轉寫文字",
    close: "關閉",
  },
};
