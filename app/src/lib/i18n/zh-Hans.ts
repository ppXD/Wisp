// 简体中文 (Simplified Chinese). Typed `Messages`, so it must mirror en.ts exactly — a missing or
// mistyped key is a compile error. To add a language, copy this file and translate the values.

import type { Messages } from "./en";

export const zhHans: Messages = {
  nav: {
    collapse: "收起",
    expand: "展开",
    collapseSidebar: "收起侧栏",
    expandSidebar: "展开侧栏",
    live: "实时",
    file: "文件",
    settings: "设置",
    themeToLight: "切换到浅色主题",
    themeToDark: "切换到深色主题",
    themeToggle: "切换主题",
    lightMode: "浅色模式",
    darkMode: "深色模式",
    language: "语言",
    languageMenu: "选择语言",
  },

  live: {
    loadingModels: "正在加载模型…",

    you: "我",
    them: "对方",
    youTip: (on, running) =>
      on
        ? running
          ? "我（麦克风）已开启 —— 点击静音"
          : "我（麦克风）已开启 —— 点击以排除转写"
        : running
          ? "我（麦克风）已静音 —— 点击取消静音"
          : "我（麦克风）已关闭 —— 点击以纳入转写",
    themTip: (on, running) =>
      on
        ? running
          ? "对方（系统音频）已开启 —— 点击静音"
          : "对方（系统音频）已开启 —— 点击以排除转写"
        : running
          ? "对方（系统音频）已静音 —— 点击取消静音"
          : "对方（系统音频）已关闭 —— 点击以纳入转写",

    status: {
      ready: "就绪",
      keyNeeded: "需要密钥",
      noModel: "无模型",
      recording: "录音中",
    },

    empty: {
      before: "选个模型，按 ",
      action: "开始",
      after: "，然后开口说话。",
    },

    start: "开始转写",
    stop: "停止转写",
    startConnecting: "连接中…",
    startDownloading: "正在下载模型…",
    startSlowHint: "正在加载模型 —— 首次运行可能需要几秒",

    advanced: "高级 · 音频、语言、说话人",
    advancedCloud: "音频 · 设备",
  },

  file: {
    dropTitle: "点击选择文件，或拖入此处",
    options: "选项 · 精度、提示、说话人",
    optionsCloud: "选项 · 提示、说话人",
    subCloudReady: { before: "mp3、m4a、wav、flac、mp4、mov… 发送至 ", after: "。" },
    subCloudPick: "请在上方选择一个云端模型。",
    subCloudNoKey: (provider) => `添加你的 ${provider} API 密钥以在云端转写。`,
    subLocalReady: { before: "mp3、m4a、wav、flac、mp4、mov… 由 ", after: " 在本地转写。" },
    subLocalMissing: { before: "", after: " 尚未下载 —— 在下方获取后即可转写。" },
  },

  common: {
    transcribeWith: "转写模型",
    transcript: "转写文本",
    close: "关闭",
  },
};
