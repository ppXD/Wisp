<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="branding/wisp-wordmark-reverse.svg">
  <img src="branding/wisp-wordmark.svg" alt="Wisp" width="300">
</picture>

### 本地优先、实时的会议转写 —— 全程在设备端、隐私安全、GPU 加速。

把你的麦克风**和**会议的声音，实时变成带说话人标注的字幕，旁边还有一个 AI 助手 —— 全部在你自己的电脑上完成。不上云、不上传、不需要账号。

<br>

[![CI](https://github.com/ppXD/Wisp/actions/workflows/ci.yml/badge.svg)](https://github.com/ppXD/Wisp/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ppXD/Wisp?include_prereleases&sort=semver&label=release&color=c96442)](https://github.com/ppXD/Wisp/releases)
[![Platforms](https://img.shields.io/badge/platform-macOS%20·%20Windows-1a1915)](#-安装)
[![Built with Rust](https://img.shields.io/badge/Rust-stable-c96442?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![Tauri v2](https://img.shields.io/badge/Tauri-v2-1a1915?logo=tauri&logoColor=ffc131)](https://tauri.app)
[![License: MIT](https://img.shields.io/badge/license-MIT-5f8c6a)](#-许可证)

[English](README.md) · **简体中文**

<br>

[![下载 macOS 版](https://img.shields.io/badge/下载-macOS%20·%20Apple%20Silicon-1a1915?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ppXD/Wisp/releases)
&nbsp;
[![下载 Windows 版](https://img.shields.io/badge/下载-Windows%20x64-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/ppXD/Wisp/releases)

<br>

<img src="branding/screenshot-live.png" alt="Wisp —— 实时转写，带说话人标注，右侧是实时 AI 助手" width="860">

</div>

---

**Wisp** 把任何对话实时变成干净、带时间戳、带说话人标注的字幕 —— 旁边还跑着一个实时 AI 助手。默认情况下，每一段音频、每一个模型都留在你的设备上。它几秒装好，**零**额外配置（不用虚拟声卡、不用内核扩展），并且针对每个平台做了底层优化 —— Apple Silicon 上用 Metal + 神经引擎，Windows 上用原生 loopback。

> 🔒 **隐私优先** · ⚡ **贴着架构优化** · 🎛️ **由你自由配置** · 🪶 **装好即用**

## ✨ 亮点

| | |
|---|---|
| 🎙️ **实时 + AI 助手** | 亚秒级流式字幕，带说话人标注；外加一个能实时总结、提取待办、即时辅导的 AI 助手 |
| 🔒 **100% 设备端** | 音频和模型从不离开你的电脑。云端引擎严格按需启用，用你自己的密钥 |
| 🍎 **Apple Silicon 原生** | Metal GPU 推理、经 Core ML 调用 Apple 神经引擎、统一内存零拷贝，以及设备端 Apple SpeechAnalyzer |
| 🔌 **零依赖** | 一键采集系统声音 —— **不用 BlackHole**、不用内核扩展、不用虚拟声卡 |
| 🎛️ **真正可定制** | 自由选择、配置、**删除**模型。语言、精度/速度、解码参数 —— 全在你手上 |
| 📄 **顶尖的文件转写** | 精准的批量转写，带说话人分离、词级时间戳、自定义词表，以及结构化导出 |
| 🧠 **私密可搜索的记忆** | 每场会议自动存入本地笔记库，按**语义**搜索，而不只是关键词 —— 设备端嵌入模型（Qwen3、BGE-M3、多语 E5），关键词 + 语义混合排序 |
| 🪶 **极小体积** | 8–22 MB 的安装包；动辄数 GB 的模型只在你需要时才下载 |

---

## 🎙️ 实时会议副驾

这是 Wisp 的核心 —— 而且很快。

- **亚秒级流式字幕。** 边说边出字（实时草稿 → 定稿），每句带时间戳。不用「停下来才能看结果」。
- **知道谁在说话。** 设备端说话人分离实时给每一句打标签 —— **你** 还是 **对方**，或 **说话人 1 / 2 / 3** —— 用持续更新的说话人质心，长会也稳定。
- **两边都收。** 你的麦克风和会议系统声音被融合到同一条时间线上，配合 **WebRTC AEC3 回声消除**，远端的声音不会从你的麦克风串回去。一键搞定 —— 不用装 loopback 驱动。
- **🤖 AI 助手 —— 会议里的第二大脑。** 一个边想边输出的实时副驾面板：
  - **滚动总结**、**待办事项**、**决议**、**未决问题**，随会议进展实时更新
  - **跟进邮件**草稿，写好即可发送
  - **实时提示**与服务业模板 —— 销售辅导、客服指引、实时情绪/语气监测
  - 带说话人上下文，知道*谁*说了*什么*；节奏可控，帮忙但不吵闹

> 用你自己的 LLM 端点（本地或云端）—— 助手与模型无关，参数完全开放（温度、惩罚项、最大 token 等）。

## 📄 顶尖的文件转写

丢进任意音频或视频文件，得到一份你能信赖的字幕：

- **默认精度优先** —— Whisper **large-v3-turbo**，更重或量化的版本一键切换。
- **说话人分离** —— *谁*说了*什么*，带**词级时间戳**和逐词说话人归属。
- **自定义词表 / 术语偏置** —— 喂进人名、产品名、行话，让它们转写正确。
- **更干净的输入** —— 神经降噪和 VAD 门控在模型看到之前就剔除非语音。
- **可选的本地 LLM 清理** —— 在不离开设备的前提下整理标点和口头语。
- **结构化 Markdown 导出** —— 总结、说话人、带时间戳的时间线，开箱即可分享。
- **实时进度** —— 即便在不透明的解码阶段也有进度，绝不让你盯着一个卡住的进度条。

<p align="center">
  <img src="branding/screenshot-file.png" alt="Wisp —— 文件转写，右侧打开 AI 笔记侧边栏" width="860">
</p>

## 🧠 一个私密、可搜索的记忆库

Wisp 转写的每场会议都会存入**本地笔记库** —— 你可以按*语义*搜索所有笔记，全程在设备端完成。

- **语义 + 关键词，融合检索。** 搜*「预算讨论」*，即使原话从没出现过这几个字，Wisp 也能找到对的那条笔记 —— SQLite FTS5 关键词匹配与向量相似度融合（混合检索）。每条结果都告诉你*为什么*命中：关键词高亮 + 语义置信度。
- **顶尖的设备端嵌入模型 —— 由你挑选。** 在设置里直接下载、切换、删除设备端嵌入模型：
  - **Qwen3-Embedding 0.6B** —— 顶尖的指令微调**解码器**嵌入模型
  - **BGE-M3** —— 1024 维、100+ 语言、强中文
  - 以及 **GTE 多语**、**多语 E5**（small / base / large）、**BGE 中文**
  
  切换模型会**原子地**重建索引 —— 切换中断或失败绝不会损坏已有索引。
- **不上传任何内容。** 笔记通过 ONNX Runtime 在本地嵌入与检索。某个任务想用托管模型？OpenAI 兼容的云端嵌入模型**按需启用**，用你自己的密钥。

## 🔒 本地优先、真正私密

转写、说话人分离、降噪、VAD 全部通过 `sherpa-onnx` 和 Metal/ANE **在设备端**运行 —— 音频从不碰网络。模型**只在你选择安装时**才从 Hugging Face 拉取，然后缓存到本地。

某个任务需要托管模型？云端引擎（OpenAI、Gemini、Groq、Qwen、Speechmatics）以**按需启用**的方式提供 —— 你填入自己的密钥（本地存储），其余一切 Wisp 仍然保持本地。

## 🍎 为 Apple Silicon 调优

Wisp 不只是「能在 Mac 上跑」—— 它贴着架构做了优化：

- **Metal GPU 推理** —— Whisper 引擎（whisper.cpp）通过 Metal 在 GPU 上执行，相比 CPU 大幅提速。
- **Apple 神经引擎** —— Whisper 编码器经 **Core ML** 运行，把最重的一段放到 **ANE** 上，给 GPU 和 CPU 腾出资源。
- **统一内存架构** —— Apple Silicon 的共享内存意味着 CPU、GPU、ANE 之间**零拷贝**交接：没有 PCIe 往返、更低延迟、更省电。
- **Apple SpeechAnalyzer** —— 在 macOS 26 上，Wisp 可把 Apple 内置的设备端语音框架（ANE 加速、**零模型下载**）作为一等引擎使用。
- **ScreenCaptureKit** —— 原生、带权限的系统声音采集，无需内核扩展。

引擎、模型和解码节奏都会**根据你的机器自动选择**（核心数、内存、GPU/神经引擎档位），所以开箱即快，想精调时也能上手。

## 🎛️ 由你配置 —— 而不是反过来

没有强制的引擎下载。没有逼你先下个几 GB 的「总结引擎」才能开始。Wisp 立即可用，**由你**决定加什么：

- **任选模型** —— 从目录里挑，速度优先或精度优先 —— 还能按模式（实时 vs 文件）分别切换。
- **导入自己的语音模型** —— 选择 sherpa-onnx `.onnx` 图或 Whisper GGML/GGUF 文件；Wisp 会先验证格式，并自动带上匹配的 tokens、encoder/decoder/joiner 图和外部权重。
- **跨地区可靠下载** —— 自动在 Hugging Face 与镜像之间切换，保留 partial 文件并断点续传，提供有限重试，以及环境变量、HTTP、SOCKS 代理支持。可在**设置 → 模型下载**中即时更换镜像或代理；转写、说话人、降噪、Core ML 和嵌入模型共用同一套策略。
- **删除模型** —— 不需要的直接在选择器里一键删掉，腾回磁盘空间。
- **一切可配** —— 转写语言、精度/速度档位、VAD 门控、降噪、解码阈值、说话人分离、自定义词表 —— 而不是被锁死在某个预设上。
- **诚实的选择器** —— 你的机器跑不动的模型会被清楚标出，下载*之前*就给出体积和硬件提示。

## 🔌 零配置、无依赖

- **macOS：** 系统声音走 **ScreenCaptureKit** —— *不用 BlackHole、不用 Loopback、不用虚拟声卡。*
- **Windows：** 系统声音走原生 **WASAPI loopback**。
- 麦克风 + 会议声音一起采集、回声消除，一键完成。装好就开始 —— 配置就这么多。

## 🪶 还有更多

- **听写** —— 全局按住说话，语音转文字通过原生文本插入注入到*任意* app。
- **极小安装包** —— 8 MB（Windows）/ 22 MB（macOS）；ML 运行时是底线，模型按需下载。
- **多语言界面** —— English、简体中文、繁體中文。
- **跨平台** —— macOS（Apple Silicon）和 Windows（x64），同一套代码。

---

## 📦 安装

从 **[Releases](https://github.com/ppXD/Wisp/releases)** 获取最新版本：

### macOS（Apple Silicon）
1. 下载 `Wisp_<版本>_aarch64.dmg`，打开，把 **Wisp** 拖进「应用程序」。
2. 当前构建未签名，首次需要清一次隔离标记：
   ```sh
   xattr -cr /Applications/Wisp.app
   ```
3. 启动 Wisp。按提示授予**麦克风**和**屏幕录制**（用于系统声音）权限。

> 仅支持 Apple Silicon（M1 或更新）。不支持 Intel Mac —— GPU/神经引擎和系统声音的路径需要现代 Apple Silicon SDK。

### Windows（x64）
1. 下载 `Wisp_<版本>_x64-setup.exe` 并运行。
2. 启动 Wisp，按提示授予麦克风权限。

---

## 🛠️ 从源码构建

需要 **Rust**（stable）、**Node** 20+，以及各平台的构建工具（macOS 上是 Xcode + `meson`/`ninja`；Windows 上是 MSVC）。

```sh
git clone --recurse-submodules https://github.com/ppXD/Wisp.git
cd Wisp/app
npm install
npm run tauri dev      # 运行应用
npm run tauri build    # 产出安装包
```

## 💻 平台支持

| 平台 | 状态 | 转写 | GPU | 系统声音 | 回声消除 |
|---|---|---|---|---|---|
| **macOS**（Apple Silicon） | ✅ 已发布 | sherpa-onnx · whisper.cpp · Apple SpeechAnalyzer | Metal + ANE | ScreenCaptureKit | WebRTC AEC3 |
| **Windows**（x64） | ✅ 已发布 | sherpa-onnx | DirectML（自动）· CPU 回退 | WASAPI loopback | 跨流去重 |

## 📄 许可证

[MIT](LICENSE) © Wisp 贡献者。

<div align="center"><sub>由 Rust · Tauri · sherpa-onnx · whisper.cpp 构建</sub></div>
