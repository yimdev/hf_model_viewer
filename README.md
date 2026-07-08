# LLM Architecture & VRAM Calculator

> 零下载解析 Hugging Face 模型拓扑，并提供动态显存（VRAM）计算。支持 MoE 与 DeepSeek MLA / DSA 架构。Web 与浏览器扩展（Manifest V3）同一份代码双形态分发。
>
> Zero-download parsing of Hugging Face model topology with dynamic VRAM estimation. Supports MoE and DeepSeek MLA / DSA attention. One codebase ships as both a web app and a Manifest V3 browser extension.

[English](#english) | [中文](#中文)

---

## 中文

### 功能特性

- **零下载解析**：通过 HTTP Range 请求只读取 `safetensors` 文件头部的 JSON 元数据，不下载任何权重数据，秒级解析超大 MoE 仓库。
- **动态 VRAM 计算**：基于解析出的张量形状与 dtype，逐张量计算权重显存，并叠加 KV Cache 与固有开销。
- **通用 KV Cache 推导**：优先从张量形状（厂商无关）推导 KV 缓存，不依赖各家命名不一致的 `config` 超参；仅在融合 QKV 等无法纯形状拆分时回退到 `config`。
- **多架构支持**：MHA / GQA / MQA / MLA（DeepSeek 系）/ DSA（DeepSeek V3.2 稀疏注意力）。
- **按模块量化策略**：`uniform`（全量化）/` keep-fp16`（仅 Linear）/ `native`（按磁盘实际 dtype）。尊重已预量化的权重——本就是 FP4 就按 FP4 计。
- **细粒度组成明细**：总览按张量类别（嵌入 / 注意力 / MLP / 归一化 / LM Head / MoE 专家）拆解显存，并叠加 KV 与开销。
- **双形态分发**：同一份源码同时构建为 GitHub Pages 静态站点与浏览器扩展。
- **中英双语界面**：Web 与扩展均提供中文 / English 一键切换，选择记忆在本地。

### 架构原则

显存计算器只统计**落入显存**的张量（如权重、KV Cache），不把仅参与**计算**的量（如 DSA 的 top-k 选择）计入 VRAM。

### 数学模型

```
Vtotal   = Vweights + Vkv_cache + Voverhead
Vweights = Σ params × B_precision / 1024³
Vkv      = (MHA/GQA/MQA) 4 · B · S · L · Hkv · Dhead / 1024³
         = (MLA)         2 · B · S · L · (kv_lora_rank + qk_rope_head_dim) / 1024³
         = (DSA)         B · S · L · [ 2·(kv_lora_rank + qk_rope_head_dim) + index_head_dim ] / 1024³
Voverhead = 2.0 + Vweights × 10%
```

其中 `B` = batch size，`S` = context length，`L` = 层数（由张量名层号推出），`Hkv` / `Dhead` 由 K/V 投影权重输出维度得出。

### 本地开发

```bash
npm install
npm run dev            # 本地开发服务器（web 形态）
BUILD_TARGET=web npm run build:web   # 构建 GitHub Pages 站点 → dist-web/
BUILD_TARGET=ext npm run build:ext   # 构建浏览器扩展包 → dist-ext/
```

### 部署到 GitHub Pages

推送后由 GitHub Actions 自动部署（` .github/workflows/deploy.yml`）：

```bash
git push -u origin main
```

随后在仓库 `Settings → Pages → Source` 选择 **GitHub Actions**。站点地址：
`https://<user>.github.io/hf_model_viewer/`

### 构建为浏览器扩展

```bash
BUILD_TARGET=ext npm run build:ext
```

在浏览器扩展管理页加载解压的 `dist-ext/` 目录（需开启「开发者模式」）。扩展通过 Background Service Worker 代理网络请求，突破网页端 CORS 限制。

---

## English

### Features

- **Zero-download parsing** — reads only the `safetensors` header JSON via HTTP Range requests, never downloading weight data; parses huge MoE repos in seconds.
- **Dynamic VRAM estimation** — computes per-tensor weight VRAM from parsed shapes and dtypes, then adds KV Cache and fixed overhead.
- **Generic KV Cache derivation** — derives KV cache primarily from tensor shapes (vendor-neutral), not from each vendor's differently-named `config` hyper-params; falls back to `config` only when shapes can't be split (e.g. fused QKV).
- **Multi-architecture** — MHA / GQA / MQA / MLA (DeepSeek family) / DSA (DeepSeek V3.2 sparse attention).
- **Per-module quantization strategy** — `uniform` (quantize all) / `keep-fp16` (Linear only) / `native` (use on-disk dtype). Respects pre-quantized weights — already FP4 stays FP4.
- **Fine-grained composition** — overview breaks VRAM down by tensor category (embedding / attention / MLP / norm / LM Head / MoE experts), plus KV and overhead.
- **Dual-form delivery** — one source tree builds both a GitHub Pages static site and a browser extension.
- **Bilingual UI** — web and extension both offer one-click Chinese / English switching, persisted locally.

### Design principle

The estimator counts only tensors that **land in VRAM** (weights, KV Cache), never quantities involved purely in **compute** (e.g. DSA's top-k selection).

### Math model

```
Vtotal   = Vweights + Vkv_cache + Voverhead
Vweights = Σ params × B_precision / 1024³
Vkv      = (MHA/GQA/MQA) 4 · B · S · L · Hkv · Dhead / 1024³
         = (MLA)         2 · B · S · L · (kv_lora_rank + qk_rope_head_dim) / 1024³
         = (DSA)         B · S · L · [ 2·(kv_lora_rank + qk_rope_head_dim) + index_head_dim ] / 1024³
Voverhead = 2.0 + Vweights × 10%
```

where `B` = batch size, `S` = context length, `L` = layer count (from tensor name indices), and `Hkv` / `Dhead` come from K/V projection output dims.

### Local development

```bash
npm install
npm run dev                           # local dev server (web form)
BUILD_TARGET=web npm run build:web    # build GitHub Pages site -> dist-web/
BUILD_TARGET=ext npm run build:ext    # build browser extension bundle -> dist-ext/
```

### Deploy to GitHub Pages

Pushed commits deploy automatically via GitHub Actions (`.github/workflows/deploy.yml`):

```bash
git push -u origin main
```

Then in the repo `Settings → Pages → Source` choose **GitHub Actions**. Site URL:
`https://<user>.github.io/hf_model_viewer/`

### Build as a browser extension

```bash
BUILD_TARGET=ext npm run build:ext
```

Load the unpacked `dist-ext/` directory in your browser's extension manager (enable "Developer mode"). The extension proxies network requests through a Background Service Worker to bypass page-side CORS.
