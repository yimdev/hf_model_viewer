# LLM VRAM Calculator — Zero-Download GPU Memory Estimator

[English](./README.md) | [简体中文](./README.zh-CN.md)

> Parse any Hugging Face model topology without downloading weights, then estimate GPU VRAM from verified Architecture Profiles. Initial profiles cover GLM 5.2, DeepSeek V4 Pro, and Hunyuan 3.

### Features

- **Zero-download parsing** — reads only the `safetensors` header JSON via HTTP Range requests, never downloading weight data; parses huge MoE repos in seconds.
- **Dynamic VRAM estimation** — computes per-tensor weight VRAM from on-disk shapes and dtypes, then adds the Effective KV Cache Payload from a verified Architecture Profile.
- **Repository-routed Architecture Profiles** — a canonical Model Repository Identifier selects one dedicated Architecture Profile. The current commit and every algorithm-dependent config input are compared with the audited baseline; drift produces a warning while valid current inputs drive the calculation.
- **Initial dedicated layouts** — GLM 5.2 IndexShare, DeepSeek V4 Pro HCA/CSA (including indexer and compressor state), and Hunyuan 3 full-context GQA. Each Profile owns one complete layout.
- **Auditable breakdown** — exposes repository provenance, Profile Assurance, structured config differences, Architecture Profile/KV Cache Layout versions, and every buffer group's layers, element count, dtype, bytes, and formula.
- **Fine-grained composition** — overview chart breaks VRAM down by Tensor Name Pattern plus the Effective KV Cache Payload, with on-disk dtypes displayed alongside each entry.
- **Bilingual UI** — one-click Chinese / English switching, persisted locally.

### Design principle

The estimator counts the effective payload that model semantics require to remain GPU-resident. It excludes top-k workspaces, framework capacity padding, allocator fragmentation, offload, and optional speculative runtimes. If any required component is unknown, total VRAM remains unknown too.

### Math model

```
Vtotal   = Vweights + Vkv_cache
Vweights = Σ params × B_dtype / 1024³
Vkv      = Σ profile-calculated buffer bytes / 1024³
```

The verified semantic payload is `B × S × 95,232 bytes` for GLM 5.2 and `B × S × 327,680 bytes` for Hunyuan 3. DeepSeek V4 Pro separately accounts for HCA/CSA local and compressed KV, indexer KV, and F32 compressor live state; the UI audit view exposes the complete per-buffer formulas.

The public `computeKV(...)` interface requires `source: { repoId, commitId }`, current config, and workload. The `estimateVRAM(...)` interface requires the same provenance plus tensor metadata. Both accept `sequenceLengths: number[]`; for ragged batches, GLM 5.2 and Hunyuan 3 use `Σ sequenceLengths`, while DeepSeek V4 Pro evaluates window and compression transitions independently for every sequence before summing each buffer.

Profile Assurance is independent of estimate completeness. An audited commit with an exact algorithm-input baseline is `verified`. A valid newer config still produces a Complete VRAM Estimate from its current values, but the result is marked `warning` and reports the current commit, audited commit, and structured config differences.

### Local development

```bash
npm install
npm run dev        # local dev server
npm run build      # build static site -> dist-web/
npm test           # run test suite
```

### Deploy to GitHub Pages

Pushed commits deploy automatically via GitHub Actions (`.github/workflows/deploy.yml`):

```bash
git push -u origin main
```

Then in the repo `Settings → Pages → Source` choose **GitHub Actions**. Site URL:
`https://<user>.github.io/hf_view/`
