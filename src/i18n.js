/* i18n.js — Internationalization (zh / en)
 * ------------------------------------------------------------
 * Shared by both the web and the extension builds. No DOM
 * dependency except optional localStorage (guarded) and navigator.
 *
 * Usage:
 *   import { t, getLang, setLang, onLangChange } from '../i18n.js';
 *   t('brand.title');                 // current language
 *   t('status.shard', { done, total, file });
 *   setLang('en');                    // persists + notifies listeners
 * ------------------------------------------------------------ */

const STR = {
  // Brand
  'brand.title': {
    zh: 'LLM 架构与显存计算器',
    en: 'LLM Architecture & VRAM Calculator',
  },
  'brand.sub': {
    zh: '零下载解析 · 动态 VRAM 计算 · 双形态分发',
    en: 'Zero-download parsing · Dynamic VRAM estimation · Dual-form delivery',
  },

  // Controls
  'ctl.repoPlaceholder': {
    zh: 'org/repo，如 Qwen/Qwen2.5-7B-Instruct',
    en: 'org/repo, e.g. Qwen/Qwen2.5-7B-Instruct',
  },
  'ctl.analyze': { zh: '分析', en: 'Analyze' },
  'ctl.advanced': { zh: '高级（受限模型可选 Token）', en: 'Advanced (token for gated models)' },
  'ctl.tokenPlaceholder': { zh: 'hf_xxx（私有/受限仓库时填写）', en: 'hf_xxx (for private / gated repos)' },
  'ctl.quantPrecision': { zh: '量化精度', en: 'Quantization precision' },
  'ctl.quantStrategy': { zh: '量化策略', en: 'Quantization strategy' },
  'ctl.stratUniform': {
    zh: '均匀量化（全部张量按精度滑杆）',
    en: 'Uniform (all tensors to selected precision)',
  },
  'ctl.stratKeepFp16': {
    zh: '仅 Linear 量化（Embedding/Norm/LM Head 保留 FP16）',
    en: 'Linear only (Embedding/Norm/LM Head stay FP16)',
  },
  'ctl.stratNative': {
    zh: '按磁盘实际精度（忽略滑杆，含预量化）',
    en: 'Native on-disk dtype (ignore slider, incl. pre-quantized)',
  },
  'ctl.quantHint': {
    zh: '权重显存以磁盘 dtype 为真值：已预量化模型（如 FP4/INT4）直接按实际占用计。',
    en: 'Weight VRAM uses on-disk dtype as ground truth: pre-quantized models (e.g. FP4/INT4) count at actual size.',
  },
  'ctl.batchSize': { zh: 'Batch Size：', en: 'Batch Size: ' },
  'ctl.contextLength': { zh: '上下文长度', en: 'Context Length' },

  // Overview
  'ov.title': { zh: '总览', en: 'Overview' },
  'ov.treeTitle': { zh: '层级结构树', en: 'Layer structure tree' },
  'ov.treeEmpty': { zh: '输入仓库 ID 并点击 Analyze 开始解析', en: 'Enter a repo ID and click Analyze to start parsing' },
  'ov.compTitle': { zh: '组成明细', en: 'Composition breakdown' },
  'ctl.empty': { zh: '尚未分析', en: 'Not analyzed yet' },

  // Result summary card
  'sum.total': { zh: '总显存需求：', en: 'Total VRAM: ' },
  'sum.weights': { zh: '权重', en: 'Weights' },
  'sum.kv': { zh: 'KV', en: 'KV' },
  'sum.overhead': { zh: '开销', en: 'Overhead' },
  'sum.weightStrategy': { zh: '权重策略：', en: 'Weight strategy: ' },
  'sum.attnArch': { zh: '注意力架构：', en: 'Attention arch: ' },
  'sum.kvFormula': { zh: 'KV 公式：', en: 'KV formula: ' },

  // Composition group titles
  'group.weight': { zh: '权重显存（稠密基础）', en: 'Weight VRAM (dense base)' },
  'group.moe': { zh: 'MoE 专家层', en: 'MoE expert layers' },
  'group.kv': { zh: 'KV Cache', en: 'KV Cache' },
  'group.overhead': { zh: '固有开销', en: 'Fixed overhead' },

  // Stats grid
  'stat.totalParams': { zh: '总参数', en: 'Total params' },
  'stat.layers': { zh: '层数', en: 'Layers' },
  'stat.moe': { zh: 'MoE', en: 'MoE' },
  'stat.moeYes': { zh: '是 × {n} 专家', en: 'Yes × {n} experts' },
  'stat.no': { zh: '否', en: 'No' },
  'stat.arch': { zh: '架构', en: 'Arch' },
  'stat.shards': { zh: '分片数', en: 'Shards' },

  // Status line
  'status.enterRepo': { zh: '请输入仓库 ID（org/repo）', en: 'Please enter a repo ID (org/repo)' },
  'status.fetching': { zh: '解析中：拉取 config.json …', en: 'Parsing: fetching config.json …' },
  'status.shard': { zh: '解析分片头部 {done}/{total}：{file}', en: 'Parsing shard header {done}/{total}: {file}' },
  'status.done': { zh: '解析完成：{shards} 个分片，{tensors} 个张量', en: 'Done: {shards} shards, {tensors} tensors' },

  // Structure tree
  'tree.col.op': { zh: '算子', en: 'Op' },
  'tree.col.shape': { zh: 'Shape', en: 'Shape' },
  'tree.col.dtype': { zh: 'DType', en: 'DType' },
  'tree.col.params': { zh: '参数量', en: 'Params' },
  'tree.col.vram': { zh: '显存', en: 'VRAM' },
  'tree.expertNote': {
    zh: '每专家同构，以下展示单一代表专家（参数量 ×{n} = 合计）。',
    en: 'All experts share the same shape; one representative expert shown (params ×{n} = total).',
  },
  'tree.paramsUnit': { zh: '参数', en: 'params' },
  'tree.mlpDense': { zh: 'MLP（稠密）', en: 'MLP (dense)' },

  // Chart
  'chart.vramLabel': { zh: '显存占用 (GB)', en: 'VRAM usage (GB)' },
  'chart.gb': { zh: 'GB', en: 'GB' },

  // Composition category labels
  'cat.embedding': { zh: '嵌入层 Embedding', en: 'Embedding' },
  'cat.attn': { zh: '注意力层（稠密）', en: 'Attention (dense)' },
  'cat.mlp': { zh: 'MLP / FFN（稠密）', en: 'MLP / FFN (dense)' },
  'cat.norm': { zh: '归一化层（稠密）', en: 'Norm (dense)' },
  'cat.other': { zh: '其他（稠密）', en: 'Other (dense)' },
  'cat.lmhead': { zh: 'LM Head', en: 'LM Head' },
  'cat.expert': { zh: 'MoE 专家层', en: 'MoE expert layers' },
  'cat.weight': { zh: '权重（未细分）', en: 'Weights (unsplit)' },
  'cat.kv': { zh: 'KV Cache', en: 'KV Cache' },
  'cat.overhead': { zh: '固有开销', en: 'Fixed overhead' },

  // KV formula labels (technical; values interpolated)
  'kv.mha.tensor': { zh: 'MHA/GQA 架构（张量推导 K/V 投影维度）', en: 'MHA/GQA arch (tensor-derived K/V proj dims)' },
  'kv.mha.tensor.detail': {
    zh: '{arch} 架构（张量推导 K/V 维度，Hkv≈{hkv}, Dhead≈{dhead}）',
    en: '{arch} arch (tensor-derived K/V dims, Hkv≈{hkv}, Dhead≈{dhead})',
  },
  'kv.mha.config': { zh: '{arch} 架构 (Hkv={hkv}, Dhead={dhead})', en: '{arch} arch (Hkv={hkv}, Dhead={dhead})' },
  'kv.mla.tensor': { zh: 'MLA 压缩架构（张量推导 latent）', en: 'MLA compressed arch (tensor-derived latent)' },
  'kv.mla.config': {
    zh: 'MLA 压缩架构 (kv_lora_rank + qk_rope_head_dim)',
    en: 'MLA compressed arch (kv_lora_rank + qk_rope_head_dim)',
  },
  'kv.dsa.tensor': {
    zh: 'DSA 稀疏注意力（张量推导 latent + FP8 索引器）',
    en: 'DSA sparse attention (tensor-derived latent + FP8 indexer)',
  },
  'kv.dsa.config': {
    zh: 'DSA 稀疏注意力 (MLA latent + FP8 索引器K)',
    en: 'DSA sparse attention (MLA latent + FP8 indexer K)',
  },
  'kv.dsa.note': {
    zh: 'DSA 主要削减计算量 O(L²)→O(L·topk) 与加载带宽；KV 容量≈稠密 MLA+索引器，几乎不降。',
    en: 'DSA mainly cuts compute O(L²)→O(L·topk) and load bandwidth; KV capacity ≈ dense MLA + indexer, barely reduced.',
  },
  'kv.v4.tensor': {
    zh: 'DeepSeek-V4 NSA（张量推导 wkv 潜变量 + 逐层 compress_ratio）',
    en: 'DeepSeek-V4 NSA (tensor-derived wkv latent + per-layer compress_ratio)',
  },
  'kv.v4.config': {
    zh: 'DeepSeek-V4 NSA (head_dim / compress_ratios / window_size)',
    en: 'DeepSeek-V4 NSA (head_dim / compress_ratios / window_size)',
  },
  'kv.v4.note': {
    zh: 'DeepSeek-V4 原生稀疏注意力：每层 KV 缓存 = (滑动窗口 {window} + S/压缩比) × {hd} 潜向量（MLA 风格单潜变量，非逐头 K/V）；压缩比=4 的层额外存 (S/4) × {ihd} 的 indexer 选择缓存。默认按 BF16 估算（启用 FP8 KV 缓存约减半）。',
    en: 'DeepSeek-V4 Native Sparse Attention: per-layer KV cache = (sliding window {window} + S/compress_ratio) × {hd} latent (MLA-style single latent, not per-head K/V); layers with compress_ratio=4 additionally store (S/4) × {ihd} indexer selection cache. Estimated at BF16 by default (enabling FP8 KV cache roughly halves it).',
  },

  // Weight strategy notes
  'weight.native': {
    zh: '权重按磁盘实际 dtype 逐张量计算（已含任何预量化，滑杆精度不生效）',
    en: 'Weights counted per-tensor at on-disk dtype (incl. any pre-quantization; slider precision ignored)',
  },
  'weight.keepFp16': {
    zh: '仅 Linear 量化到目标精度；Embedding / Norm / LM Head 保留 FP16',
    en: 'Only Linear quantized to target precision; Embedding / Norm / LM Head stay FP16',
  },
  'weight.uniform': {
    zh: '全模型均匀量化到目标精度；已预量化层按磁盘实际精度计（不重复压缩）',
    en: 'Whole model uniformly quantized to target precision; pre-quantized layers counted at on-disk dtype (no double compression)',
  },

  // Errors (user-facing)
  'err.noSafetensors': {
    zh: '该模型未提供 Safetensors 格式，无法进行远程 Range 碎片化解析，请更换现代大模型仓库。',
    en: 'This model does not provide the Safetensors format; remote Range-based header parsing is unavailable. Please use a modern model repo.',
  },
  'err.badRepoId': { zh: '请输入合法的仓库 ID，形如 org/repo', en: 'Please enter a valid repo ID, e.g. org/repo' },
  'err.configFetch': {
    zh: '无法获取 config.json（仓库是否存在或网络异常）：',
    en: 'Failed to fetch config.json (repo may not exist or network error): ',
  },
  'err.badSafetensors': { zh: '文件 {file} 过短，不是合法的 safetensors', en: 'File {file} is too short to be a valid safetensors' },
  'err.missingLayers': { zh: '缺少 num_hidden_layers', en: 'Missing num_hidden_layers' },
  'err.missingAttn': { zh: '缺注意力头配置', en: 'Missing attention head config' },
  'err.missingKvLora': { zh: '缺 kv_lora_rank', en: 'Missing kv_lora_rank' },
  'err.missingHeadDim': { zh: '缺 head_dim，无法计算 KV 缓存', en: 'Missing head_dim; cannot compute KV cache' },

  // Language toggle
  'lang.label': { zh: '语言', en: 'Language' },
};

const LS_KEY = 'hf-view-lang';
const listeners = new Set();

function detectLang() {
  try {
    if (typeof localStorage !== 'undefined') {
      const saved = localStorage.getItem(LS_KEY);
      if (saved === 'zh' || saved === 'en') return saved;
    }
  } catch {
    /* ignore */
  }
  try {
    if (typeof navigator !== 'undefined' && navigator.language) {
      return navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en';
    }
  } catch {
    /* ignore */
  }
  return 'en';
}

let current = detectLang();

export function getLang() {
  return current;
}

export function setLang(lang) {
  if (lang !== 'zh' && lang !== 'en') return;
  if (lang === current) return;
  current = lang;
  try {
    localStorage.setItem(LS_KEY, lang);
  } catch {
    /* ignore */
  }
  listeners.forEach((cb) => cb(lang));
}

export function onLangChange(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function t(key, vars) {
  const entry = STR[key];
  let s = entry ? entry[current] ?? entry.en : key;
  if (vars && typeof vars === 'object') {
    for (const [k, v] of Object.entries(vars)) {
      s = s.replace(new RegExp('\\{' + k + '\\}', 'g'), v);
    }
  }
  return s;
}
