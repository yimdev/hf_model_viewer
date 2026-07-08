/* vram/estimate.js — Dynamic VRAM estimator
 * ------------------------------------------------------------
 * Math model (Spec §4):
 *   Vtotal = Vweights + Vkv_cache + Voverhead
 *   Vweights = Σparams × B_precision / 1024^3
 *   Vkv_cache = 4·B·S·L·Hkv·Dhead / 1024^3   (MHA/GQA/MQA)
 *   Vkv_cache = 2·B·S·L·(kv_lora_rank + qk_rope_head_dim) / 1024^3   (MLA)
 *   Vkv_cache = B·S·L·[ 2·(kv_lora_rank + qk_rope_head_dim) + index_head_dim ] / 1024^3   (DSA)
 *   Voverhead = 2.0 + Vweights × 10%
 *
 * [Generic approach] KV Cache is derived primarily from the parsed
 *   safetensors tensor SHAPES (see kv/ dir: detect.js detects the arch and
 *   mha/mla/dsa compute it). Each vendor names attention hyper-params
 *   differently (num_hidden_layers vs num_layers; num_key_value_heads vs
 *   multi_query_group_num; head_dim vs kv_channels …), but "KV elements
 *   stored per token per layer" equals the sum of K/V projection output dims
 *   — a vendor-neutral ground truth, and the layer count L comes from tensor
 *   name indices. Only DeepSeek DSA's extra "indexer cache" needs one stable
 *   field. When shape-based derivation fails (e.g. fused QKV in ChatGLM /
 *   GPT-NeoX) we fall back to config hyper-params.
 *
 * Design principle: the estimator counts only tensors that LAND in VRAM,
 *   not quantities involved in compute (e.g. DSA top-k is excluded).
 * ------------------------------------------------------------ */

const GB = 1024 ** 3;

const BPP = { fp16: 2, bf16: 2, int8: 1, int4: 0.5 };

import { computeKV } from './kv/index.js';
import { tensorParams } from '../tree/buildTree.js';
import { t } from '../i18n.js';

export function bytesPerParam(precision) {
  return BPP[precision] ?? 2;
}

/* ------------------------------------------------------------
 * On-disk dtype -> bytes per param (the ground truth for weight VRAM).
 * Every tensor in safetensors metadata carries its own dtype; if a model
 * is published already pre-quantized (INT4/FP4/FP8 …), its real footprint is
 * dictated by that dtype and must NOT be re-compressed or inflated by the
 * precision slider.
 * ------------------------------------------------------------ */
const DTYPE_BYTES = {
  F64: 8, F32: 4, F16: 2, BF16: 2,
  F8_E4M3FN: 1, F8_E5M2: 1, F8_E4M3FNUZ: 1, F8_E5M2FNUZ: 1,
  INT8: 1, UINT8: 1, INT4: 0.5, UINT4: 0.5,
  F4: 0.5, NF4: 0.5, F4E2M3FNUZ: 0.5,
  I16: 2, I32: 4, I64: 8, BOOL: 0.125,
};

/** Parse a safetensors dtype string to bytes/param; unknown dtype falls back to fp16 (2B). */
export function bytesForDtype(dtype, fallback = 2) {
  if (typeof dtype === 'string' && DTYPE_BYTES[dtype] != null) return DTYPE_BYTES[dtype];
  if (typeof dtype === 'string') {
    if (/(INT|UINT|F4|NF4)/.test(dtype) && /4/.test(dtype)) return 0.5;
    if (/8/.test(dtype)) return 1;
    if (/16/.test(dtype)) return 2;
    if (/32/.test(dtype)) return 4;
    if (/64/.test(dtype)) return 8;
  }
  return fallback;
}

/** Protected tensor: not quantized when strategy is "keep-fp16". */
function isProtectedTensor(name) {
  return /embed_tokens|lm_head|norm/i.test(name);
}

/* ------------------------------------------------------------
 * Tensor categorization (for the fine-grained composition breakdown).
 * Consistent with buildTree's grouping but finer — the dense base layers are
 * split into embedding / attention / MLP / norm / other / LM Head so the
 * overview can expand them.
 * ------------------------------------------------------------ */
const EXPERT_TOKEN_RE = /experts\.\d+/i;
function categorizeTensor(name) {
  if (/embed_tokens/i.test(name)) return 'embedding';
  if (/lm_head/i.test(name)) return 'lmhead';
  if (EXPERT_TOKEN_RE.test(name)) return 'expert';
  const lm = name.match(/\.(\d+)\./);
  if (lm) {
    const remainder = name.slice(lm.index + lm[0].length);
    // Order matters: norm before attention, else post_attention_layernorm
    // would be misclassified as attention.
    if (/norm/i.test(remainder)) return 'norm';
    if (/(?:^|[._])(?:mlp|ffn)/i.test(remainder)) return 'mlp';
    if (/(?:^|[._])(?:self_attn|attention)/i.test(remainder)) return 'attn';
    return 'other';
  }
  if (/norm/i.test(name)) return 'norm';
  return 'other';
}

/**
 * Effective bytes/param for a single tensor — combining on-disk dtype and
 * quantization strategy:
 *   - native:       use on-disk dtype directly (slider ignored; real file size)
 *   - keep-fp16:    protected tensors (Embedding/Norm/LMHead) stay ≥FP16;
 *                   others take min(native, target)
 *   - uniform:      all tensors take min(native, target) — simulates quant but
 *                   respects already pre-quantized weights
 * Core: min(native, target) guarantees "already FP4 stays FP4" — never inflated
 * or doubly compressed by the slider.
 */
function effBppFor(t, { targetPrecision, strategy }) {
  const nativeB = bytesForDtype(t.dtype, 2);
  const targetB = BPP[targetPrecision] ?? 2;
  if (strategy === 'native') return nativeB;
  if (strategy === 'keep-fp16' && isProtectedTensor(t.name)) {
    return nativeB > 2 ? 2 : nativeB; // keep FP16 or better
  }
  return Math.min(nativeB, targetB);
}

/**
 * Compute weight bytes per tensor (also produces effBppMap for tree
 * reconciliation and byCategory for the fine-grained breakdown).
 * @returns {{totalBytes:number, baseBytes:number, expertBytes:number, effBppMap:Map, byCategory:Map}|null}
 */
export function computeWeightBytes(tensors, opts = {}) {
  if (!Array.isArray(tensors) || !tensors.length) return null;
  const map = new Map();
  const byCategory = new Map();
  let totalBytes = 0, baseBytes = 0, expertBytes = 0;
  for (const t of tensors) {
    const params = t.params != null ? t.params : tensorParams(t.shape);
    const eff = effBppFor(t, opts);
    map.set(t.name, eff);
    const b = params * eff;
    totalBytes += b;
    const cat = categorizeTensor(t.name);
    byCategory.set(cat, (byCategory.get(cat) || 0) + b);
    if (cat === 'expert') expertBytes += b;
    else baseBytes += b;
  }
  return { totalBytes, baseBytes, expertBytes, effBppMap: map, byCategory };
}

export function buildEffBppMap(tensors, opts) {
  return computeWeightBytes(tensors, opts)?.effBppMap ?? new Map();
}

/**
 * @param {object} config    parsed model config.json
 * @param {object} tree      result of buildTree
 * @param {object} opts      { precision, batch, seq, tensors? }
 *        tensors: flat parsed tensor list (from analyze), used for generic KV derivation
 */
export function estimateVRAM(
  config,
  tree,
  { precision = 'fp16', batch = 1, seq = 8192, tensors = null, strategy = 'uniform' } = {},
) {
  const bpp = bytesPerParam(precision);
  const totalParams = tree.totalParams;

  // Weight VRAM: prefer per-tensor on-disk dtype + quantization strategy.
  // (On-disk precision is the ground truth; the slider only simulates quant on
  // full-precision tensors, pre-quantized layers use the on-disk value.)
  const w = tensors && tensors.length ? computeWeightBytes(tensors, { targetPrecision: precision, strategy }) : null;
  let vWeights, baseWeightsGB, moeWeightsGB, weightNote;
  if (w) {
    vWeights = w.totalBytes / GB;
    baseWeightsGB = w.baseBytes / GB;
    moeWeightsGB = w.expertBytes / GB;
  } else {
    // Fallback: no tensor detail -> uniform full-model × slider precision.
    vWeights = (totalParams * bpp) / GB;
    baseWeightsGB = (tree.baseParams * bpp) / GB;
    moeWeightsGB = (tree.expertParams * bpp) / GB;
  }
  if (strategy === 'native') {
    weightNote = t('weight.native');
  } else if (strategy === 'keep-fp16') {
    weightNote = t('weight.keepFp16');
  } else {
    weightNote = t('weight.uniform');
  }

  let attnArch = 'mha';
  let vKV = null;
  let kvFormulaLabel = '';
  let kvNote = '';
  let kvUnknown = false;
  let kvMethod = 'config';

  // KV Cache derivation delegated to kv/ module: tensor path first, config fallback.
  const kv = computeKV({ config, tensors, precision, batch, seq });
  vKV = kv.vKV;
  attnArch = kv.attnArch ?? 'mha';
  kvFormulaLabel = kv.formulaLabel || '';
  kvNote = kv.note || '';
  kvUnknown = !!kv.kvUnknown;
  kvMethod = kv.kvMethod || 'config';

  const vOverhead = 2.0 + vWeights * 0.1;
  const vKVsafe = vKV ?? 0;
  const vTotal = vWeights + vKVsafe + vOverhead;

  // Fine-grained composition (for overview breakdown): split weights by
  // tensor category, then add KV / overhead.
  const CAT_META = {
    embedding: { labelKey: 'cat.embedding', group: 'weight' },
    attn: { labelKey: 'cat.attn', group: 'weight' },
    mlp: { labelKey: 'cat.mlp', group: 'weight' },
    norm: { labelKey: 'cat.norm', group: 'weight' },
    other: { labelKey: 'cat.other', group: 'weight' },
    lmhead: { labelKey: 'cat.lmhead', group: 'weight' },
    expert: { labelKey: 'cat.expert', group: 'moe' },
  };
  const composition = [];
  if (w) {
    for (const [key, meta] of Object.entries(CAT_META)) {
      // Only list a category that actually exists in this model (byCategory hit);
      // a dense model has no experts -> "MoE expert layers" is hidden. A present
      // but tiny category is still shown truthfully.
      if (!w.byCategory.has(key)) continue;
      const gb = (w.byCategory.get(key) || 0) / GB;
      composition.push({ key, labelKey: meta.labelKey, group: meta.group, gb });
    }
  } else {
    composition.push({ key: 'weight', labelKey: 'cat.weight', group: 'weight', gb: vWeights });
  }
  composition.push({ key: 'kv', labelKey: 'cat.kv', group: 'kv', gb: vKVsafe });
  composition.push({ key: 'overhead', labelKey: 'cat.overhead', group: 'overhead', gb: vOverhead });

  // Chart decomposition (already uses per-tensor effective bytes).
  return {
    precision,
    bpp,
    totalParams,
    vWeights,
    weightNote,
    vKV,
    kvUnknown,
    kvFormulaLabel,
    kvNote,
    kvMethod,
    attnArch,
    isMLA: attnArch === 'mla',
    vOverhead,
    vTotal,
    composition,
    breakdown: {
      baseWeightsGB,
      moeWeightsGB,
      kvGB: vKV,
      overheadGB: vOverhead,
    },
  };
}
