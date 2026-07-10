/* vram/estimate.js — Dynamic VRAM estimator
 * ------------------------------------------------------------
 * Math model (Spec §4):
 *   Vtotal = Vweights + Vkv_cache
 *   Vweights = Σparams × B_dtype / 1024^3
 *   Vkv_cache = Σ verified Architecture Profile buffer bytes / 1024^3
 *
 * KV Cache is fail-closed. A curated model-class catalog selects Architecture
 * Profile candidates; every candidate validates its full config+safetensors
 * signature, and exactly one dedicated layout must match before returning an
 * auditable buffer list.
 * Unknown or mismatched profiles keep KV and total VRAM unknown.
 * ------------------------------------------------------------ */

const GB = 1024 ** 3;

import { computeKV } from './kv/index.js';
import { tensorParams } from '../tree/buildTree.js';
import { isNumericPathSegment } from '../tree/pathSegments.js';

/* ------------------------------------------------------------
 * On-disk dtype -> bytes per param (the ground truth for weight VRAM).
 * Every tensor in safetensors metadata carries its own dtype; its real
 * footprint is dictated by that dtype alone.
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

/* ------------------------------------------------------------
 * Tensor categorization remains only for the dense / routed-expert totals.
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
    if (/(?:^|[._])(?:self_attn|attention|attn)/i.test(remainder)) return 'attn';
    return 'other';
  }
  if (/norm/i.test(name)) return 'norm';
  return 'other';
}

function tensorNamePattern(name) {
  return name
    .split('.')
    .map((segment) => (isNumericPathSegment(segment) ? '*' : segment))
    .join('.');
}

/**
 * Compute weight bytes per tensor (also produces effBppMap for tree
 * reconciliation plus byte and DType indexes by Tensor Name Pattern for the
 * overview chart). Every tensor uses its on-disk dtype bytes.
 * @returns {{totalBytes:number, baseBytes:number, expertBytes:number, effBppMap:Map, byTensorNamePattern:Map, dtypesByTensorNamePattern:Map}|null}
 */
export function computeWeightBytes(tensors) {
  if (!Array.isArray(tensors) || !tensors.length) return null;
  const map = new Map();
  const byTensorNamePattern = new Map();
  const dtypesByTensorNamePattern = new Map();
  let totalBytes = 0, baseBytes = 0, expertBytes = 0;
  for (const t of tensors) {
    const params = t.params != null ? t.params : tensorParams(t.shape);
    const eff = bytesForDtype(t.dtype, 2);
    map.set(t.name, eff);
    const b = params * eff;
    totalBytes += b;
    const pattern = tensorNamePattern(t.name);
    byTensorNamePattern.set(pattern, (byTensorNamePattern.get(pattern) || 0) + b);
    if (!dtypesByTensorNamePattern.has(pattern)) dtypesByTensorNamePattern.set(pattern, new Set());
    dtypesByTensorNamePattern.get(pattern).add(t.dtype);
    const cat = categorizeTensor(t.name);
    if (cat === 'expert') expertBytes += b;
    else baseBytes += b;
  }
  return {
    totalBytes,
    baseBytes,
    expertBytes,
    effBppMap: map,
    byTensorNamePattern,
    dtypesByTensorNamePattern,
  };
}

export function buildEffBppMap(tensors) {
  return computeWeightBytes(tensors)?.effBppMap ?? new Map();
}

/**
 * @param {object} config    parsed model config.json
 * @param {object} tree      result of buildTree
 * @param {object} opts      { batch, seq, sequenceLengths?, tensors? }
 *        tensors: flat parsed tensor list (from analyze), used for weight accounting
 *                 and Architecture Profile signature validation
 */
export function estimateVRAM(
  config,
  tree,
  {
    batch = 1, seq = 8192, sequenceLengths = null,
    tensors = null,
  } = {},
) {
  const totalParams = tree.totalParams;

  const w = tensors && tensors.length ? computeWeightBytes(tensors) : null;
  let vWeights, baseWeightsGB, moeWeightsGB;
  if (w) {
    vWeights = w.totalBytes / GB;
    baseWeightsGB = w.baseBytes / GB;
    moeWeightsGB = w.expertBytes / GB;
  } else {
    // Fallback: no tensor detail -> uniform 2 bytes per param.
    vWeights = (totalParams * 2) / GB;
    baseWeightsGB = (tree.baseParams * 2) / GB;
    moeWeightsGB = (tree.expertParams * 2) / GB;
  }
  const kv = computeKV({ config, tensors, batch, seq, sequenceLengths });
  const vKV = kv.vKV;
  const kvNote = kv.note || '';
  const kvUnknown = !!kv.kvUnknown;

  const complete = !kvUnknown && Number.isFinite(vKV);
  const vTotal = complete ? vWeights + vKV : null;

  // Overview composition: sum weights by Tensor Name Pattern, then add KV.
  const composition = [];
  if (w) {
    for (const [key, bytes] of w.byTensorNamePattern) {
      const dtypes = [...w.dtypesByTensorNamePattern.get(key)].sort();
      composition.push({ key, label: key, dtypes, colorKey: 'weight', group: 'weight', gb: bytes / GB });
    }
  } else {
    composition.push({ key: 'weight', labelKey: 'cat.weight', dtypes: [], colorKey: 'weight', group: 'weight', gb: vWeights });
  }
  if (complete) {
    const dtypes = [...new Set(kv.buffers.map((buffer) => buffer.dtype))].sort();
    composition.push({ key: 'kv', labelKey: 'cat.kv', dtypes, colorKey: 'kv', group: 'kv', gb: vKV });
  }
  composition.sort((a, b) => b.gb - a.gb || a.key.localeCompare(b.key));

  return {
    complete,
    kvStatus: kv.status,
    kvProfile: kv.profile,
    kvBuffers: kv.buffers || [],
    kvDiagnostic: kv.diagnostic || null,
    totalParams,
    vWeights,
    vKV,
    kvUnknown,
    kvNote,
    vTotal,
    composition,
    breakdown: {
      baseWeightsGB,
      moeWeightsGB,
      kvGB: vKV,
    },
  };
}
