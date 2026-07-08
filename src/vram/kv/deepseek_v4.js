/* vram/kv/deepseek_v4.js — DeepSeek-V4 NSA (Native Sparse Attention + MLA latent) KV Cache
 * ------------------------------------------------------------
 * DeepSeek-V4 attention = MLA-style latent KV + NSA compression + top-k selection.
 * Per layer the KV cache holds TWO buffers (see inference/model.py):
 *   1) Attention.kv_cache — the compressor writes INTO its [win:] slice, so one
 *      buffer covers both the sliding window and the coarse compressed KV:
 *        size = (window_size + S/compress_ratio) * head_dim   [per (B, token)]
 *   2) Indexer.kv_cache — a SEPARATE buffer, only when compress_ratio == 4
 *      (the NSA selection branch builds its own rotated compressed KV):
 *        size = (S/4) * index_head_dim   [per (B, token)]
 *
 * The main KV is a single MLA latent of `head_dim` dims (wkv: dim -> head_dim),
 * shared across all query heads — NOT per-head K/V. The rope dims
 * (rope_head_dim) are folded inside this latent. The window is a fixed ring,
 * independent of the context length S.
 *
 * VRAM basis: KV cache stored at BF16 (2B), consistent with MLA/DSA in this
 * tool. (The reference inference code FP8-simulates the non-rope latent dims;
 * enabling fp8 KV cache would roughly halve the figure.)
 * ------------------------------------------------------------ */

import { num, outDim } from './detect.js';
import { t } from '../../i18n.js';

const GB = 1024 ** 3;
const KV_BYTES = 2; // bf16

/** Sum KV-cache elements across all attention layers for context length S. */
function kvElemsPerLayerSum({ S, L, compressRatios, windowSize, headDim, indexHeadDim }) {
  let total = 0;
  const n = Number.isFinite(L) ? L : 0;
  for (let l = 0; l < n; l++) {
    // Per-layer compression ratio; fall back to the last entry if the array is
    // shorter than the layer count (configs sometimes pad the tail).
    const cr = compressRatios && l < compressRatios.length
      ? compressRatios[l]
      : (compressRatios && compressRatios.length ? compressRatios[compressRatios.length - 1] : 0);
    const crEff = Number.isFinite(cr) ? cr : 0;
    // Main attention buffer: fixed sliding-window ring + compressed cache.
    const main = (num(windowSize, 0) + (crEff > 0 ? S / crEff : 0)) * num(headDim, 0);
    // Indexer selection buffer only for the NSA branch (compress_ratio == 4).
    const indexer = crEff === 4 ? (S / 4) * num(indexHeadDim, 0) : 0;
    total += main + indexer;
  }
  return total;
}

export default {
  name: 'deepseek_v4',

  /** Tensor path: derive head_dim from the main attn.wkv.weight shape; the
   *  per-layer compress_ratio / window / index dims come from config (they are
   *  architecture hyper-params, not encoded in tensor shapes). */
  computeFromTensors({ batch, seq, L, attnNames, byName, config = {} }) {
    const wkv = attnNames.find((n) => /attn\.wkv\.weight$/i.test(n));
    const headDim = wkv ? outDim(byName.get(wkv)) : num(config.head_dim, NaN);
    if (!Number.isFinite(headDim)) return null; // fall through to config path
    const S = seq;
    const windowSize = num(config.window_size ?? config.sliding_window, 0);
    const indexHeadDim = num(config.index_head_dim, 128);
    const compressRatios = Array.isArray(config.compress_ratios) ? config.compress_ratios : [];
    const elems = kvElemsPerLayerSum({ S, L, compressRatios, windowSize, headDim, indexHeadDim });
    const vKV = (batch * elems * KV_BYTES) / GB;
    return {
      vKV,
      attnArch: 'deepseek_v4',
      formulaLabel: t('kv.v4.tensor'),
      note: t('kv.v4.note', { window: num(windowSize, 0), hd: headDim, ihd: indexHeadDim }),
    };
  },

  /** config fallback path. */
  computeFromConfig({ batch, seq, config = {} }) {
    const L = num(config.n_layers ?? config.num_hidden_layers ?? config.num_layers, NaN);
    if (!Number.isFinite(L)) throw new Error(t('err.missingLayers'));
    const headDim = num(config.head_dim, NaN);
    if (!Number.isFinite(headDim)) throw new Error(t('err.missingHeadDim'));
    const S = seq;
    const windowSize = num(config.window_size ?? config.sliding_window, 0);
    const indexHeadDim = num(config.index_head_dim, 128);
    const compressRatios = Array.isArray(config.compress_ratios) ? config.compress_ratios : [];
    const elems = kvElemsPerLayerSum({ S, L, compressRatios, windowSize, headDim, indexHeadDim });
    const vKV = (batch * elems * KV_BYTES) / GB;
    return {
      vKV,
      attnArch: 'deepseek_v4',
      formulaLabel: t('kv.v4.config'),
      note: t('kv.v4.note', { window: num(windowSize, 0), hd: headDim, ihd: indexHeadDim }),
    };
  },
};
