/* vram/kv/dsa.js — DSA (DeepSeek Sparse Attention, V3.2) KV Cache
 * ------------------------------------------------------------
 * DSA = MLA latent + FP8 indexer K cache:
 *   Vkv = B · S · L · [ 2·(kv_lora_rank + qk_rope_head_dim) + index_head_dim ] / 1024^3
 *
 * Architectural fact (important): the core benefit of DSA is reduced COMPUTE
 *   O(L²)→O(L·topk) and reduced memory BANDWIDTH (only top-k KV loaded per
 *   autoregressive step). It does NOT shrink KV Cache capacity. The VRAM KV
 *   Cache ≈ dense MLA full latent + indexer, roughly flat, slightly larger by
 *   index_head_dim / (kv_lora_rank + qk_rope_head_dim) ≈ 11%.
 *   — The top-k selection is a load-time behavior during compute, occupies no
 *   VRAM, and must NOT be counted in Vkv (see the VRAM-basis principle).
 *
 * VRAM basis: MLA latent at BF16 (2B); indexer K fixed FP8 (1B).
 * ------------------------------------------------------------ */

import { num, outDim } from './detect.js';
import { t } from '../../i18n.js';

const GB = 1024 ** 3;

/** i18n key for the DSA note (also used by the MLA→DSA upgrade in index.js). */
export const DSA_NOTE_KEY = 'kv.dsa.note';

const MLA_KV_RE = /kv_a_proj|kv_b_proj|kv_proj|kv_a_layernorm/i;
const INDEXER_RE = /indexer|lightning|index_key|index_k/i;

export default {
  name: 'dsa',

  /** Tensor path: MLA latent + indexer K (FP8). If the indexer tensor is
   *  missing, return null so the upgrade logic in index.js handles it. */
  computeFromTensors({ batch, seq, L, attnNames, byName }) {
    const mlaKv = attnNames.find((n) => MLA_KV_RE.test(n));
    const indexer = attnNames.find((n) => INDEXER_RE.test(n));
    if (!mlaKv || !indexer) return null; // missing indexer -> MLA+config upgrade in index.js
    const kvElements = outDim(byName.get(mlaKv));
    const indexerDim = outDim(byName.get(indexer));
    const vKV = (batch * seq * L * (kvElements * 2 + indexerDim * 1)) / GB;
    return {
      vKV,
      attnArch: 'dsa',
      formulaLabel: t('kv.dsa.tensor'),
      note: t(DSA_NOTE_KEY),
    };
  },

  /** config fallback path. */
  computeFromConfig({ batch, seq, config }) {
    const L = num(config.num_hidden_layers ?? config.n_layers ?? config.num_layers, NaN);
    if (!Number.isFinite(L)) throw new Error(t('err.missingLayers'));
    const Dcomp = num(config.kv_lora_rank, NaN);
    const Drope = num(config.qk_rope_head_dim ?? 0, 0);
    const Didx = num(config.index_head_dim ?? 128, 128);
    if (!Number.isFinite(Dcomp)) throw new Error(t('err.missingKvLora'));
    const vKV = (batch * seq * L * (2 * (Dcomp + Drope) + Didx)) / GB;
    return {
      vKV,
      attnArch: 'dsa',
      formulaLabel: t('kv.dsa.config'),
      note: t(DSA_NOTE_KEY),
    };
  },
};
