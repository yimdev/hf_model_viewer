/* vram/kv/mla.js — MLA (Multi-head Latent Attention, DeepSeek family) KV Cache
 * ------------------------------------------------------------
 * Formula (Spec §4.2.2):
 *   Vkv = 2 · B · S · L · (kv_lora_rank + qk_rope_head_dim) / 1024^3
 * Key point: MLA does not store per-head K/V; it stores one compressed latent
 *   (kv_lora_rank + qk_rope_head_dim), given by the output dim of
 *   kv_a_proj_with_mqa. The latent is shared and counted once.
 *
 * VRAM basis: latent at BF16 (2B). DSA adds an FP8 indexer on top (see dsa.js).
 * ------------------------------------------------------------ */

import { num, outDim } from './detect.js';
import { t } from '../../i18n.js';

const GB = 1024 ** 3;

const MLA_KV_RE = /kv_a_proj|kv_b_proj|kv_proj|kv_a_layernorm/i;

export default {
  name: 'mla',

  /** Tensor path: kvElements = out(kv_a_proj_with_mqa). */
  computeFromTensors({ batch, seq, L, attnNames, byName }) {
    const mlaKv = attnNames.find((n) => MLA_KV_RE.test(n));
    if (!mlaKv) return null;
    const kvElements = outDim(byName.get(mlaKv));
    const vKV = (batch * seq * L * kvElements * 2) / GB;
    return { vKV, attnArch: 'mla', formulaLabel: t('kv.mla.tensor'), note: '' };
  },

  /** config fallback path. */
  computeFromConfig({ batch, seq, config }) {
    const L = num(config.num_hidden_layers ?? config.n_layers ?? config.num_layers, NaN);
    if (!Number.isFinite(L)) throw new Error(t('err.missingLayers'));
    const Dcomp = num(config.kv_lora_rank, NaN);
    const Drope = num(config.qk_rope_head_dim ?? 0, 0);
    if (!Number.isFinite(Dcomp)) throw new Error(t('err.missingKvLora'));
    const vKV = (2 * batch * seq * L * (Dcomp + Drope)) / GB;
    return {
      vKV,
      attnArch: 'mla',
      formulaLabel: t('kv.mla.config'),
      note: '',
    };
  },
};
