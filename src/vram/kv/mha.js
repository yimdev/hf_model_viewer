/* vram/kv/mha.js — Standard attention (MHA / GQA / MQA) KV Cache
 * ------------------------------------------------------------
 * Formula (Spec §4.2.2):
 *   Vkv = 4 · B · S · L · Hkv · Dhead / 1024^3
 * where Hkv is the number of KV heads (MHA: = nHeads; GQA: < nHeads;
 * MQA: = 1).
 *
 * Two input paths:
 *   - computeFromTensors: preferred; derived from K/V projection shapes (vendor-neutral).
 *   - computeFromConfig: fallback; derived from config.json hyper-params
 *     (with fused-QKV field fallbacks).
 *
 * VRAM basis: KV counted at BF16 (2B), matching Spec; not scaled by the
 * precision slider.
 * ------------------------------------------------------------ */

import { num, outDim } from './detect.js';
import { t } from '../../i18n.js';

const GB = 1024 ** 3;

function labelOf(Hkv, nHeads) {
  if (!Number.isFinite(nHeads) || nHeads <= 0) return 'mha';
  if (Hkv === nHeads) return 'mha';
  if (Hkv === 1) return 'mqa';
  return 'gqa';
}

export default {
  name: 'mha',

  /** Tensor path: kvElements = out(k_proj) + out(v_proj); L given by caller. */
  computeFromTensors({ batch, seq, L, attnNames, byName, config }) {
    const kProj = attnNames.find((n) => /(^|[._])k_proj/i.test(n));
    const vProj = attnNames.find((n) => /(^|[._])v_proj/i.test(n));
    if (!kProj || !vProj) return null; // fused QKV -> let config handle it

    const kvElements = outDim(byName.get(kProj)) + outDim(byName.get(vProj));
    const vKV = (batch * seq * L * kvElements * 2) / GB;

    // Use the most stable num_attention_heads only for label refinement
    // (numeric values still come from tensors).
    const kOut = outDim(byName.get(kProj));
    const qProj = attnNames.find((n) => /(^|[._])q_proj/i.test(n));
    const qOut = qProj ? outDim(byName.get(qProj)) : NaN;
    const nHeads = num(config.num_attention_heads ?? config.n_heads, NaN);

    let attnArch = 'mha';
    let formulaLabel = t('kv.mha.tensor');
    if (Number.isFinite(qOut) && Number.isFinite(nHeads) && nHeads > 0) {
      const Dhead = qOut / nHeads;
      const Hkv = Number.isFinite(Dhead) && Dhead > 0 ? Math.round(kOut / Dhead) : kOut;
      attnArch = labelOf(Hkv, nHeads);
      formulaLabel = t('kv.mha.tensor.detail', {
        arch: attnArch.toUpperCase(),
        hkv: Hkv,
        dhead: Math.round(Dhead),
      });
    }
    return { vKV, attnArch, formulaLabel, note: '' };
  },

  /** config fallback path: multi-source field fallbacks covering vendor names. */
  computeFromConfig({ batch, seq, config }) {
    const L = num(config.num_hidden_layers ?? config.n_layers ?? config.num_layers, NaN);
    if (!Number.isFinite(L)) throw new Error(t('err.missingLayers'));

    const nHeads = config.num_attention_heads ?? config.n_heads;
    let Hkv = config.num_key_value_heads ?? config.num_kv_heads;
    if (Hkv === undefined && config.multi_query_attention) {
      Hkv = config.multi_query_group_num ?? 1;
    }
    if (Hkv === undefined) Hkv = nHeads;
    const hidden = config.hidden_size ?? config.d_model;
    const Dhead = config.head_dim ?? config.kv_channels ?? Math.floor(hidden / nHeads);
    if (![Hkv, Dhead, nHeads].every(Number.isFinite)) throw new Error(t('err.missingAttn'));

    const vKV = (4 * batch * seq * L * Hkv * Dhead) / GB;
    const attnArch = labelOf(Hkv, nHeads);
    return {
      vKV,
      attnArch,
      formulaLabel: t('kv.mha.config', { arch: attnArch.toUpperCase(), hkv: Hkv, dhead: Dhead }),
      note: '',
    };
  },
};
