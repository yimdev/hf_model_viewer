/* vram/kv/index.js — KV Cache dispatcher (registry + dual-path routing)
 * ------------------------------------------------------------
 * Each architecture lives in its own file (mha.js / mla.js / dsa.js); arch
 * detection lives in detect.js. All three sit in this directory so adding an
 * architecture is just: add a file + register it here.
 *
 * Dispatch strategy:
 *   1) Primary "tensor path": derive KV from safetensors tensor shapes
 *      (vendor-neutral, no field-name guessing). detectArchFromTensors picks
 *      the arch, then the module's computeFromTensors runs.
 *   2) Fallback "config path": only when tensors can't be split by shape
 *      (e.g. fused QKV) or are missing; detectAttnArch picks the arch, then
 *      the module's computeFromConfig runs.
 *   3) DSA upgrade: tensors identified as MLA latent but config exposes
 *      index_head_dim -> add the FP8 indexer.
 *
 * Returns: { vKV, attnArch, formulaLabel, note, kvUnknown?, kvMethod }
 *   attnArch ∈ { 'mha' | 'gqa' | 'mqa' | 'mla' | 'dsa' } (lowercase, maps
 *     directly to UI badges)
 *   kvMethod ∈ { 'tensors' | 'config' }
 * ------------------------------------------------------------ */

import mha from './mha.js';
import mla from './mla.js';
import dsa from './dsa.js';
import { DSA_NOTE_KEY } from './dsa.js';
import {
  detectAttnArch,
  detectArchFromTensors,
  extractLayerTensors,
  num,
  GB,
} from './detect.js';
import { t } from '../../i18n.js';

const REGISTRY = { mha, mla, dsa };

/**
 * Unified entry point: compute KV Cache.
 * @param {object} opts { config, tensors?, precision?, batch?, seq? }
 */
export function computeKV({ config = {}, tensors = null, precision = 'fp16', batch = 1, seq = 8192 } = {}) {
  // ── Primary path: derive from tensor shapes (generic, no field guessing) ──
  if (Array.isArray(tensors) && tensors.length) {
    const parsed = extractLayerTensors(tensors);
    if (parsed) {
      const archName = detectArchFromTensors(parsed.attnNames);
      if (archName) {
        const mod = REGISTRY[archName];
        const r = mod.computeFromTensors({ ...parsed, config, batch, seq });
        if (r && Number.isFinite(r.vKV)) {
          // DSA upgrade: tensors identified as MLA latent, but config exposes
          // index_head_dim -> add the FP8 indexer.
          if (archName === 'mla' && detectAttnArch(config) === 'dsa') {
            const Didx = num(config.index_head_dim ?? 128, 128);
            r.vKV += (batch * seq * parsed.L * Didx * 1) / GB;
            r.attnArch = 'dsa';
            r.formulaLabel = t('kv.dsa.tensor');
            r.note = t(DSA_NOTE_KEY);
          }
          return { ...r, kvMethod: 'tensors' };
        }
      }
    }
  }

  // ── Fallback: derive from config hyper-params (covers fused QKV etc.) ──
  const archName = detectAttnArch(config);
  const mod = REGISTRY[archName] || REGISTRY.mha;
  try {
    const r = mod.computeFromConfig({ config, batch, seq });
    return { ...r, kvMethod: 'config' };
  } catch {
    return {
      vKV: null,
      attnArch: archName,
      formulaLabel: '',
      note: '',
      kvUnknown: true,
      kvMethod: 'config',
    };
  }
}

export { detectAttnArch, detectArchFromTensors };
