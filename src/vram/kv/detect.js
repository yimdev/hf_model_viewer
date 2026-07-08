/* vram/kv/detect.js — Attention-arch detection + tensor geometry parsing (shared utils)
 * ------------------------------------------------------------
 * This file sits in the same directory as mha.js / mla.js / dsa.js and does
 * two things:
 *   1) Detect the arch: detectAttnArch (from config.json) /
 *      detectArchFromTensors (from tensor names).
 *   2) Parse the raw tensor list into a "representative layer" attention
 *      tensor set, consumed by each arch module.
 *
 * Design principle: KV Cache counts only tensors that LAND in VRAM; the layer
 * count L comes directly from tensor name indices (no reliance on config's
 * num_hidden_layers / num_layers naming differences).
 * ------------------------------------------------------------ */

const GB = 1024 ** 3;

export function num(v, d = 1) {
  return Number.isFinite(v) ? v : d;
}

/** Parse the layer index from a tensor name; supports layers.N / layer.N /
 *  h.N / blocks.N / transformer.layers.N. */
export function layerOf(name) {
  const m = name.match(/(?:^|[._])(?:layers?|h|blocks?)\.(\d+)\./);
  if (m) return +m[1];
  return -1;
}

/** Whether a tensor is attention-related (excludes mlp / experts / ffn / moe;
 *  includes DSA indexer). */
export function isAttnTensor(name) {
  if (/(?:mlp|expert|ffn|moe)/i.test(name)) return false;
  return /(?:attn|attention|self_attn|kv|k_proj|v_proj|q_proj|query_key_value|qkv|indexer|index_key|index_k)/i.test(name);
}

/** Linear weight [out, in] output dim = shape[0]. */
export function outDim(meta) {
  const s = meta && meta.shape;
  return Array.isArray(s) && s.length >= 1 ? s[0] : NaN;
}

/**
 * Parse the tensor list, extracting attention tensors of the "representative
 * layer". Representative layer = the one with the most attention tensors
 * (ties broken by larger layer index), more robust than just taking the max
 * layer (avoids an incomplete last layer missing K/V projections).
 * @returns {{L:number, sampleLayer:number, attnNames:string[], byName:Map}|null}
 *          null when no layer index is found (e.g. pure embedding) -> config fallback.
 */
export function extractLayerTensors(tensors) {
  if (!Array.isArray(tensors) || tensors.length === 0) return null;
  const layerSet = new Set();
  const attnCount = new Map();
  for (const t of tensors) {
    const L = layerOf(t.name);
    if (L < 0) continue;
    layerSet.add(L);
    if (isAttnTensor(t.name)) attnCount.set(L, (attnCount.get(L) || 0) + 1);
  }
  if (layerSet.size === 0) return null;
  const L = Math.max(...layerSet) + 1;

  // Representative layer = most attention tensors; tie -> largest layer index.
  let repLayer = -1;
  let repCount = -1;
  for (const lyr of layerSet) {
    const c = attnCount.get(lyr) || 0;
    if (c > repCount || (c === repCount && lyr > repLayer)) {
      repCount = c;
      repLayer = lyr;
    }
  }

  const byName = new Map(tensors.map((t) => [t.name, t]));
  const attnNames = tensors
    .filter((t) => layerOf(t.name) === repLayer && isAttnTensor(t.name))
    .map((t) => t.name);
  return { L, sampleLayer: repLayer, attnNames, byName };
}

/**
 * Detect arch from config.json; priority dsa > mla > mha.
 * Only used as the config fallback when tensors can't be split by shape.
 */
export function detectAttnArch(config = {}) {
  const arch = Array.isArray(config.architectures) ? config.architectures.join(' ') : '';
  const modelType = config.model_type || '';
  // DeepSeek-V4: NSA + MLA latent. Detect BEFORE the generic index_* heuristic,
  // because V4 also exposes index_head_dim / index_topk (which would otherwise
  // be misclassified as V3.2 DSA, whose formula does not apply to V4).
  if (
    modelType === 'deepseek_v4' ||
    /deepseekv4/i.test(arch) ||
    /DeepseekV4ForCausalLM/i.test(arch)
  ) {
    return 'deepseek_v4';
  }
  const hasDSA =
    config.index_head_dim !== undefined ||
    config.index_topk !== undefined ||
    /deepseekv3\.?2/i.test(arch) ||
    modelType === 'deepseek_v32';
  if (hasDSA) return 'dsa';
  if (config.kv_lora_rank !== undefined && config.kv_lora_rank !== null) return 'mla';
  return 'mha';
}

/**
 * Detect arch from representative-layer attention tensor names: dsa > mla > mha.
 * Returns 'dsa' | 'mla' | 'mha' | null (null = fused QKV, can't split by shape).
 */
export function detectArchFromTensors(attnNames = []) {
  const has = (re) => attnNames.some((n) => re.test(n));
  if (has(/indexer|lightning|index_key|index_k/i)) return 'dsa';
  // DeepSeek-V4: MLA latent (attn.wkv.weight) + NSA compressor (attn.compressor.*).
  if (has(/attn\.wkv\.weight/i) && has(/compressor/i)) return 'deepseek_v4';
  if (has(/kv_a_proj|kv_b_proj|kv_proj|kv_a_layernorm/i)) return 'mla';
  if (has(/(^|[._])k_proj/i) && has(/(^|[._])v_proj/i)) return 'mha';
  return null;
}

export { GB };
