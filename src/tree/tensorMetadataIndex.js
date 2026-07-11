import { buildTensorNameTree } from './buildTensorNameTree.js';
import { groupRepeatedTensorSubtrees } from './groupRepeatedTensorSubtrees.js';
import { isNumericPathSegment } from './pathSegments.js';

const DTYPE_BYTES = Object.freeze({
  F64: 8, F32: 4, F16: 2, BF16: 2,
  F8_E4M3FN: 1, F8_E5M2: 1, F8_E4M3FNUZ: 1, F8_E5M2FNUZ: 1,
  INT8: 1, UINT8: 1, INT4: 0.5, UINT4: 0.5,
  F4: 0.5, NF4: 0.5, F4E2M3FNUZ: 0.5,
  I16: 2, I32: 4, I64: 8, BOOL: 0.125,
});

const LAYER_RE = /\.(\d+)\./;
const EXPERT_RE = /experts\.(\d+)\./;
const EXPERT_TOKEN_RE = /experts\.\d+/i;
const cache = new WeakMap();

function tensorParams(shape) {
  if (!Array.isArray(shape) || shape.length === 0) return 1;
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function bytesForDtype(dtype, fallback = 2) {
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

function tensorNamePattern(name) {
  return name
    .split('.')
    .map((segment) => (isNumericPathSegment(segment) ? '*' : segment))
    .join('.');
}

function attachSubtreeStats(node) {
  let params = 0;
  let bytes = 0;
  let tensorCount = 0;

  for (const tensor of node.tensors) {
    params += tensor.params;
    bytes += tensor.weightBytes;
    tensorCount += 1;
  }
  for (const child of node.children) {
    const stats = attachSubtreeStats(child);
    params += stats.params;
    bytes += stats.bytes;
    tensorCount += stats.tensors;
  }
  node.stats = Object.freeze({ params, bytes, tensors: tensorCount });
  return node.stats;
}

function buildSummary(tensors) {
  let totalParams = 0;
  let baseParams = 0;
  let expertParams = 0;
  let maxLayerIndex = -1;
  const expertsByLayer = new Map();

  for (const tensor of tensors) {
    totalParams += tensor.params;
    const layerMatch = tensor.name.match(LAYER_RE);
    const remainder = layerMatch
      ? tensor.name.slice(layerMatch.index + layerMatch[0].length)
      : '';
    const expertMatch = remainder.match(EXPERT_RE);
    if (layerMatch) maxLayerIndex = Math.max(maxLayerIndex, Number(layerMatch[1]));
    if (layerMatch && expertMatch) {
      expertParams += tensor.params;
      const layer = Number(layerMatch[1]);
      if (!expertsByLayer.has(layer)) expertsByLayer.set(layer, new Set());
      expertsByLayer.get(layer).add(Number(expertMatch[1]));
    } else {
      baseParams += tensor.params;
    }
  }

  return Object.freeze({
    totalParams,
    baseParams,
    expertParams,
    numLayers: maxLayerIndex + 1,
    isMoe: expertsByLayer.size > 0,
    numExperts: Math.max(0, ...[...expertsByLayer.values()].map((ids) => ids.size)),
  });
}

export function createTensorMetadataIndex(tensors) {
  if (!Array.isArray(tensors)) throw new TypeError('Tensor Metadata Index requires an array');
  const cached = cache.get(tensors);
  if (cached) return cached;

  const byTensorNamePattern = new Map();
  const dtypesByTensorNamePattern = new Map();
  const normalizedTensors = tensors.map((tensor) => {
    const params = tensor.params != null ? tensor.params : tensorParams(tensor.shape);
    const bytesPerParameter = bytesForDtype(tensor.dtype, 2);
    const weightBytes = params * bytesPerParameter;
    const pattern = tensorNamePattern(tensor.name);
    byTensorNamePattern.set(pattern, (byTensorNamePattern.get(pattern) || 0) + weightBytes);
    if (!dtypesByTensorNamePattern.has(pattern)) dtypesByTensorNamePattern.set(pattern, new Set());
    dtypesByTensorNamePattern.get(pattern).add(tensor.dtype);
    return Object.freeze({ ...tensor, params, bytesPerParameter, weightBytes });
  });

  const summary = buildSummary(normalizedTensors);
  const tensorNamePatterns = [...byTensorNamePattern].map(([key, bytes]) => Object.freeze({
    key,
    bytes,
    dtypes: Object.freeze([...dtypesByTensorNamePattern.get(key)].sort()),
  }));
  const tensorNameTree = groupRepeatedTensorSubtrees(buildTensorNameTree(normalizedTensors));
  attachSubtreeStats(tensorNameTree);

  const index = Object.freeze({
    tensors: Object.freeze(normalizedTensors),
    summary,
    totalWeightBytes: normalizedTensors.reduce((sum, tensor) => sum + tensor.weightBytes, 0),
    baseWeightBytes: normalizedTensors.reduce(
      (sum, tensor) => sum + (EXPERT_TOKEN_RE.test(tensor.name) ? 0 : tensor.weightBytes),
      0,
    ),
    expertWeightBytes: normalizedTensors.reduce(
      (sum, tensor) => sum + (EXPERT_TOKEN_RE.test(tensor.name) ? tensor.weightBytes : 0),
      0,
    ),
    tensorNamePatterns: Object.freeze(tensorNamePatterns),
    tensorNameTree,
  });
  cache.set(tensors, index);
  return index;
}
