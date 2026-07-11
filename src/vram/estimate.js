const GB = 1024 ** 3;

import { createTensorMetadataIndex } from '../tree/tensorMetadataIndex.js';
import { computeKV } from './kv/index.js';

export function estimateVRAM({
  source = {},
  config = {},
  tensors = [],
  workload: {
    batch = 1, seq = 8192, sequenceLengths = null,
  } = {},
  tensorMetadataIndex = null,
} = {}) {
  const tensorIndex = tensorMetadataIndex || createTensorMetadataIndex(tensors);
  const vWeights = tensorIndex.totalWeightBytes / GB;
  const result = computeKV({
    source,
    config,
    batch,
    seq,
    sequenceLengths,
  });
  const computed = result.calculation.status === 'computed';
  const vKV = result.vKV;
  const vTotal = computed ? vWeights + vKV : null;

  const composition = tensorIndex.tensorNamePatterns.map(({ key, bytes, dtypes }) => ({
    key,
    label: key,
    dtypes,
    colorKey: 'weight',
    group: 'weight',
    gb: bytes / GB,
  }));
  if (computed) {
    const dtypes = [...new Set(result.buffers.map((buffer) => buffer.dtype))].sort();
    composition.push({
      key: 'kv',
      labelKey: 'cat.kv',
      dtypes,
      colorKey: 'kv',
      group: 'kv',
      gb: vKV,
    });
  }
  composition.sort((left, right) => right.gb - left.gb || left.key.localeCompare(right.key));

  return Object.freeze({
    complete: computed,
    calculation: result.calculation,
    assurance: result.assurance,
    provenance: result.provenance,
    profile: result.profile,
    approximation: result.approximation || null,
    buffers: result.buffers,
    note: result.note,
    totalParams: tensorIndex.summary.totalParams,
    vWeights,
    vKV,
    vTotal,
    composition: Object.freeze(composition),
    breakdown: Object.freeze({
      baseWeightsGB: tensorIndex.baseWeightBytes / GB,
      moeWeightsGB: tensorIndex.expertWeightBytes / GB,
      kvGB: vKV,
    }),
  });
}
