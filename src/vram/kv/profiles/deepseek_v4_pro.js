import {
  makeBuffer, validateSequenceWorkload,
} from '../profile-primitives.js';
import { defineArchitectureProfile, profileConfigInput } from '../profile-definition.js';

const COMPRESS_RATIOS = [
  128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
  4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
  4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
  4, 0,
];
const PROFILE = Object.freeze({
  id: 'deepseek-v4-pro-instruct-b5968e9',
  version: '2.0.0',
  label: 'DeepSeek V4 Pro',
  layout: Object.freeze({ id: 'deepseek-v4-pro-csa-hca-bf16-v1', version: '2.0.0' }),
});

function layerGroup(label, indices) {
  return { label, count: indices.length, indices };
}

function calculateLayout({ inputs, workload: { batch, seq, sequenceLengths } }) {
  const {
    layerCount, headDim, indexHeadDim, slidingWindow, maxContext, compressionByLayer,
  } = inputs;
  const workload = validateSequenceWorkload({
    batch,
    seq,
    sequenceLengths,
    maxContext,
    minimumBatch: 1,
    allowEmptySequenceLengths: false,
  });
  if (workload.error) return workload;
  const sum = (calculate) => workload.entries.reduce(
    (total, { length, count }) => total + count * calculate(length),
    0,
  );
  const localAndCompressed128 = sum((length) => Math.min(length, slidingWindow) + Math.floor(length / 128));
  const remainder128 = sum((length) => length % 128);
  const localAndCompressed4 = sum((length) => Math.min(length, slidingWindow) + Math.floor(length / 4));
  const compressed4 = sum((length) => Math.floor(length / 4));
  const active4 = sum((length) => (length < 4 ? length : 4 + (length % 4)));
  const hcaLayers = compressionByLayer.slice(0, layerCount)
    .map((ratio, layer) => ({ ratio, layer }))
    .filter(({ ratio }) => ratio === 128)
    .map(({ layer }) => layer);
  const csaLayers = compressionByLayer.slice(0, layerCount)
    .map((ratio, layer) => ({ ratio, layer }))
    .filter(({ ratio }) => ratio === 4)
    .map(({ layer }) => layer);
  const hca = layerGroup('HCA layers (compression ratio 128)', hcaLayers);
  const csa = layerGroup('CSA layers (compression ratio 4)', csaLayers);
  const buffers = [
    makeBuffer({
      id: 'hca-kv', label: 'HCA local and compressed KV', layerGroup: hca,
      elements: hcaLayers.length * headDim * localAndCompressed128,
      dtype: 'BF16', bytesPerElement: 2,
      formula: `Σᵢ ${hcaLayers.length} × ${headDim} × (min(Sᵢ,${slidingWindow}) + floor(Sᵢ/128))`,
    }),
    makeBuffer({
      id: 'hca-kv-state', label: 'HCA compressor KV state', layerGroup: hca,
      elements: hcaLayers.length * headDim * remainder128,
      dtype: 'F32', bytesPerElement: 4,
      formula: `Σᵢ ${hcaLayers.length} × ${headDim} × (Sᵢ mod 128)`,
    }),
    makeBuffer({
      id: 'hca-score-state', label: 'HCA compressor score state', layerGroup: hca,
      elements: hcaLayers.length * headDim * remainder128,
      dtype: 'F32', bytesPerElement: 4,
      formula: `Σᵢ ${hcaLayers.length} × ${headDim} × (Sᵢ mod 128)`,
    }),
    makeBuffer({
      id: 'csa-kv', label: 'CSA local and compressed KV', layerGroup: csa,
      elements: csaLayers.length * headDim * localAndCompressed4,
      dtype: 'BF16', bytesPerElement: 2,
      formula: `Σᵢ ${csaLayers.length} × ${headDim} × (min(Sᵢ,${slidingWindow}) + floor(Sᵢ/4))`,
    }),
    makeBuffer({
      id: 'csa-indexer-kv', label: 'CSA compressed indexer KV', layerGroup: csa,
      elements: csaLayers.length * indexHeadDim * compressed4,
      dtype: 'BF16', bytesPerElement: 2,
      formula: `Σᵢ ${csaLayers.length} × ${indexHeadDim} × floor(Sᵢ/4)`,
    }),
    makeBuffer({
      id: 'csa-kv-state', label: 'CSA compressor KV state', layerGroup: csa,
      elements: csaLayers.length * headDim * 2 * active4,
      dtype: 'F32', bytesPerElement: 4,
      formula: `Σᵢ ${csaLayers.length} × ${headDim * 2} × A4(Sᵢ)`,
    }),
    makeBuffer({
      id: 'csa-score-state', label: 'CSA compressor score state', layerGroup: csa,
      elements: csaLayers.length * headDim * 2 * active4,
      dtype: 'F32', bytesPerElement: 4,
      formula: `Σᵢ ${csaLayers.length} × ${headDim * 2} × A4(Sᵢ)`,
    }),
    makeBuffer({
      id: 'csa-indexer-kv-state', label: 'CSA indexer compressor KV state', layerGroup: csa,
      elements: csaLayers.length * indexHeadDim * 2 * active4,
      dtype: 'F32', bytesPerElement: 4,
      formula: `Σᵢ ${csaLayers.length} × ${indexHeadDim * 2} × A4(Sᵢ)`,
    }),
    makeBuffer({
      id: 'csa-indexer-score-state', label: 'CSA indexer compressor score state', layerGroup: csa,
      elements: csaLayers.length * indexHeadDim * 2 * active4,
      dtype: 'F32', bytesPerElement: 4,
      formula: `Σᵢ ${csaLayers.length} × ${indexHeadDim * 2} × A4(Sᵢ)`,
    }),
  ];

  return {
    buffers,
    note: 'MTP and runtime capacity are excluded; index_topk limits reads, not cache capacity.',
  };
}

export default defineArchitectureProfile({
  ...PROFILE,
  repositories: [{
    repoId: 'deepseek-ai/DeepSeek-V4-Pro',
    auditedCommitId: 'b5968e9190ef611bbf34a7229255be88a0e937c1',
    baselineInputs: {
      layerCount: 61,
      headDim: 512,
      indexHeadDim: 128,
      slidingWindow: 128,
      maxContext: 1048576,
      compressionByLayer: COMPRESS_RATIOS,
    },
  }],
  configInputs: {
    layerCount: profileConfigInput.positiveInteger('num_hidden_layers'),
    headDim: profileConfigInput.positiveInteger('head_dim'),
    indexHeadDim: profileConfigInput.positiveInteger('index_head_dim'),
    slidingWindow: profileConfigInput.positiveInteger('sliding_window'),
    maxContext: profileConfigInput.positiveInteger('max_position_embeddings'),
    compressionByLayer: profileConfigInput.array('compress_ratios'),
  },
  validateInputs(inputs) {
    if (inputs.compressionByLayer.length < inputs.layerCount) {
      return [{ input: 'compressionByLayer', code: 'invalid_length' }];
    }
    if (inputs.compressionByLayer.slice(0, inputs.layerCount).some((ratio) => ![4, 128].includes(ratio))) {
      return [{ input: 'compressionByLayer', code: 'unsupported_ratio' }];
    }
    return [];
  },
  calculateLayout,
});
