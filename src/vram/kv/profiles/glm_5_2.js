import { makeBuffer, validateSequenceWorkload } from '../profile-primitives.js';
import { defineArchitectureProfile, profileConfigInput } from '../profile-definition.js';

const FULL_INDEXER_LAYERS = [
  0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74,
];
const INDEXER_TYPES = Array.from({ length: 78 }, (_, layer) =>
  FULL_INDEXER_LAYERS.includes(layer) ? 'full' : 'shared');

function calculateLayout({
  inputs: {
    layerCount, kvLoraRank, ropeHeadDim, indexHeadDim, indexerTypes, maxContext,
  },
  workload: { batch, seq, sequenceLengths },
}) {
  const workload = validateSequenceWorkload({ batch, seq, sequenceLengths, maxContext });
  if (workload.error) return workload;
  const tokenCount = workload.tokenCount;
  const fullIndexerLayers = indexerTypes
    .slice(0, layerCount)
    .reduce((count, type) => count + (type === 'full' ? 1 : 0), 0);
  const tokenFormula = 'T = B × S or Σ sequence_lengths';
  const buffers = [
    makeBuffer({
      id: 'mla-latent',
      label: 'MLA compressed latent',
      layerGroup: { label: 'all backbone layers', count: layerCount, range: [0, layerCount - 1] },
      elements: tokenCount * layerCount * kvLoraRank,
      dtype: 'BF16',
      bytesPerElement: 2,
      formula: `T × ${layerCount} × ${kvLoraRank}; ${tokenFormula}`,
    }),
    makeBuffer({
      id: 'mla-rope-key',
      label: 'MLA RoPE key',
      layerGroup: { label: 'all backbone layers', count: layerCount, range: [0, layerCount - 1] },
      elements: tokenCount * layerCount * ropeHeadDim,
      dtype: 'BF16',
      bytesPerElement: 2,
      formula: `T × ${layerCount} × ${ropeHeadDim}; ${tokenFormula}`,
    }),
    makeBuffer({
      id: 'indexer-key',
      label: 'IndexShare full-layer indexer key',
      layerGroup: {
        label: 'full indexer layers',
        count: fullIndexerLayers,
        indices: indexerTypes
          .slice(0, layerCount)
          .flatMap((type, layer) => (type === 'full' ? [layer] : [])),
      },
      elements: tokenCount * fullIndexerLayers * indexHeadDim,
      dtype: 'BF16',
      bytesPerElement: 2,
      formula: `T × ${fullIndexerLayers} × ${indexHeadDim}; ${tokenFormula}`,
    }),
  ];

  return {
    buffers,
    note: 'All history is retained; index_topk limits reads, not cache capacity.',
  };
}

const BASELINE_INPUTS = Object.freeze({
  layerCount: 78,
  kvLoraRank: 512,
  ropeHeadDim: 64,
  indexHeadDim: 128,
  indexerTypes: INDEXER_TYPES,
  maxContext: 1048576,
});

export default defineArchitectureProfile({
  id: 'glm-5.2-semantic-bf16-v1',
  version: '2.0.0',
  label: 'GLM 5.2',
  layout: { id: 'glm-5.2-indexshare-bf16-v1', version: '2.0.0' },
  repositories: [
    {
      repoId: 'zai-org/GLM-5.2',
      auditedCommitId: 'b4734de4facf877f85769a911abafc5283eab3d9',
      baselineInputs: BASELINE_INPUTS,
    },
    {
      repoId: 'zai-org/GLM-5.2-FP8',
      auditedCommitId: 'ba978f7d347eaf65d22f1a86833408afdb953541',
      baselineInputs: BASELINE_INPUTS,
    },
  ],
  configInputs: {
    layerCount: profileConfigInput.positiveInteger('num_hidden_layers'),
    kvLoraRank: profileConfigInput.positiveInteger('kv_lora_rank'),
    ropeHeadDim: profileConfigInput.positiveInteger('qk_rope_head_dim'),
    indexHeadDim: profileConfigInput.positiveInteger('index_head_dim'),
    indexerTypes: profileConfigInput.array('indexer_types'),
    maxContext: profileConfigInput.positiveInteger('max_position_embeddings'),
  },
  validateInputs(inputs) {
    if (inputs.indexerTypes.length < inputs.layerCount) {
      return [{ input: 'indexerTypes', code: 'invalid_length' }];
    }
    if (inputs.indexerTypes.slice(0, inputs.layerCount).some((type) => !['full', 'shared'].includes(type))) {
      return [{ input: 'indexerTypes', code: 'unsupported_type' }];
    }
    return [];
  },
  calculateLayout,
});
