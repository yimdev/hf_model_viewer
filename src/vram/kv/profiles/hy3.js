import { makeBuffer, validateSequenceWorkload } from '../profile-primitives.js';
import { defineArchitectureProfile, profileConfigInput } from '../profile-definition.js';

function calculateLayout({
  inputs: { layerCount, kvHeadCount, headDim, maxContext },
  workload: { batch, seq, sequenceLengths },
}) {
  const workload = validateSequenceWorkload({ batch, seq, sequenceLengths, maxContext });
  if (workload.error) return workload;
  const elementsPerBuffer = workload.tokenCount * layerCount * kvHeadCount * headDim;
  const tokenFormula = 'T = B × S or Σ sequence_lengths';
  const layerGroup = {
    label: 'all backbone layers',
    count: layerCount,
    range: [0, layerCount - 1],
  };
  const formula = `T × ${layerCount} × ${kvHeadCount} × ${headDim}; ${tokenFormula}`;
  const buffers = [
    makeBuffer({
      id: 'main.key',
      label: 'Full-context normalized and RoPE key',
      layerGroup,
      elements: elementsPerBuffer,
      dtype: 'BF16',
      bytesPerElement: 2,
      formula,
    }),
    makeBuffer({
      id: 'main.value',
      label: 'Full-context value',
      layerGroup,
      elements: elementsPerBuffer,
      dtype: 'BF16',
      bytesPerElement: 2,
      formula,
    }),
  ];
  return {
    buffers,
    note: 'The MTP layer is excluded as an optional speculative runtime.',
  };
}

const BASELINE_INPUTS = Object.freeze({
  layerCount: 80,
  kvHeadCount: 8,
  headDim: 128,
  maxContext: 262144,
});

export default defineArchitectureProfile({
  id: 'hy3-instruct-semantic-bf16-v1',
  version: '2.0.0',
  label: 'Hunyuan 3 (Hy3)',
  layout: { id: 'hy3-full-gqa-bf16-v1', version: '2.0.0' },
  repositories: [
    {
      repoId: 'tencent/Hy3',
      auditedCommitId: '716aa7241bd6d95896be4ebfc761162a9c4d49ef',
      baselineInputs: BASELINE_INPUTS,
    },
    {
      repoId: 'tencent/Hy3-preview',
      auditedCommitId: 'b53bd705bef15f0a9e52eade60a4353eaaa6c6b8',
      baselineInputs: BASELINE_INPUTS,
    },
  ],
  configInputs: {
    layerCount: profileConfigInput.positiveInteger('num_hidden_layers'),
    kvHeadCount: profileConfigInput.positiveInteger('num_key_value_heads'),
    headDim: profileConfigInput.positiveInteger('head_dim'),
    maxContext: profileConfigInput.positiveInteger('max_position_embeddings'),
  },
  calculateLayout,
});
