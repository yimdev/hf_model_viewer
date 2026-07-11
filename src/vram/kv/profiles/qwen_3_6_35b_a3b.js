import { makeBuffer, validateSequenceWorkload } from '../profile-primitives.js';
import { defineArchitectureProfile, profileConfigInput } from '../profile-definition.js';

const LAYER_TYPES = Array.from(
  { length: 40 },
  (_, layer) => ((layer + 1) % 4 === 0 ? 'full_attention' : 'linear_attention'),
);
const DTYPE = Object.freeze({
  bfloat16: Object.freeze({ semantic: 'BF16', bytes: 2 }),
  float16: Object.freeze({ semantic: 'F16', bytes: 2 }),
  float32: Object.freeze({ semantic: 'F32', bytes: 4 }),
});

function dtypeInput(path) {
  return Object.freeze({
    path,
    validate: (value) => (DTYPE[value] ? true : `one of ${Object.keys(DTYPE).join(', ')}`),
  });
}

function layerGroup(label, indices) {
  return { label, count: indices.length, indices };
}

function calculateLayout({ inputs, workload: { batch, seq, sequenceLengths } }) {
  const {
    layerCount,
    layerTypes,
    kvHeadCount,
    headDim,
    linearKeyHeadCount,
    linearValueHeadCount,
    linearKeyHeadDim,
    linearValueHeadDim,
    linearConvKernelDim,
    cacheDtype,
    recurrentStateDtype,
    maxContext,
  } = inputs;
  const workload = validateSequenceWorkload({ batch, seq, sequenceLengths, maxContext });
  if (workload.error) return workload;

  const fullAttentionLayers = layerTypes
    .slice(0, layerCount)
    .flatMap((type, layer) => (type === 'full_attention' ? [layer] : []));
  const linearAttentionLayers = layerTypes
    .slice(0, layerCount)
    .flatMap((type, layer) => (type === 'linear_attention' ? [layer] : []));
  const activeSequenceCount = workload.entries.reduce(
    (count, entry) => count + (entry.length > 0 ? entry.count : 0),
    0,
  );
  const cacheType = DTYPE[cacheDtype];
  const recurrentType = DTYPE[recurrentStateDtype];
  const fullLayerGroup = layerGroup('full-attention layers', fullAttentionLayers);
  const linearLayerGroup = layerGroup('linear-attention layers', linearAttentionLayers);
  const fullElements = workload.tokenCount * fullAttentionLayers.length * kvHeadCount * headDim;
  const convWidth = 2 * linearKeyHeadCount * linearKeyHeadDim
    + linearValueHeadCount * linearValueHeadDim;
  const convHistoryLength = linearConvKernelDim - 1;
  const tokenFormula = 'T = B × S or Σ sequence_lengths';
  const activeFormula = 'N = active sequence count';

  return {
    buffers: [
      makeBuffer({
        id: 'full-attention.key',
        label: 'Full-context GQA key',
        layerGroup: fullLayerGroup,
        elements: fullElements,
        dtype: cacheType.semantic,
        bytesPerElement: cacheType.bytes,
        formula: `T × ${fullAttentionLayers.length} × ${kvHeadCount} × ${headDim}; ${tokenFormula}`,
      }),
      makeBuffer({
        id: 'full-attention.value',
        label: 'Full-context GQA value',
        layerGroup: fullLayerGroup,
        elements: fullElements,
        dtype: cacheType.semantic,
        bytesPerElement: cacheType.bytes,
        formula: `T × ${fullAttentionLayers.length} × ${kvHeadCount} × ${headDim}; ${tokenFormula}`,
      }),
      makeBuffer({
        id: 'linear-attention.conv-state',
        label: 'Gated DeltaNet causal-convolution state',
        layerGroup: linearLayerGroup,
        elements: activeSequenceCount
          * linearAttentionLayers.length
          * convWidth
          * convHistoryLength,
        dtype: cacheType.semantic,
        bytesPerElement: cacheType.bytes,
        formula: `N × ${linearAttentionLayers.length} × (${2 * linearKeyHeadCount * linearKeyHeadDim} + ${linearValueHeadCount * linearValueHeadDim}) × (${linearConvKernelDim} − 1); ${activeFormula}`,
      }),
      makeBuffer({
        id: 'linear-attention.recurrent-state',
        label: 'Gated DeltaNet recurrent key-value state',
        layerGroup: linearLayerGroup,
        elements: activeSequenceCount
          * linearAttentionLayers.length
          * linearValueHeadCount
          * linearKeyHeadDim
          * linearValueHeadDim,
        dtype: recurrentType.semantic,
        bytesPerElement: recurrentType.bytes,
        formula: `N × ${linearAttentionLayers.length} × ${linearValueHeadCount} × ${linearKeyHeadDim} × ${linearValueHeadDim}; ${activeFormula}`,
      }),
    ],
    note: 'The recurrent state follows mamba_ssm_dtype; causal convolution retains K−1 historical projections. Vision encoding and optional MTP are excluded, while visual tokens count toward text sequence length.',
  };
}

export default defineArchitectureProfile({
  id: 'qwen3.6-35b-a3b-semantic-v1',
  version: '1.0.0',
  label: 'Qwen 3.6 35B A3B',
  layout: { id: 'qwen3.6-hybrid-gdn-gqa-v1', version: '1.0.0' },
  repositories: [{
    repoId: 'Qwen/Qwen3.6-35B-A3B',
    auditedCommitId: '995ad96eacd98c81ed38be0c5b274b04031597b0',
    baselineInputs: {
      layerCount: 40,
      layerTypes: LAYER_TYPES,
      kvHeadCount: 2,
      headDim: 256,
      linearKeyHeadCount: 16,
      linearValueHeadCount: 32,
      linearKeyHeadDim: 128,
      linearValueHeadDim: 128,
      linearConvKernelDim: 4,
      cacheDtype: 'bfloat16',
      recurrentStateDtype: 'float32',
      maxContext: 262144,
    },
  }],
  configInputs: {
    layerCount: profileConfigInput.positiveInteger('text_config.num_hidden_layers'),
    layerTypes: profileConfigInput.array('text_config.layer_types'),
    kvHeadCount: profileConfigInput.positiveInteger('text_config.num_key_value_heads'),
    headDim: profileConfigInput.positiveInteger('text_config.head_dim'),
    linearKeyHeadCount: profileConfigInput.positiveInteger('text_config.linear_num_key_heads'),
    linearValueHeadCount: profileConfigInput.positiveInteger('text_config.linear_num_value_heads'),
    linearKeyHeadDim: profileConfigInput.positiveInteger('text_config.linear_key_head_dim'),
    linearValueHeadDim: profileConfigInput.positiveInteger('text_config.linear_value_head_dim'),
    linearConvKernelDim: profileConfigInput.positiveInteger('text_config.linear_conv_kernel_dim'),
    cacheDtype: dtypeInput('text_config.dtype'),
    recurrentStateDtype: dtypeInput('text_config.mamba_ssm_dtype'),
    maxContext: profileConfigInput.positiveInteger('text_config.max_position_embeddings'),
  },
  validateInputs(inputs) {
    if (inputs.layerTypes.length < inputs.layerCount) {
      return [{ input: 'layerTypes', code: 'invalid_length' }];
    }
    if (inputs.layerTypes.slice(0, inputs.layerCount).some(
      (type) => !['full_attention', 'linear_attention'].includes(type),
    )) {
      return [{ input: 'layerTypes', code: 'unsupported_type' }];
    }
    if (inputs.linearValueHeadCount % inputs.linearKeyHeadCount !== 0) {
      return [{ input: 'linearValueHeadCount', code: 'invalid_head_ratio' }];
    }
    return [];
  },
  calculateLayout,
});
