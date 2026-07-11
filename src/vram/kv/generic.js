import { makeBuffer, validateSequenceWorkload } from './profile-primitives.js';
import {
  firstDefined,
  maxContextLength,
  modelDefault,
  textModelConfig,
} from './config-access.js';

const COMMIT_ID_RE = /^[0-9a-f]{40}$/;
const REPO_ID_RE = /^[\w.-]+\/[\w.-]+$/;
const GB = 1024 ** 3;
const APPROXIMATION = Object.freeze({
  id: 'generic-mha-gqa-v1',
  version: '1.0.0',
  label: 'Generic MHA/GQA KV Cache Estimate',
});

const DTYPE_ALIASES = Object.freeze({
  BF16: { dtype: 'BF16', bytes: 2 },
  bfloat16: { dtype: 'BF16', bytes: 2 },
  F16: { dtype: 'F16', bytes: 2 },
  float16: { dtype: 'F16', bytes: 2 },
  half: { dtype: 'F16', bytes: 2 },
  F32: { dtype: 'F32', bytes: 4 },
  float32: { dtype: 'F32', bytes: 4 },
  F64: { dtype: 'F64', bytes: 8 },
  float64: { dtype: 'F64', bytes: 8 },
  F8_E4M3FN: { dtype: 'F8', bytes: 1 },
  float8_e4m3fn: { dtype: 'F8', bytes: 1 },
  fp8: { dtype: 'F8', bytes: 1 },
  INT8: { dtype: 'INT8', bytes: 1 },
  int8: { dtype: 'INT8', bytes: 1 },
});

function positiveInteger(value) {
  return Number.isInteger(value) && value > 0;
}

function unknownResult({ source, code, details = null, assumptions = [] }) {
  return Object.freeze({
    calculation: Object.freeze({
      status: 'unknown',
      diagnostic: Object.freeze({ code, ...(details ? { details } : {}) }),
    }),
    assurance: null,
    provenance: Object.freeze({
      repoId: source.repoId,
      commitId: source.commitId,
      auditedCommitId: null,
    }),
    profile: null,
    approximation: Object.freeze({ ...APPROXIMATION, assumptions: Object.freeze(assumptions) }),
    buffers: Object.freeze([]),
    totalBytes: null,
    vKV: null,
    note: '',
  });
}

export function computeGenericKV({
  source = {}, config = {}, batch = 1, seq = 8192, sequenceLengths = null,
} = {}) {
  if (!REPO_ID_RE.test(source.repoId || '') || !COMMIT_ID_RE.test(source.commitId || '')) {
    return unknownResult({ source, code: 'invalid_model_provenance' });
  }

  const modelConfig = textModelConfig(config);
  const layerCount = firstDefined(modelConfig, ['num_hidden_layers', 'n_layer', 'num_layers']);
  const attentionHeadCount = firstDefined(
    modelConfig,
    ['num_attention_heads', 'n_head', 'num_heads'],
  );
  const configuredKVHeadCount = firstDefined(
    modelConfig,
    ['num_key_value_heads', 'num_kv_heads', 'multi_query_group_num'],
  );
  const kvHeadCount = configuredKVHeadCount ?? attentionHeadCount;
  const hiddenSize = firstDefined(modelConfig, ['hidden_size', 'n_embd', 'd_model']);
  const configuredHeadDim = firstDefined(
    modelConfig,
    ['head_dim', 'attention_head_dim'],
  );
  const headDim = configuredHeadDim
    ?? (positiveInteger(hiddenSize) && positiveInteger(attentionHeadCount)
      ? hiddenSize / attentionHeadCount
      : undefined);
  const issues = [
    ['layerCount', layerCount],
    ['kvHeadCount', kvHeadCount],
    ['headDim', headDim],
  ].flatMap(([input, value]) => (
    positiveInteger(value) ? [] : [{ input, code: value == null ? 'missing' : 'invalid', value }]
  ));
  if (issues.length > 0) {
    return unknownResult({
      source,
      code: 'generic_config_insufficient',
      details: { issues: Object.freeze(issues) },
    });
  }

  const configuredMaxContext = maxContextLength(config);
  const maxContext = configuredMaxContext ?? Number.MAX_SAFE_INTEGER;
  const workload = validateSequenceWorkload({ batch, seq, sequenceLengths, maxContext });
  if (workload.error) {
    return unknownResult({
      source,
      code: workload.error,
      details: workload.details,
    });
  }

  const configuredCacheDtype = modelDefault(config, ['kv_cache_dtype', 'cache_dtype']);
  const configuredModelDtype = modelDefault(config, ['torch_dtype', 'dtype']);
  const configuredDtype = configuredCacheDtype ?? configuredModelDtype;
  const dtype = DTYPE_ALIASES[configuredDtype] || DTYPE_ALIASES.BF16;
  const assumptions = [
    Object.freeze({ code: 'uniform_full_context_attention' }),
    Object.freeze({ code: 'standard_key_value_buffers' }),
  ];
  if (configuredKVHeadCount == null) {
    assumptions.push(Object.freeze({
      code: 'kv_heads_defaulted_to_attention_heads',
      assumedKVHeadCount: kvHeadCount,
    }));
  }
  if (configuredHeadDim == null) {
    assumptions.push(Object.freeze({
      code: 'head_dim_derived',
      hiddenSize,
      attentionHeadCount,
      assumedHeadDim: headDim,
    }));
  }
  if (!DTYPE_ALIASES[configuredDtype]) {
    assumptions.push(Object.freeze({
      code: 'cache_dtype_defaulted',
      configuredValue: configuredDtype ?? null,
      assumedDtype: dtype.dtype,
    }));
  } else if (configuredCacheDtype == null) {
    assumptions.push(Object.freeze({
      code: 'cache_dtype_from_model_dtype',
      configuredValue: configuredDtype,
      assumedDtype: dtype.dtype,
    }));
  }
  if (configuredMaxContext == null) {
    assumptions.push(Object.freeze({ code: 'max_context_unavailable' }));
  }

  const elementsPerBuffer = workload.tokenCount * layerCount * kvHeadCount * headDim;
  const tokenFormula = 'T = B × S or Σ sequence_lengths';
  let buffers;
  try {
    buffers = Object.freeze([
      makeBuffer({
        id: 'generic.key',
        label: 'Approximate full-context key cache',
        layerGroup: { label: 'all configured layers', count: layerCount, range: [0, layerCount - 1] },
        elements: elementsPerBuffer,
        dtype: dtype.dtype,
        bytesPerElement: dtype.bytes,
        formula: `T × ${layerCount} × ${kvHeadCount} × ${headDim}; ${tokenFormula}`,
      }),
      makeBuffer({
        id: 'generic.value',
        label: 'Approximate full-context value cache',
        layerGroup: { label: 'all configured layers', count: layerCount, range: [0, layerCount - 1] },
        elements: elementsPerBuffer,
        dtype: dtype.dtype,
        bytesPerElement: dtype.bytes,
        formula: `T × ${layerCount} × ${kvHeadCount} × ${headDim}; ${tokenFormula}`,
      }),
    ]);
  } catch (error) {
    return unknownResult({
      source,
      code: 'profile_calculation_out_of_range',
      details: { message: error.message },
      assumptions,
    });
  }
  const totalBytes = buffers.reduce((sum, buffer) => sum + buffer.bytes, 0);
  if (!Number.isSafeInteger(totalBytes) || totalBytes < 0) {
    return unknownResult({
      source,
      code: 'profile_calculation_out_of_range',
      assumptions,
    });
  }
  return Object.freeze({
    calculation: Object.freeze({ status: 'computed', diagnostic: null }),
    assurance: Object.freeze({
      status: 'approximate',
      warnings: Object.freeze([Object.freeze({ code: 'generic_fallback' })]),
    }),
    provenance: Object.freeze({
      repoId: source.repoId,
      commitId: source.commitId,
      auditedCommitId: null,
    }),
    profile: null,
    approximation: Object.freeze({ ...APPROXIMATION, assumptions: Object.freeze(assumptions) }),
    buffers,
    totalBytes,
    vKV: totalBytes / GB,
    note: 'Approximate standard MHA/GQA KV Cache; model-specific cache semantics may differ.',
  });
}
