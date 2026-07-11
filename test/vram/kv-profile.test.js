import test from 'node:test';
import assert from 'node:assert/strict';

import { computeKV } from '../../src/vram/kv/index.js';
import { makeBuffer } from '../../src/vram/kv/profile-primitives.js';
import { deepseekV4ProFixture, glm52Fixture, hy3Fixture } from './profile-fixtures.js';

const SOURCES = Object.freeze({
  glm52: {
    repoId: 'zai-org/GLM-5.2',
    commitId: 'b4734de4facf877f85769a911abafc5283eab3d9',
  },
  glm52Fp8: {
    repoId: 'zai-org/GLM-5.2-FP8',
    commitId: 'ba978f7d347eaf65d22f1a86833408afdb953541',
  },
  deepseek: {
    repoId: 'deepseek-ai/DeepSeek-V4-Pro',
    commitId: 'b5968e9190ef611bbf34a7229255be88a0e937c1',
  },
  hy3: {
    repoId: 'tencent/Hy3',
    commitId: '716aa7241bd6d95896be4ebfc761162a9c4d49ef',
  },
  hy3Preview: {
    repoId: 'tencent/Hy3-preview',
    commitId: 'b53bd705bef15f0a9e52eade60a4353eaaa6c6b8',
  },
});

function calculate(source, fixture, workload = {}) {
  return computeKV({ source, config: fixture.config, ...workload });
}

test('unsupported Model Repository Identifier uses an approximate standard GQA fallback', () => {
  const result = computeKV({
    source: {
      repoId: 'unknown/model',
      commitId: '1111111111111111111111111111111111111111',
    },
    config: {
      num_hidden_layers: 2,
      hidden_size: 1024,
      num_attention_heads: 8,
      num_key_value_heads: 2,
      torch_dtype: 'bfloat16',
      max_position_embeddings: 4096,
    },
    batch: 1,
    seq: 10,
  });

  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.assurance.status, 'approximate');
  assert.equal(result.profile, null);
  assert.equal(result.approximation.id, 'generic-mha-gqa-v1');
  assert.equal(result.totalBytes, 20_480);
  assert.deepEqual(
    result.approximation.assumptions.map((assumption) => assumption.code),
    [
      'uniform_full_context_attention',
      'standard_key_value_buffers',
      'head_dim_derived',
      'cache_dtype_from_model_dtype',
    ],
  );
  assert.deepEqual(
    result.buffers.map(({ id, elements, bytes, dtype }) => ({ id, elements, bytes, dtype })),
    [
      { id: 'generic.key', elements: 5120, bytes: 10_240, dtype: 'BF16' },
      { id: 'generic.value', elements: 5120, bytes: 10_240, dtype: 'BF16' },
    ],
  );
});

test('generic fallback only requires dimensions that are not already explicit', () => {
  const result = computeKV({
    source: {
      repoId: 'unknown/explicit-dimensions',
      commitId: '1111111111111111111111111111111111111111',
    },
    config: {
      num_hidden_layers: 1,
      num_key_value_heads: 2,
      head_dim: 64,
    },
    batch: 1,
    seq: 1,
  });

  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.totalBytes, 512);
  assert.equal(result.buffers[0].dtype, 'BF16');
  assert.ok(
    result.approximation.assumptions.some((assumption) => (
      assumption.code === 'cache_dtype_defaulted'
    )),
  );
});

test('generic fallback combines nested text dimensions with root model defaults', () => {
  const result = computeKV({
    source: {
      repoId: 'unknown/nested-config',
      commitId: '1111111111111111111111111111111111111111',
    },
    config: {
      torch_dtype: 'float16',
      max_position_embeddings: 4096,
      text_config: {
        num_hidden_layers: 1,
        num_attention_heads: 4,
        hidden_size: 256,
      },
    },
    batch: 1,
    seq: 1,
  });

  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.totalBytes, 1024);
  assert.equal(result.buffers[0].dtype, 'F16');
  assert.ok(
    result.approximation.assumptions.some((assumption) => (
      assumption.code === 'cache_dtype_from_model_dtype'
    )),
  );
  assert.ok(
    result.approximation.assumptions.every((assumption) => (
      assumption.code !== 'max_context_unavailable'
    )),
  );
});

test('explicit cache dtype outranks model dtype across config scopes', () => {
  const result = computeKV({
    source: {
      repoId: 'unknown/cache-dtype-precedence',
      commitId: '1111111111111111111111111111111111111111',
    },
    config: {
      kv_cache_dtype: 'fp8',
      text_config: {
        num_hidden_layers: 1,
        num_attention_heads: 4,
        hidden_size: 256,
        torch_dtype: 'bfloat16',
      },
    },
    batch: 1,
    seq: 1,
  });

  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.totalBytes, 512);
  assert.equal(result.buffers[0].dtype, 'F8');
  assert.ok(
    result.approximation.assumptions.every((assumption) => (
      !['cache_dtype_defaulted', 'cache_dtype_from_model_dtype'].includes(assumption.code)
    )),
  );
});

test('supported repository requires an immutable commit ID', () => {
  const result = computeKV({
    source: { repoId: SOURCES.hy3.repoId, commitId: 'main' },
    config: hy3Fixture().config,
  });

  assert.equal(result.calculation.status, 'unknown');
  assert.equal(result.calculation.diagnostic.code, 'invalid_model_provenance');
});

test('generic fallback requires a canonical repository ID', () => {
  const result = computeKV({
    source: { commitId: '1111111111111111111111111111111111111111' },
    config: {
      num_hidden_layers: 1,
      num_key_value_heads: 1,
      head_dim: 64,
    },
    batch: 1,
    seq: 1,
  });

  assert.equal(result.calculation.status, 'unknown');
  assert.equal(result.calculation.diagnostic.code, 'invalid_model_provenance');
});

test('GLM 5.2 returns its IndexShare buffer breakdown', () => {
  const result = calculate(SOURCES.glm52, glm52Fixture(), { batch: 1, seq: 1 });

  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.assurance.status, 'verified');
  assert.equal(result.profile.id, 'glm-5.2-semantic-bf16-v1');
  assert.equal(result.profile.layout.id, 'glm-5.2-indexshare-bf16-v1');
  assert.equal(result.totalBytes, 95_232);
  assert.deepEqual(
    result.buffers.map(({ id, bytes, dtype }) => ({ id, bytes, dtype })),
    [
      { id: 'mla-latent', bytes: 79_872, dtype: 'BF16' },
      { id: 'mla-rope-key', bytes: 9_984, dtype: 'BF16' },
      { id: 'indexer-key', bytes: 5_376, dtype: 'BF16' },
    ],
  );
});

test('GLM 5.2 FP8 repository alias uses the same KV Cache Layout', () => {
  const result = calculate(SOURCES.glm52Fp8, glm52Fixture({ fp8: true }), { batch: 1, seq: 1 });

  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.assurance.status, 'verified');
  assert.equal(result.totalBytes, 95_232);
});

test('DeepSeek V4 Pro returns its HCA and CSA stateful buffer breakdown', () => {
  const result = calculate(SOURCES.deepseek, deepseekV4ProFixture(), { batch: 1, seq: 128 });

  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.assurance.status, 'verified');
  assert.equal(result.totalBytes, 10_484_736);
  assert.deepEqual(
    result.buffers.map(({ id, bytes, dtype }) => ({ id, bytes, dtype })),
    [
      { id: 'hca-kv', bytes: 4_094_976, dtype: 'BF16' },
      { id: 'hca-kv-state', bytes: 0, dtype: 'F32' },
      { id: 'hca-score-state', bytes: 0, dtype: 'F32' },
      { id: 'csa-kv', bytes: 4_915_200, dtype: 'BF16' },
      { id: 'csa-indexer-kv', bytes: 245_760, dtype: 'BF16' },
      { id: 'csa-kv-state', bytes: 491_520, dtype: 'F32' },
      { id: 'csa-score-state', bytes: 491_520, dtype: 'F32' },
      { id: 'csa-indexer-kv-state', bytes: 122_880, dtype: 'F32' },
      { id: 'csa-indexer-score-state', bytes: 122_880, dtype: 'F32' },
    ],
  );
});

test('Hunyuan 3 returns its full-context GQA buffer breakdown', () => {
  const result = calculate(SOURCES.hy3, hy3Fixture(), { batch: 1, seq: 1 });

  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.assurance.status, 'verified');
  assert.equal(result.totalBytes, 327_680);
  assert.deepEqual(
    result.buffers.map(({ id, bytes, dtype }) => ({ id, bytes, dtype })),
    [
      { id: 'main.key', bytes: 163_840, dtype: 'BF16' },
      { id: 'main.value', bytes: 163_840, dtype: 'BF16' },
    ],
  );
});

test('Hunyuan 3 preview repository alias is independently audited', () => {
  const result = calculate(SOURCES.hy3Preview, hy3Fixture(), { batch: 1, seq: 1 });

  assert.equal(result.assurance.status, 'verified');
  assert.equal(result.provenance.auditedCommitId, SOURCES.hy3Preview.commitId);
  assert.equal(result.totalBytes, 327_680);
});

test('current config max context controls workload validation', () => {
  const fixture = glm52Fixture();
  fixture.config.max_position_embeddings = 4096;
  const atMax = calculate(SOURCES.glm52, fixture, { batch: 1, seq: 4096 });
  const beyondMax = calculate(SOURCES.glm52, fixture, { batch: 1, seq: 4097 });

  assert.equal(atMax.calculation.status, 'computed');
  assert.equal(atMax.assurance.status, 'warning');
  assert.equal(beyondMax.calculation.status, 'unknown');
  assert.equal(beyondMax.calculation.diagnostic.code, 'profile_input_out_of_range');
});

test('DeepSeek V4 Pro compressor state transitions preserve golden totals', () => {
  const fixture = deepseekV4ProFixture();
  const vectors = new Map([
    [3, 1_489_920],
    [4, 2_024_960],
    [7, 3_514_880],
    [8, 2_821_120],
    [127, 27_399_680],
    [128, 10_484_736],
    [129, 10_918_912],
    [1048576, 10_335_600_640],
  ]);

  for (const [seq, expectedBytes] of vectors) {
    assert.equal(calculate(SOURCES.deepseek, fixture, { batch: 1, seq }).totalBytes, expectedBytes);
  }
});

test('ragged workloads use current Layout semantics', () => {
  const glm = calculate(SOURCES.glm52, glm52Fixture(), { sequenceLengths: [1, 2048, 2049] });
  const hy3 = calculate(SOURCES.hy3, hy3Fixture(), { sequenceLengths: [1, 4096, 8192] });
  const deepseekFixture = deepseekV4ProFixture();
  const lengths = [3, 4, 7, 128, 129];
  const deepseek = calculate(SOURCES.deepseek, deepseekFixture, { sequenceLengths: lengths });
  const independentlySummed = lengths.reduce(
    (sum, seq) => sum + calculate(SOURCES.deepseek, deepseekFixture, { batch: 1, seq }).totalBytes,
    0,
  );

  assert.equal(glm.totalBytes, (1 + 2048 + 2049) * 95_232);
  assert.equal(hy3.totalBytes, 4_026_859_520);
  assert.equal(deepseek.totalBytes, independentlySummed);
});

test('Profile workload semantics preserve zero-workload differences', () => {
  for (const [source, fixture] of [
    [SOURCES.glm52, glm52Fixture()],
    [SOURCES.hy3, hy3Fixture()],
  ]) {
    assert.equal(calculate(source, fixture, { batch: 0, seq: 128 }).totalBytes, 0);
    assert.equal(calculate(source, fixture, { sequenceLengths: [] }).totalBytes, 0);
  }
  const deepseek = calculate(SOURCES.deepseek, deepseekV4ProFixture(), { batch: 0, seq: 128 });
  assert.equal(deepseek.calculation.status, 'unknown');
  assert.equal(deepseek.calculation.diagnostic.code, 'profile_input_out_of_range');
});

test('safe-integer validation keeps impossible calculations unknown', () => {
  const result = calculate(SOURCES.hy3, hy3Fixture(), {
    batch: Number.MAX_SAFE_INTEGER,
    seq: 262144,
  });

  assert.equal(result.calculation.status, 'unknown');
  assert.equal(result.calculation.diagnostic.code, 'profile_calculation_out_of_range');
});

test('generic fallback rejects an unsafe sum even when each buffer is safe', () => {
  const result = computeKV({
    source: {
      repoId: 'unknown/unsafe-total',
      commitId: '1111111111111111111111111111111111111111',
    },
    config: {
      num_hidden_layers: 1,
      num_key_value_heads: 1,
      head_dim: 1,
      kv_cache_dtype: 'fp8',
    },
    batch: 5_000_000_000_000_000,
    seq: 1,
  });

  assert.equal(result.calculation.status, 'unknown');
  assert.equal(result.calculation.diagnostic.code, 'profile_calculation_out_of_range');
});

test('buffer validation rejects a DType and byte-width mismatch', () => {
  assert.throws(
    () => makeBuffer({
      id: 'bad-bf16', label: 'bad', layerGroup: {}, elements: 1,
      dtype: 'BF16', bytesPerElement: 1, formula: 'invalid',
    }),
    /Dtype width mismatch/,
  );
});
