import test from 'node:test';
import assert from 'node:assert/strict';

import { estimateVRAM } from '../../src/vram/estimate.js';
import { glm52Fixture } from './profile-fixtures.js';

const GLM_SOURCE = Object.freeze({
  repoId: 'zai-org/GLM-5.2',
  commitId: 'b4734de4facf877f85769a911abafc5283eab3d9',
});

test('Complete VRAM Estimate stays unknown when generic config dimensions are insufficient', () => {
  const result = estimateVRAM({
    source: {
      repoId: 'unknown/model',
      commitId: '1111111111111111111111111111111111111111',
    },
    config: {},
    tensors: [{ name: 'model.embed_tokens.weight', shape: [10, 10], dtype: 'BF16' }],
    workload: { batch: 1, seq: 1024 },
  });

  assert.equal(result.complete, false);
  assert.equal(result.calculation.status, 'unknown');
  assert.equal(result.calculation.diagnostic.code, 'generic_config_insufficient');
  assert.equal(result.vKV, null);
  assert.equal(result.vTotal, null);
  assert.equal(result.breakdown.kvGB, null);
  assert.equal(result.composition.some((item) => item.key === 'kv'), false);
  assert.deepEqual(result.buffers, []);
});

test('Generic KV Cache Estimate can complete total VRAM with approximate assurance', () => {
  const result = estimateVRAM({
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
    },
    tensors: [{ name: 'model.weight', shape: [10, 10], dtype: 'BF16' }],
    workload: { batch: 1, seq: 10 },
  });

  assert.equal(result.complete, true);
  assert.equal(result.assurance.status, 'approximate');
  assert.equal(result.approximation.id, 'generic-mha-gqa-v1');
  assert.equal(result.vKV, 20_480 / (1024 ** 3));
  assert.equal(result.vTotal, 20_680 / (1024 ** 3));
});

test('warning Calculation Assurance still permits a Complete VRAM Estimate', () => {
  const fixture = glm52Fixture();
  const result = estimateVRAM({
    source: { ...GLM_SOURCE, commitId: '1111111111111111111111111111111111111111' },
    config: fixture.config,
    tensors: fixture.tensors,
    workload: { batch: 1, seq: 1 },
  });

  assert.equal(result.complete, true);
  assert.equal(result.assurance.status, 'warning');
  assert.equal(result.vKV, 95_232 / (1024 ** 3));
  assert.equal(result.vTotal, result.vWeights + result.vKV);
  assert.equal(result.buffers.reduce((sum, buffer) => sum + buffer.bytes, 0), 95_232);
  assert.deepEqual(result.composition.find((item) => item.key === 'kv').dtypes, ['BF16']);
});

test('Complete VRAM Estimate forwards ragged sequence lengths', () => {
  const fixture = glm52Fixture();
  const result = estimateVRAM({
    source: GLM_SOURCE,
    config: fixture.config,
    tensors: fixture.tensors,
    workload: { sequenceLengths: [1, 2, 3] },
  });

  assert.equal(result.complete, true);
  assert.equal(result.buffers.reduce((sum, buffer) => sum + buffer.bytes, 0), 6 * 95_232);
});

test('weight composition merges Tensor Name Patterns and sorts them by size', () => {
  const result = estimateVRAM({
    source: {
      repoId: 'unknown/model',
      commitId: '1111111111111111111111111111111111111111',
    },
    config: {},
    tensors: [
      {
        name: 'language_model.model.layers.0.self_attn.k_norm.weight',
        shape: [2],
        dtype: 'BF16',
      },
      {
        name: 'language_model.model.layers.1.self_attn.k_norm.weight',
        shape: [3],
        dtype: 'F32',
      },
      {
        name: 'language_model.model.layers.0.self_attn.q_proj.weight',
        shape: [4],
        dtype: 'BF16',
      },
    ],
  });

  const weights = result.composition.filter((item) => item.group === 'weight');
  assert.deepEqual(
    weights.map((item) => [item.label, item.dtypes, item.gb]),
    [
      ['language_model.model.layers.*.self_attn.k_norm.weight', ['BF16', 'F32'], 16 / (1024 ** 3)],
      ['language_model.model.layers.*.self_attn.q_proj.weight', ['BF16'], 8 / (1024 ** 3)],
    ],
  );
});
