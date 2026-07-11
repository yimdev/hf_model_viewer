import test from 'node:test';
import assert from 'node:assert/strict';

import { computeKV } from '../../src/vram/kv/index.js';
import { deepseekV4ProFixture } from './profile-fixtures.js';

const DEEPSEEK_REPO_ID = 'deepseek-ai/DeepSeek-V4-Pro';
const DEEPSEEK_AUDITED_COMMIT_ID = 'b5968e9190ef611bbf34a7229255be88a0e937c1';

test('audited repository revision produces a verified Profile calculation', () => {
  const { config } = deepseekV4ProFixture();

  const result = computeKV({
    source: { repoId: DEEPSEEK_REPO_ID, commitId: DEEPSEEK_AUDITED_COMMIT_ID },
    config,
    batch: 1,
    seq: 128,
  });

  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.assurance.status, 'verified');
  assert.deepEqual(result.assurance.warnings, []);
  assert.deepEqual(result.provenance, {
    repoId: DEEPSEEK_REPO_ID,
    commitId: DEEPSEEK_AUDITED_COMMIT_ID,
    auditedCommitId: DEEPSEEK_AUDITED_COMMIT_ID,
  });
  assert.equal(result.totalBytes, 10_484_736);
});

test('valid config drift calculates from current inputs and lowers Profile Assurance', () => {
  const { config } = deepseekV4ProFixture();
  config.num_hidden_layers = 1;
  config.compress_ratios = [128];

  const result = computeKV({
    source: { repoId: DEEPSEEK_REPO_ID, commitId: DEEPSEEK_AUDITED_COMMIT_ID },
    config,
    batch: 1,
    seq: 128,
  });

  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.assurance.status, 'warning');
  assert.equal(result.totalBytes, 132_096);
  assert.deepEqual(result.assurance.warnings.map((warning) => warning.code), [
    'config_mismatch',
  ]);
  assert.deepEqual(
    result.assurance.warnings[0].differences.map((difference) => difference.configPath),
    ['num_hidden_layers', 'compress_ratios'],
  );
});

test('new repository commit calculates with an explicit commit warning', () => {
  const { config } = deepseekV4ProFixture();
  const commitId = '1111111111111111111111111111111111111111';

  const result = computeKV({
    source: { repoId: DEEPSEEK_REPO_ID, commitId },
    config,
    batch: 1,
    seq: 128,
  });

  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.assurance.status, 'warning');
  assert.deepEqual(result.assurance.warnings, [{
    code: 'commit_mismatch',
    currentCommitId: commitId,
    auditedCommitId: DEEPSEEK_AUDITED_COMMIT_ID,
  }]);
});

test('missing algorithm input keeps calculation unknown', () => {
  const { config } = deepseekV4ProFixture();
  delete config.compress_ratios;

  const result = computeKV({
    source: { repoId: DEEPSEEK_REPO_ID, commitId: DEEPSEEK_AUDITED_COMMIT_ID },
    config,
    batch: 1,
    seq: 128,
  });

  assert.equal(result.calculation.status, 'unknown');
  assert.equal(result.calculation.diagnostic.code, 'invalid_profile_config');
  assert.equal(result.totalBytes, null);
  assert.deepEqual(result.calculation.diagnostic.details.issues[0], {
    input: 'compressionByLayer',
    configPath: 'compress_ratios',
    code: 'missing',
  });
});

test('invalid provenance blocks calculation without inventing Profile Assurance', () => {
  const { config } = deepseekV4ProFixture();

  const result = computeKV({
    source: { repoId: DEEPSEEK_REPO_ID, commitId: 'main' },
    config,
  });

  assert.equal(result.calculation.status, 'unknown');
  assert.equal(result.calculation.diagnostic.code, 'invalid_model_provenance');
  assert.equal(result.assurance, null);
});
