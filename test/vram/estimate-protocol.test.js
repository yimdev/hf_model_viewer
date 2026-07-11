import test from 'node:test';
import assert from 'node:assert/strict';

import { estimateVRAM } from '../../src/vram/index.js';
import { glm52Fixture } from './profile-fixtures.js';

test('Complete VRAM Estimate exposes calculation, assurance, and provenance', () => {
  const fixture = glm52Fixture();

  const result = estimateVRAM({
    source: {
      repoId: 'zai-org/GLM-5.2',
      commitId: 'b4734de4facf877f85769a911abafc5283eab3d9',
    },
    config: fixture.config,
    tensors: fixture.tensors,
    workload: { batch: 1, seq: 1 },
  });

  assert.equal(result.complete, true);
  assert.equal(result.calculation.status, 'computed');
  assert.equal(result.assurance.status, 'verified');
  assert.equal(result.provenance.repoId, 'zai-org/GLM-5.2');
  assert.equal(result.vKV, 95_232 / (1024 ** 3));
  assert.equal(result.vTotal, result.vWeights + result.vKV);
  assert.equal('kvStatus' in result, false);
  assert.equal('kvUnknown' in result, false);
  assert.equal('kvDiagnostic' in result, false);
});
