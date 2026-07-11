import test from 'node:test';
import assert from 'node:assert/strict';

import { renderKVDetails } from '../../src/ui/kvDetailsView.js';

test('KV audit adapter renders provenance and truncates structured config differences', () => {
  const differences = Array.from({ length: 13 }, (_, index) => ({
    input: `input${index}`,
    configPath: `field${index}`,
    auditedValue: index,
    currentValue: index + 1,
  }));
  const container = { innerHTML: '' };

  renderKVDetails(container, {
    calculation: { status: 'computed', diagnostic: null },
    assurance: {
      status: 'warning',
      warnings: [{ code: 'config_mismatch', differences }],
    },
    provenance: {
      repoId: 'org/model',
      commitId: '1111111111111111111111111111111111111111',
      auditedCommitId: '2222222222222222222222222222222222222222',
    },
    profile: {
      id: 'profile', version: '1.0.0', label: 'Profile',
      layout: { id: 'layout', version: '1.0.0' },
    },
    buffers: [{
      id: 'kv', label: 'KV', layerGroup: {}, elements: 1,
      dtype: 'BF16', bytesPerElement: 2, bytes: 2, gb: 2 / (1024 ** 3), formula: '1 × 2',
    }],
    vKV: 2 / (1024 ** 3),
  });

  assert.match(container.innerHTML, /Repository: <code>org\/model<\/code>/);
  assert.match(container.innerHTML, /Current commit: <code>1111111111111111111111111111111111111111<\/code>/);
  assert.match(container.innerHTML, /Calculated from audited commit: <code>2222222222222222222222222222222222222222<\/code>/);
  assert.match(container.innerHTML, /Unverified result/);
  assert.match(container.innerHTML, /field11/);
  assert.doesNotMatch(container.innerHTML, /field12/);
  assert.match(container.innerHTML, /<div>\+1<\/div>/);
  assert.doesNotMatch(container.innerHTML, /First-party evidence/);
});
