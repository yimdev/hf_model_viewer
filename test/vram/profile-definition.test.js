import test from 'node:test';
import assert from 'node:assert/strict';

import { createArchitectureLayoutCatalog } from '../../src/vram/kv/catalog.js';
import { defineArchitectureProfile, profileConfigInput } from '../../src/vram/kv/profile-definition.js';

function profileDefinition(overrides = {}) {
  return {
    id: 'test-profile',
    version: '1.0.0',
    label: 'Test Profile',
    layout: { id: 'test-layout', version: '1.0.0' },
    repositories: [{
      repoId: 'org/model',
      auditedCommitId: '1111111111111111111111111111111111111111',
      baselineInputs: { layerCount: 1 },
    }],
    configInputs: {
      layerCount: profileConfigInput.positiveInteger('num_hidden_layers'),
    },
    calculateLayout: () => ({ buffers: [] }),
    ...overrides,
  };
}

test('Architecture Layout Catalog rejects duplicate repository registrations', () => {
  const first = defineArchitectureProfile(profileDefinition());
  const second = defineArchitectureProfile(profileDefinition({ id: 'another-profile' }));

  assert.throws(
    () => createArchitectureLayoutCatalog([first, second]),
    /Duplicate Architecture Profile repoId: org\/model/,
  );
});

test('Architecture Profile definition requires a complete valid audited baseline', () => {
  assert.throws(
    () => defineArchitectureProfile(profileDefinition({
      repositories: [{
        repoId: 'org/model',
        auditedCommitId: 'main',
        baselineInputs: {},
      }],
    })),
    /invalid audited commit/,
  );
  assert.throws(
    () => defineArchitectureProfile(profileDefinition({
      repositories: [{
        repoId: 'org/model',
        auditedCommitId: '1111111111111111111111111111111111111111',
        baselineInputs: {},
      }],
    })),
    /baseline inputs do not match config inputs/,
  );
});

test('Architecture Profile definition rejects a baseline that violates Layout invariants', () => {
  assert.throws(
    () => defineArchitectureProfile(profileDefinition({
      validateInputs: () => [{ input: 'layerCount', code: 'unsupported' }],
    })),
    /baseline inputs violate Layout invariants/,
  );
});

test('Architecture Profile definition validates identity, layout, repository, and config paths', () => {
  for (const overrides of [
    { id: '' },
    { version: 'v1' },
    { label: '' },
    { layout: null },
    { layout: { id: '', version: '1.0.0' } },
    { layout: { id: 'layout', version: 'v1' } },
    { repositories: [{
      repoId: 'not canonical',
      auditedCommitId: '1111111111111111111111111111111111111111',
      baselineInputs: { layerCount: 1 },
    }] },
    { configInputs: { layerCount: profileConfigInput.positiveInteger(undefined) } },
    { configInputs: { layerCount: { path: 'num_hidden_layers' } } },
  ]) {
    assert.throws(() => defineArchitectureProfile(profileDefinition(overrides)));
  }
});

test('Architecture Profile definition owns an immutable copy of audited inputs', () => {
  const auditedValues = [1, 2];
  const profile = defineArchitectureProfile(profileDefinition({
    repositories: [{
      repoId: 'org/model',
      auditedCommitId: '1111111111111111111111111111111111111111',
      baselineInputs: { values: auditedValues },
    }],
    configInputs: { values: profileConfigInput.array('values') },
  }));

  auditedValues.push(3);

  assert.deepEqual(profile.repositories[0].baselineInputs.values, [1, 2]);
  assert.throws(() => profile.repositories[0].baselineInputs.values.push(4));
});
