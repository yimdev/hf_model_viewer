import test from 'node:test';
import assert from 'node:assert/strict';

import { createApplicationSession } from '../../src/app/session.js';

test('application session owns loading, derivation, workload, and ready state', async () => {
  const phases = [];
  const ingest = async (repoId, { onShard }) => {
    onShard(1, 1, 'model.safetensors');
    return {
      repoId,
      commitId: '1111111111111111111111111111111111111111',
      config: {
        architectures: ['UnknownForCausalLM'],
        max_position_embeddings: 4096,
      },
      tensors: [
        { name: 'model.layers.0.weight', shape: [2, 3], dtype: 'BF16' },
      ],
      shardFiles: ['model.safetensors'],
      shardCount: 1,
    };
  };
  const session = createApplicationSession({ ingest });
  session.subscribe((snapshot) => phases.push(snapshot.phase));

  const loaded = await session.load('org/model');

  assert.equal(loaded.phase, 'ready');
  assert.equal(loaded.workload.seq, 4096);
  assert.equal(loaded.model.tensorMetadataIndex.totalWeightBytes, 12);
  assert.equal(loaded.estimate.vWeights, 12 / (1024 ** 3));
  assert.equal(loaded.estimate.vTotal, null);
  assert.ok(phases.includes('loading'));

  const updated = session.setWorkload({ batch: 2, seq: 1024 });
  assert.deepEqual(updated.workload, { batch: 2, seq: 1024 });
  assert.equal(updated.phase, 'ready');
});

test('application session preserves a structured ingestion failure', async () => {
  const failure = Object.assign(new Error('invalid_repo_id'), {
    code: 'invalid_repo_id',
    details: { repoId: 'bad' },
  });
  const session = createApplicationSession({ ingest: async () => { throw failure; } });

  const snapshot = await session.load('bad');

  assert.equal(snapshot.phase, 'error');
  assert.equal(snapshot.error, failure);
  assert.equal(snapshot.model, null);
});
