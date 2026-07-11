import test from 'node:test';
import assert from 'node:assert/strict';

import { createModelIngestion, IngestionError } from '../../src/engine/index.js';
import { makeMemoryTransport } from '../../src/platform/memoryNet.js';

const COMMIT_ID = '1234567890123456789012345678901234567890';
const PINNED_BASE = `https://huggingface.co/org/model/resolve/${COMMIT_ID}`;

function safetensorsRoutes(fileName, header, base = PINNED_BASE) {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const length = new Uint8Array(8);
  new DataView(length.buffer).setBigUint64(0, BigInt(json.byteLength), true);
  const url = `${base}/${fileName}`;
  return {
    [`${url}#0-7`]: length,
    [`${url}#8-${7 + json.byteLength}`]: json,
  };
}

test('model ingestion resolves canonical provenance and pins every model file', async () => {
  const transport = makeMemoryTransport({
    text: {
      'https://huggingface.co/api/models/org/alias/revision/main': JSON.stringify({
        id: 'org/model',
        sha: COMMIT_ID,
      }),
      [`${PINNED_BASE}/config.json`]: JSON.stringify({
        architectures: ['ExampleForCausalLM'],
      }),
      [`${PINNED_BASE}/model.safetensors.index.json`]: JSON.stringify({
        weight_map: { 'model.weight': 'model.safetensors' },
      }),
    },
    range: safetensorsRoutes('model.safetensors', {
      'model.weight': { dtype: 'BF16', shape: [2, 3], data_offsets: [0, 12] },
    }, PINNED_BASE),
  });
  const ingest = createModelIngestion({ transport });

  const result = await ingest('org/alias');

  assert.equal(result.repoId, 'org/model');
  assert.equal(result.commitId, COMMIT_ID);
  assert.equal(result.shardCount, 1);
});

test('model ingestion runs through the in-memory transport adapter', async () => {
  const transport = makeMemoryTransport({
    text: {
      'https://huggingface.co/api/models/org/model/revision/main': JSON.stringify({
        id: 'org/model', sha: COMMIT_ID,
      }),
      [`${PINNED_BASE}/config.json`]: JSON.stringify({ architectures: ['ExampleForCausalLM'] }),
      [`${PINNED_BASE}/model.safetensors.index.json`]: JSON.stringify({
        metadata: { total_size: 12 },
        weight_map: { 'model.weight': 'model-00001-of-00001.safetensors' },
      }),
    },
    range: safetensorsRoutes('model-00001-of-00001.safetensors', {
      'model.weight': { dtype: 'BF16', shape: [2, 3], data_offsets: [0, 12] },
    }),
  });
  const ingest = createModelIngestion({ transport });
  const progress = [];

  const result = await ingest('org/model', {
    onShard: (done, total, file) => progress.push({ done, total, file }),
  });

  assert.equal(result.shardCount, 1);
  assert.equal(result.config.__hf_index_meta.total_size, 12);
  assert.deepEqual(result.tensors, [
    {
      name: 'model.weight', dtype: 'BF16', shape: [2, 3],
      shard: 'model-00001-of-00001.safetensors',
    },
  ]);
  assert.deepEqual(progress, [
    { done: 1, total: 1, file: 'model-00001-of-00001.safetensors' },
  ]);
});

test('model ingestion exposes stable failure facts instead of localized text', async () => {
  const ingest = createModelIngestion({ transport: makeMemoryTransport() });

  for (const repoId of ['invalid', 'org/model@deadbeef', 'org/model/revision/main', 'org/model extra']) {
    await assert.rejects(
      () => ingest(repoId),
      (error) => error instanceof IngestionError
        && error.code === 'invalid_repo_id'
        && error.details.repoId === repoId,
    );
  }
});

test('model ingestion keeps single-file fallback policy behind its interface', async () => {
  const transport = makeMemoryTransport({
    text: {
      'https://huggingface.co/api/models/org/model/revision/main': JSON.stringify({
        id: 'org/model', sha: COMMIT_ID,
      }),
      [`${PINNED_BASE}/config.json`]: JSON.stringify({ architectures: ['ExampleForCausalLM'] }),
    },
    range: safetensorsRoutes('model.safetensors', {
      'single.weight': { dtype: 'F32', shape: [1], data_offsets: [0, 4] },
    }),
  });
  const ingest = createModelIngestion({ transport });

  const result = await ingest('org/model');

  assert.deepEqual(result.shardFiles, ['model.safetensors']);
  assert.deepEqual(result.tensors, [
    { name: 'single.weight', dtype: 'F32', shape: [1], shard: 'model.safetensors' },
  ]);
});
