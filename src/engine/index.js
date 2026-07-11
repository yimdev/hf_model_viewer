/* engine/index.js — Parse orchestration (analyze)
 * ------------------------------------------------------------
 * Pipeline:
 *   config.json -> global hyper-params
 *   model.safetensors.index.json / single file -> shard list
 *   parallel HTTP Range -> per-shard header JSON -> flat tensor list
 * ------------------------------------------------------------ */

import { makeFetchTransport } from '../platform/net.js';
import { readSafetensorsHeader } from './safetensors.js';
import { mapLimit } from './util.js';
import { IngestionError } from './errors.js';

const REVISION_URL = (repo) => `https://huggingface.co/api/models/${repo}/revision/main`;
const SHARD_BASE = (repo, commitId) => `https://huggingface.co/${repo}/resolve/${commitId}`;

/**
 * Parse a Hugging Face repo, returning { repoId, config, tensors, shardFiles, shardCount }.
 * @param {string} repoId  e.g. org/repo
 * @param {object} [opts]
 * @param {string} [opts.token]  optional Hugging Face Access Token (gated / private models)
 * @param {(done:number,total:number,file:string)=>void} [opts.onShard]  shard progress callback
 */
export function createModelIngestion({ transport }) {
  if (!transport) throw new TypeError('Model ingestion requires a transport adapter');

  return async function ingest(repoId, { token, onShard } = {}) {
    if (!repoId || !/^[\w.-]+\/[\w.-]+$/.test(repoId)) {
      throw new IngestionError('invalid_repo_id', { repoId });
    }

    const auth = token ? { Authorization: `Bearer ${token}` } : {};

    let source;
    try {
      const metadata = JSON.parse(await transport.text(REVISION_URL(repoId), auth));
      source = { repoId: metadata.id, commitId: metadata.sha };
      if (
        typeof source.repoId !== 'string'
        || !/^[\w.-]+\/[\w.-]+$/.test(source.repoId)
        || !/^[0-9a-f]{40}$/.test(source.commitId)
      ) {
        throw new Error('Invalid repository metadata');
      }
    } catch (error) {
      throw new IngestionError(
        'provenance_fetch_failed',
        { repoId, message: error.message },
        { cause: error },
      );
    }

    const base = SHARD_BASE(source.repoId, source.commitId);

  // 1) config.json
    let config;
    try {
      config = JSON.parse(await transport.text(`${base}/config.json`, auth));
    } catch (error) {
      throw new IngestionError('config_fetch_failed', { message: error.message }, { cause: error });
    }

  // 2) Shard discovery: prefer index.json, fall back to single model.safetensors.
    let shardFiles = [];
    try {
      const idx = JSON.parse(await transport.text(`${base}/model.safetensors.index.json`, auth));
      shardFiles = [...new Set(Object.values(idx.weight_map))];
      if (idx.metadata) config = { ...config, __hf_index_meta: idx.metadata };
    } catch {
      try {
        await readSafetensorsHeader(transport, base, 'model.safetensors', auth);
        shardFiles = ['model.safetensors'];
      } catch {
        throw new IngestionError('safetensors_unavailable', { repoId: source.repoId });
      }
    }

    if (!shardFiles.length) {
      throw new IngestionError('safetensors_unavailable', { repoId: source.repoId });
    }

  // 3) Fetch all shard headers in parallel (cap 16 concurrent, keeps huge
  //    MoE repos under ~3s).
    const headersMap = {};
    let done = 0;
    await mapLimit(shardFiles, 16, async (file) => {
      headersMap[file] = await readSafetensorsHeader(transport, base, file, auth);
      done += 1;
      onShard?.(done, shardFiles.length, file);
    });

  // 4) Merge into a flat tensor list.
    const tensors = [];
    for (const file of shardFiles) {
      const h = headersMap[file];
      for (const [name, meta] of Object.entries(h)) {
        if (name === '__metadata__') continue;
        tensors.push({ name, dtype: meta.dtype, shape: meta.shape, shard: file });
      }
    }

    return {
      ...source,
      config,
      tensors,
      shardFiles,
      shardCount: shardFiles.length,
    };
  };
}

export const analyze = createModelIngestion({ transport: makeFetchTransport() });
export { IngestionError } from './errors.js';
