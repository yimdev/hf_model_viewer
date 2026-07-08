/* engine/index.js — Parse orchestration (analyze)
 * ------------------------------------------------------------
 * Pipeline:
 *   config.json -> global hyper-params
 *   model.safetensors.index.json / single file -> shard list
 *   parallel HTTP Range -> per-shard header JSON -> flat tensor list
 * ------------------------------------------------------------ */

import { makeNet } from '../platform/net.js';
import { readSafetensorsHeader } from './safetensors.js';
import { mapLimit } from './util.js';
import { t } from '../i18n.js';

const CONFIG_URL = (repo) => `https://huggingface.co/${repo}/resolve/main/config.json`;
const INDEX_URL = (repo) => `https://huggingface.co/${repo}/resolve/main/model.safetensors.index.json`;
const SHARD_BASE = (repo) => `https://huggingface.co/${repo}/resolve/main`;

const NOT_SAFETENSORS = t('err.noSafetensors');

/**
 * Parse a Hugging Face repo, returning { repoId, config, tensors, shardFiles, shardCount }.
 * @param {string} repoId  e.g. org/repo
 * @param {object} [opts]
 * @param {string} [opts.token]  optional Hugging Face Access Token (gated / private models)
 * @param {(done:number,total:number,file:string)=>void} [opts.onShard]  shard progress callback
 */
export async function analyze(repoId, { token, onShard } = {}) {
  if (!repoId || !/^[\w.-]+\/[\w.-]+/.test(repoId)) {
    throw new Error(t('err.badRepoId'));
  }

  const net = makeNet();
  const auth = token ? { Authorization: `Bearer ${token}` } : {};

  // 1) config.json
  let config;
  try {
    config = JSON.parse(await net.text(CONFIG_URL(repoId), auth));
  } catch (e) {
    throw new Error(`${t('err.configFetch')}${e.message}`);
  }

  // 2) Shard discovery: prefer index.json, fall back to single model.safetensors.
  const base = SHARD_BASE(repoId);
  let shardFiles = [];
  try {
    const idx = JSON.parse(await net.text(INDEX_URL(repoId), auth));
    shardFiles = [...new Set(Object.values(idx.weight_map))];
    if (idx.metadata) config.__hf_index_meta = idx.metadata;
  } catch {
    try {
      await readSafetensorsHeader(net, base, 'model.safetensors', auth);
      shardFiles = ['model.safetensors'];
    } catch {
      throw new Error(NOT_SAFETENSORS);
    }
  }

  if (!shardFiles.length) throw new Error(NOT_SAFETENSORS);

  // 3) Fetch all shard headers in parallel (cap 16 concurrent, keeps huge
  //    MoE repos under ~3s).
  const headersMap = {};
  let done = 0;
  await mapLimit(shardFiles, 16, async (file) => {
    headersMap[file] = await readSafetensorsHeader(net, base, file, auth);
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

  return { repoId, config, tensors, shardFiles, shardCount: shardFiles.length };
}
