import { analyze } from '../engine/index.js';
import { createTensorMetadataIndex } from '../tree/index.js';
import { estimateVRAM } from '../vram/index.js';

function maxContextLength(config) {
  const value = config.max_position_embeddings
    ?? config.max_sequence_length
    ?? config.max_position_embedding;
  return typeof value === 'number' && value > 0 ? value : null;
}

function freezeSnapshot(snapshot) {
  return Object.freeze({
    ...snapshot,
    workload: Object.freeze({ ...snapshot.workload }),
    progress: snapshot.progress ? Object.freeze({ ...snapshot.progress }) : null,
  });
}

export function createApplicationSession({ ingest = analyze } = {}) {
  const listeners = new Set();
  let snapshot = freezeSnapshot({
    phase: 'idle',
    repoId: '',
    workload: { batch: 1, seq: 8192 },
    progress: null,
    model: null,
    estimate: null,
    error: null,
  });

  const publish = (changes) => {
    snapshot = freezeSnapshot({ ...snapshot, ...changes });
    listeners.forEach((listener) => listener(snapshot));
    return snapshot;
  };

  const recompute = () => {
    if (!snapshot.model) return snapshot;
    const { model, workload } = snapshot;
    return publish({
      estimate: estimateVRAM({
        source: { repoId: model.repoId, commitId: model.commitId },
        config: model.config,
        tensors: model.tensors,
        workload,
        tensorMetadataIndex: model.tensorMetadataIndex,
      }),
    });
  };

  return Object.freeze({
    getSnapshot() {
      return snapshot;
    },

    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    setWorkload({ batch = snapshot.workload.batch, seq = snapshot.workload.seq } = {}) {
      publish({ workload: { batch, seq } });
      return recompute();
    },

    async load(repoId, { token } = {}) {
      publish({ phase: 'loading', repoId, progress: null, error: null });
      try {
        const result = await ingest(repoId, {
          token,
          onShard: (done, total, file) => publish({ progress: { done, total, file } }),
        });
        const tensorMetadataIndex = createTensorMetadataIndex(result.tensors);
        const maximum = maxContextLength(result.config);
        const model = Object.freeze({
          ...result,
          tensorMetadataIndex,
          maxContextLength: maximum,
        });
        publish({
          phase: 'ready',
          model,
          workload: { ...snapshot.workload, seq: maximum || 8192 },
          progress: null,
          error: null,
        });
        return recompute();
      } catch (error) {
        return publish({ phase: 'error', progress: null, error });
      }
    },
  });
}
