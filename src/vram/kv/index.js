import { profileForRepoId } from './catalog.js';

export function computeKV({
  source = {}, config = {}, batch = 1, seq = 8192, sequenceLengths = null,
} = {}) {
  const { repoId, commitId } = source;
  const profile = profileForRepoId(repoId);
  if (!profile) {
    return Object.freeze({
      calculation: Object.freeze({
        status: 'unknown',
        diagnostic: Object.freeze({
          code: 'unsupported_model_architecture',
          details: Object.freeze({ repoId }),
        }),
      }),
      assurance: null,
      provenance: Object.freeze({ repoId, commitId, auditedCommitId: null }),
      profile: null,
      buffers: Object.freeze([]),
      totalBytes: null,
      vKV: null,
      note: '',
    });
  }

  return profile.evaluate({
    source,
    config,
    workload: { batch, seq, sequenceLengths },
  });
}
