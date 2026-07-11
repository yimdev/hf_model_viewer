import { profileForRepoId } from './catalog.js';
import { computeGenericKV } from './generic.js';

export function computeKV({
  source = {}, config = {}, batch = 1, seq = 8192, sequenceLengths = null,
} = {}) {
  const { repoId } = source;
  const profile = profileForRepoId(repoId);
  if (!profile) {
    return computeGenericKV({ source, config, batch, seq, sequenceLengths });
  }

  return profile.evaluate({
    source,
    config,
    workload: { batch, seq, sequenceLengths },
  });
}
