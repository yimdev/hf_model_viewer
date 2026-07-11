import glm52 from './profiles/glm_5_2.js';
import deepseekV4Pro from './profiles/deepseek_v4_pro.js';
import hy3 from './profiles/hy3.js';
import qwen36A3B from './profiles/qwen_3_6_35b_a3b.js';

export function createArchitectureLayoutCatalog(profiles) {
  const byRepoId = new Map();
  for (const profile of profiles) {
    for (const repository of profile.repositories) {
      if (byRepoId.has(repository.repoId)) {
        throw new Error(`Duplicate Architecture Profile repoId: ${repository.repoId}`);
      }
      byRepoId.set(repository.repoId, profile);
    }
  }

  return Object.freeze({
    profileForRepoId(repoId) {
      return byRepoId.get(repoId) || null;
    },
  });
}

const catalog = createArchitectureLayoutCatalog([glm52, deepseekV4Pro, hy3, qwen36A3B]);

export function profileForRepoId(repoId) {
  return catalog.profileForRepoId(repoId);
}
