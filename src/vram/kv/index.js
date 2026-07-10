/* vram/kv/index.js — Verified Architecture Profile dispatcher.
 *
 * Production KV calculation is fail-closed: exact model-class aliases select
 * curated profile candidates, then every candidate validates its own config
 * and safetensors signature before one dedicated layout may run. There is no
 * tensor-name heuristic or generic MHA/MLA fallback.
 */

import { profileCandidates } from './catalog.js';
import { modelClassIdentifiers, unknownResult } from './profile-primitives.js';

export function resolveProfileCandidate(candidates, input) {
  const evaluations = candidates.map((profile) => {
    try {
      return { profile, match: profile.match(input) };
    } catch (error) {
      return {
        profile,
        match: { matched: false, mismatches: [`profile_match_error:${error.message}`] },
      };
    }
  });
  const matches = evaluations.filter(({ match }) => match.matched);
  if (matches.length === 1) {
    return { status: 'matched', profile: matches[0].profile, evaluations };
  }
  if (matches.length > 1) {
    return { status: 'conflict', profileIds: matches.map(({ profile }) => profile.id), evaluations };
  }
  return { status: 'no_match', evaluations };
}

export function computeKV({
  config = {}, tensors = null, batch = 1, seq = 8192, sequenceLengths = null,
} = {}) {
  const modelClassIds = modelClassIdentifiers(config);
  if (modelClassIds.length === 0) {
    return unknownResult({ code: 'missing_model_class_identifier', modelClassIds });
  }

  const candidates = profileCandidates(modelClassIds);
  if (candidates.length === 0) {
    return unknownResult({ code: 'unsupported_model_architecture', modelClassIds });
  }
  if (modelClassIds.length > 1 && candidates.length > 1) {
    return unknownResult({
      code: 'conflicting_architecture_profiles',
      modelClassIds,
      status: 'conflict',
      details: { profileIds: candidates.map((candidate) => candidate.id) },
    });
  }
  const resolution = resolveProfileCandidate(candidates, { config, tensors });
  if (resolution.status === 'conflict') {
    return unknownResult({
      code: 'conflicting_architecture_profiles',
      modelClassIds,
      status: 'conflict',
      details: { profileIds: resolution.profileIds },
    });
  }
  if (resolution.status === 'no_match') {
    const candidatesDetails = resolution.evaluations.map(({ profile, match }) => ({
      profileId: profile.id,
      mismatches: match.mismatches,
    }));
    return unknownResult({
      code: 'profile_signature_mismatch',
      modelClassIds,
      details: candidatesDetails.length === 1
        ? candidatesDetails[0]
        : { candidates: candidatesDetails },
    });
  }

  const profile = resolution.profile;
  let result;
  try {
    result = profile.compute({ config, tensors, batch, seq, sequenceLengths });
  } catch (error) {
    return unknownResult({
      code: 'profile_calculation_out_of_range',
      modelClassIds,
      details: { profileId: profile.id, message: error.message },
    });
  }
  if (result.error) {
    return unknownResult({
      code: result.error,
      modelClassIds,
      details: result.details,
    });
  }
  return result;
}
