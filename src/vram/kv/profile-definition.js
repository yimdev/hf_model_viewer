const GB = 1024 ** 3;
const COMMIT_ID_RE = /^[0-9a-f]{40}$/;
const IDENTIFIER_RE = /^[a-z0-9][a-z0-9._-]*$/;
const INPUT_NAME_RE = /^[A-Za-z][A-Za-z0-9_]*$/;
const SEMVER_RE = /^\d+\.\d+\.\d+$/;
const REPO_ID_RE = /^[\w.-]+\/[\w.-]+$/;

function cloneAndFreeze(value) {
  if (Array.isArray(value)) return Object.freeze(value.map(cloneAndFreeze));
  if (value && typeof value === 'object') {
    return Object.freeze(Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, cloneAndFreeze(entry)]),
    ));
  }
  return value;
}

function deepEqual(left, right) {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left)
      && Array.isArray(right)
      && left.length === right.length
      && left.every((value, index) => deepEqual(value, right[index]));
  }
  if (left && right && typeof left === 'object' && typeof right === 'object') {
    const leftKeys = Object.keys(left).sort();
    const rightKeys = Object.keys(right).sort();
    return deepEqual(leftKeys, rightKeys)
      && leftKeys.every((key) => deepEqual(left[key], right[key]));
  }
  return false;
}

function readPath(config, path) {
  const segments = Array.isArray(path) ? path : String(path).split('.');
  return segments.reduce(
    (value, segment) => (value == null ? undefined : value[segment]),
    config,
  );
}

function issueFor(name, descriptor, value) {
  if (value === undefined) {
    return { input: name, configPath: descriptor.path, code: 'missing' };
  }
  const validation = descriptor.validate?.(value);
  if (validation === true || validation == null) return null;
  return {
    input: name,
    configPath: descriptor.path,
    code: 'invalid',
    value,
    expected: typeof validation === 'string' ? validation : validation?.expected,
  };
}

function profileIdentity(definition) {
  return Object.freeze({
    id: definition.id,
    version: definition.version,
    label: definition.label,
    layout: cloneAndFreeze(definition.layout),
  });
}

function assuranceFor({ audit, commitId, differences }) {
  const warnings = [];
  if (commitId !== audit.auditedCommitId) {
    warnings.push(Object.freeze({
      code: 'commit_mismatch',
      currentCommitId: commitId,
      auditedCommitId: audit.auditedCommitId,
    }));
  }
  if (differences.length > 0) {
    warnings.push(Object.freeze({ code: 'config_mismatch', differences }));
  }
  return Object.freeze({
    status: warnings.length === 0 ? 'verified' : 'warning',
    warnings: Object.freeze(warnings),
  });
}

function unknownResult({ profile, provenance, assurance, code, details = null }) {
  return Object.freeze({
    calculation: Object.freeze({
      status: 'unknown',
      diagnostic: Object.freeze({ code, ...(details ? { details } : {}) }),
    }),
    assurance,
    provenance,
    profile,
    buffers: Object.freeze([]),
    totalBytes: null,
    vKV: null,
    note: '',
  });
}

export function defineArchitectureProfile(definition) {
  const inputEntries = Object.entries(definition.configInputs || {});
  const inputNames = inputEntries.map(([name]) => name).sort();
  if (!IDENTIFIER_RE.test(definition.id || '')) throw new Error('Profile id is invalid');
  if (!SEMVER_RE.test(definition.version || '')) throw new Error(`Profile ${definition.id} version is invalid`);
  if (typeof definition.label !== 'string' || definition.label.trim() === '') {
    throw new Error(`Profile ${definition.id} label is invalid`);
  }
  if (
    !definition.layout
    || !IDENTIFIER_RE.test(definition.layout.id || '')
    || !SEMVER_RE.test(definition.layout.version || '')
  ) {
    throw new Error(`Profile ${definition.id} Layout identity is invalid`);
  }
  if (inputNames.length === 0) throw new Error(`Profile ${definition.id} has no config inputs`);
  for (const [name, descriptor] of inputEntries) {
    const segments = Array.isArray(descriptor?.path) ? descriptor.path : [descriptor?.path];
    if (
      !INPUT_NAME_RE.test(name)
      || segments.length === 0
      || segments.some((segment) => typeof segment !== 'string' || segment.trim() === '')
      || typeof descriptor?.validate !== 'function'
    ) {
      throw new Error(`Profile ${definition.id} config input ${name} has an invalid path`);
    }
  }
  if (typeof definition.calculateLayout !== 'function') {
    throw new Error(`Profile ${definition.id} has no Layout calculation`);
  }

  const repositories = (definition.repositories || []).map((audit) => {
    if (!REPO_ID_RE.test(audit.repoId || '')) {
      throw new Error(`Profile ${definition.id} has an invalid repoId`);
    }
    if (!COMMIT_ID_RE.test(audit.auditedCommitId)) {
      throw new Error(`Profile ${definition.id} has an invalid audited commit`);
    }
    const baselineNames = Object.keys(audit.baselineInputs || {}).sort();
    if (!deepEqual(inputNames, baselineNames)) {
      throw new Error(`Profile ${definition.id} baseline inputs do not match config inputs`);
    }
    for (const [name, descriptor] of inputEntries) {
      const issue = issueFor(name, descriptor, audit.baselineInputs[name]);
      if (issue) throw new Error(`Profile ${definition.id} has invalid baseline input ${name}`);
    }
    const baselineInputs = cloneAndFreeze(audit.baselineInputs);
    const baselineIssues = definition.validateInputs?.(baselineInputs) || [];
    if (baselineIssues.length > 0) {
      throw new Error(`Profile ${definition.id} baseline inputs violate Layout invariants`);
    }
    return Object.freeze({
      repoId: audit.repoId,
      auditedCommitId: audit.auditedCommitId,
      baselineInputs,
    });
  });
  if (repositories.length === 0) throw new Error(`Profile ${definition.id} has no repositories`);

  const identity = profileIdentity(definition);
  const auditByRepoId = new Map(repositories.map((audit) => [audit.repoId, audit]));

  return Object.freeze({
    ...identity,
    repositories: Object.freeze(repositories),

    evaluate({ source = {}, config = {}, workload = {} }) {
      const { repoId, commitId } = source;
      const audit = auditByRepoId.get(repoId);
      if (!audit) throw new Error(`Profile ${definition.id} cannot evaluate ${repoId}`);
      const provenance = Object.freeze({
        repoId,
        commitId,
        auditedCommitId: audit.auditedCommitId,
      });
      if (!COMMIT_ID_RE.test(commitId)) {
        return unknownResult({
          profile: identity,
          provenance,
        assurance: null,
          code: 'invalid_model_provenance',
        });
      }

      const inputs = {};
      const issues = [];
      const differences = [];
      for (const [name, descriptor] of inputEntries) {
        const value = cloneAndFreeze(readPath(config, descriptor.path));
        inputs[name] = value;
        const issue = issueFor(name, descriptor, value);
        if (issue) issues.push(Object.freeze(issue));
        if (!deepEqual(value, audit.baselineInputs[name])) {
          differences.push(Object.freeze({
            input: name,
            configPath: descriptor.path,
            auditedValue: cloneAndFreeze(audit.baselineInputs[name]),
            currentValue: value,
          }));
        }
      }
      const crossFieldIssues = issues.length === 0
        ? definition.validateInputs?.(Object.freeze({ ...inputs })) || []
        : [];
      issues.push(...crossFieldIssues.map((issue) => Object.freeze(issue)));
      const frozenDifferences = Object.freeze(differences);
      const assurance = assuranceFor({ audit, commitId, differences: frozenDifferences });
      if (issues.length > 0) {
        return unknownResult({
          profile: identity,
          provenance,
          assurance,
          code: 'invalid_profile_config',
          details: { issues: Object.freeze(issues) },
        });
      }

      let layoutResult;
      try {
        layoutResult = definition.calculateLayout({
          inputs: Object.freeze({ ...inputs }),
          workload: Object.freeze({ ...workload }),
        });
      } catch (error) {
        return unknownResult({
          profile: identity,
          provenance,
          assurance,
          code: 'profile_calculation_out_of_range',
          details: { message: error.message },
        });
      }
      if (layoutResult.error) {
        return unknownResult({
          profile: identity,
          provenance,
          assurance,
          code: layoutResult.error,
          details: layoutResult.details,
        });
      }

      const buffers = Object.freeze(layoutResult.buffers || []);
      const totalBytes = buffers.reduce((sum, buffer) => sum + buffer.bytes, 0);
      if (!Number.isSafeInteger(totalBytes) || totalBytes < 0) {
        return unknownResult({
          profile: identity,
          provenance,
          assurance,
          code: 'profile_calculation_out_of_range',
        });
      }
      return Object.freeze({
        calculation: Object.freeze({ status: 'computed', diagnostic: null }),
        assurance,
        provenance,
        profile: identity,
        buffers,
        totalBytes,
        vKV: totalBytes / GB,
        note: layoutResult.note || '',
      });
    },
  });
}

export const profileConfigInput = Object.freeze({
  positiveInteger(path) {
    return Object.freeze({
      path,
      validate: (value) => (Number.isInteger(value) && value > 0 ? true : 'positive integer'),
    });
  },
  array(path) {
    return Object.freeze({
      path,
      validate: (value) => (Array.isArray(value) ? true : 'array'),
    });
  },
});
