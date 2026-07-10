const GB = 1024 ** 3;
const SEMANTIC_DTYPE_BYTES = Object.freeze({ BF16: 2, F32: 4 });

export function sameArray(actual, expected) {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

export function tensorMatches(byName, name, shape, dtype) {
  const tensor = byName.get(name);
  return tensor && tensor.dtype === dtype && sameArray(tensor.shape, shape);
}

export function validateSequenceWorkload({ batch, seq, sequenceLengths, maxContext }) {
  if (sequenceLengths != null) {
    if (!Array.isArray(sequenceLengths) || sequenceLengths.length === 0) {
      return {
        error: 'profile_input_out_of_range',
        details: { sequenceLengths: 'expected a non-empty array', maxContext },
      };
    }
    let tokenCount = 0;
    const entries = [];
    for (let index = 0; index < sequenceLengths.length; index++) {
      const length = sequenceLengths[index];
      if (!Number.isInteger(length) || length < 0 || length > maxContext) {
        return {
          error: 'profile_input_out_of_range',
          details: { sequenceIndex: index, value: length, maxContext },
        };
      }
      tokenCount += length;
      if (!Number.isSafeInteger(tokenCount)) {
        return {
          error: 'profile_calculation_out_of_range',
          details: { sequenceCount: sequenceLengths.length, maxContext },
        };
      }
      entries.push({ length, count: 1 });
    }
    return { entries, tokenCount, ragged: true };
  }

  if (!Number.isInteger(batch) || batch < 1 || !Number.isInteger(seq) || seq < 0 || seq > maxContext) {
    return { error: 'profile_input_out_of_range', details: { batch, seq, maxContext } };
  }
  const tokenCount = batch * seq;
  if (!Number.isSafeInteger(tokenCount)) {
    return { error: 'profile_calculation_out_of_range', details: { batch, seq, maxContext } };
  }
  return { entries: [{ length: seq, count: batch }], tokenCount, ragged: false };
}

export function modelClassIdentifiers(config = {}) {
  return Array.isArray(config.architectures)
    ? config.architectures.filter((name) => typeof name === 'string' && name.length > 0)
    : [];
}

export function makeBuffer({
  id,
  label,
  layerGroup,
  elements,
  dtype,
  bytesPerElement,
  formula,
  evidenceIds = [],
}) {
  if (SEMANTIC_DTYPE_BYTES[dtype] !== bytesPerElement) {
    throw new Error(`Dtype width mismatch for ${id}: ${dtype} is not ${bytesPerElement} bytes`);
  }
  const bytes = elements * bytesPerElement;
  if (
    !Number.isSafeInteger(elements)
    || !Number.isSafeInteger(bytesPerElement)
    || !Number.isSafeInteger(bytes)
    || elements < 0
    || bytesPerElement <= 0
  ) {
    throw new Error(`Invalid KV buffer size for ${id}`);
  }
  return {
    id,
    label,
    layerGroup,
    elements,
    dtype,
    bytesPerElement,
    bytes,
    gb: bytes / GB,
    formula,
    evidenceIds,
  };
}

export function verifiedResult({ profile, buffers, note = '' }) {
  const totalBytes = buffers.reduce((sum, buffer) => sum + buffer.bytes, 0);
  if (!Number.isSafeInteger(totalBytes) || totalBytes < 0) {
    throw new Error(`Invalid KV total for ${profile.id}`);
  }
  return {
    status: 'verified',
    kvUnknown: false,
    vKV: totalBytes / GB,
    totalBytes,
    profile,
    buffers,
    diagnostic: null,
    note,
  };
}

export function unknownResult({ code, modelClassIds, details = null, status = 'unsupported' }) {
  return {
    status,
    kvUnknown: true,
    vKV: null,
    totalBytes: null,
    profile: null,
    buffers: [],
    diagnostic: {
      code,
      modelClassIdentifiers: modelClassIds,
      ...(details ? { details } : {}),
    },
    note: '',
  };
}
