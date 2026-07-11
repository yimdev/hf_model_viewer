/* Shared representation and validation primitives; no model layout formulas live here. */
const GB = 1024 ** 3;
const SEMANTIC_DTYPE_BYTES = Object.freeze({
  F8: 1, INT8: 1, F16: 2, BF16: 2, F32: 4, F64: 8,
});

export function validateSequenceWorkload({
  batch,
  seq,
  sequenceLengths,
  maxContext,
  minimumBatch = 0,
  allowEmptySequenceLengths = true,
}) {
  if (sequenceLengths != null) {
    if (!Array.isArray(sequenceLengths)) {
      return {
        error: 'profile_input_out_of_range', details: { sequenceLengths: 'expected an array', maxContext },
      };
    }
    if (!allowEmptySequenceLengths && sequenceLengths.length === 0) {
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

  if (
    !Number.isInteger(batch)
    || batch < minimumBatch
    || !Number.isInteger(seq)
    || seq < 0
    || seq > maxContext
  ) {
    return { error: 'profile_input_out_of_range', details: { batch, seq, maxContext } };
  }
  const tokenCount = batch * seq;
  if (!Number.isSafeInteger(tokenCount)) {
    return { error: 'profile_calculation_out_of_range', details: { batch, seq, maxContext } };
  }
  return { entries: [{ length: seq, count: batch }], tokenCount, ragged: false };
}

export function makeBuffer({
  id,
  label,
  layerGroup,
  elements,
  dtype,
  bytesPerElement,
  formula,
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
  };
}
