/* engine/safetensors.js — Zero-download Safetensors header parsing
 * ------------------------------------------------------------
 * Safetensors format:
 *   [ 8-byte little-endian uint64 = header length N ][ N bytes UTF-8 JSON ][ weight data ]
 * JSON structure: { "<tensor_name>": { dtype, shape, data_offsets:[s,e] }, "__metadata__": {...} }
 *
 * This module reads only the first 8 bytes + the header JSON via HTTP Range,
 * never touching the weight data.
 * ------------------------------------------------------------ */

import { t } from '../i18n.js';

const HEADER_LEN_BYTES = 8;

export async function readSafetensorsHeader(net, baseUrl, fileName, headers = {}) {
  const url = `${baseUrl}/${fileName}`;

  // 1) Read the first 8 bytes -> header length.
  const lenBuf = await net.range(url, 0, HEADER_LEN_BYTES - 1, headers);
  if (lenBuf.byteLength < HEADER_LEN_BYTES) {
    throw new Error(t('err.badSafetensors', { file: fileName }));
  }
  const dv = new DataView(lenBuf.buffer, lenBuf.byteOffset, lenBuf.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true)); // little-endian

  // 2) Read the header JSON slice.
  const headerBuf = await net.range(
    url,
    HEADER_LEN_BYTES,
    HEADER_LEN_BYTES + headerLen - 1,
    headers,
  );
  const text = new TextDecoder().decode(headerBuf);
  const json = JSON.parse(text);
  return json; // { tensorName: {dtype, shape, data_offsets}, __metadata__ }
}
