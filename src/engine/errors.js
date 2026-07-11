export class IngestionError extends Error {
  constructor(code, details = null, options = {}) {
    super(code, options);
    this.name = 'IngestionError';
    this.code = code;
    this.details = details;
  }
}
