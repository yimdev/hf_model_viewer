const MAX_CONTEXT_KEYS = Object.freeze([
  'max_position_embeddings',
  'max_position_embedding',
  'max_sequence_length',
  'max_sequence_len',
]);

export function firstDefined(config, keys) {
  for (const key of keys) {
    if (config?.[key] != null) return config[key];
  }
  return undefined;
}

export function textModelConfig(config = {}) {
  return config.text_config && typeof config.text_config === 'object'
    ? config.text_config
    : config;
}

export function modelDefault(config, keys) {
  const textConfig = textModelConfig(config);
  return firstDefined(textConfig, keys) ?? firstDefined(config, keys);
}

export function maxContextLength(config = {}) {
  const value = modelDefault(config, MAX_CONTEXT_KEYS);
  return Number.isInteger(value) && value > 0 ? value : null;
}
