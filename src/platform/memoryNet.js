function routeKey(url, start = null, end = null) {
  return start == null ? url : `${url}#${start}-${end}`;
}

export function makeMemoryTransport({ text = {}, range = {} } = {}) {
  const textRoutes = new Map(Object.entries(text));
  const rangeRoutes = new Map(Object.entries(range));

  return Object.freeze({
    async text(url) {
      const value = textRoutes.get(routeKey(url));
      if (value instanceof Error) throw value;
      if (value == null) throw new Error(`Missing in-memory text route: ${url}`);
      return value;
    },

    async range(url, start, end) {
      const key = routeKey(url, start, end);
      const value = rangeRoutes.get(key);
      if (value instanceof Error) throw value;
      if (value == null) throw new Error(`Missing in-memory range route: ${key}`);
      return value instanceof Uint8Array ? value : new Uint8Array(value);
    },
  });
}
