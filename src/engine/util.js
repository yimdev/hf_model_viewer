/* engine/util.js — Concurrency-limited utility */

/**
 * Iterate items with a fixed concurrency cap, running fn on each (avoids
 * firing 160+ shard requests at once and overwhelming the browser).
 */
export async function mapLimit(items, limit, fn) {
  let i = 0;
  const n = Math.max(1, Math.min(limit, items.length));
  const workers = Array.from({ length: n }, async () => {
    while (i < items.length) {
      const idx = i++;
      await fn(items[idx], idx);
    }
  });
  await Promise.all(workers);
}
