/* ============================================================
 * platform/net.js — Network abstraction layer (dual-form core)
 * ------------------------------------------------------------
 * Design goal: business core (engine / vram / tree) does not depend on the
 * runtime environment.
 *  - Web form: calls the browser global fetch directly.
 *  - Extension form (Manifest V3): all requests route to the Background
 *    Service Worker, which issues them under the host_permissions declared in
 *    the manifest, bypassing page-side CORS.
 *
 * Exposes two capabilities:
 *   net.text(url, headers)             -> full text
 *   net.range(url, start, end, headers) -> Uint8Array of the byte range
 * ============================================================ */

const HF_UA = { 'User-Agent': 'hf-vram-estimator/1.0' };

function isExtensionContext() {
  return (
    typeof chrome !== 'undefined' &&
    chrome.runtime &&
    chrome.runtime.id &&
    typeof chrome.runtime.sendMessage === 'function'
  );
}

/**
 * Perform a raw fetch. In extension context it talks to the background via
 * chrome.runtime; the return shape is unified as { ok, status, body: number[] }
 * (body is a byte array).
 */
function rawFetch(url, { headers = {}, method = 'GET', range } = {}) {
  const h = { ...HF_UA, ...headers };
  if (range) h['Range'] = `bytes=${range[0]}-${range[1]}`;

  if (isExtensionContext()) {
    return chrome.runtime.sendMessage({ __hfNet: true, url, method, headers: h });
  }

  return fetch(url, { method, headers: h }).then(async (res) => {
    const buf = await res.arrayBuffer();
    return { ok: res.ok, status: res.status, body: Array.from(new Uint8Array(buf)) };
  });
}

export function makeNet() {
  return {
    async text(url, headers = {}) {
      const r = await rawFetch(url, { headers });
      if (!r.ok) throw new Error(`HTTP ${r.status} fetch failed: ${url}`);
      return new TextDecoder().decode(new Uint8Array(r.body));
    },

    async range(url, start, end, headers = {}) {
      const r = await rawFetch(url, { headers, range: [start, end] });
      if (!r.ok) throw new Error(`HTTP ${r.status} Range request failed: ${url} (${start}-${end})`);
      return new Uint8Array(r.body);
    },
  };
}

/* Extension Background Service Worker entry (referenced only in ext build). */
export function installBackgroundNetHandler() {
  if (typeof chrome === 'undefined' || !chrome.runtime) return;
  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg && msg.__hfNet) {
      (async () => {
        try {
          const res = await fetch(msg.url, { method: msg.method || 'GET', headers: msg.headers || {} });
          const buf = await res.arrayBuffer();
          sendResponse({ ok: res.ok, status: res.status, body: Array.from(new Uint8Array(buf)) });
        } catch (e) {
          sendResponse({ ok: false, status: 0, error: String(e) });
        }
      })();
      return true; // keep the message channel open for the async response
    }
    return false;
  });
}
