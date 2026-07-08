/* Extension entry: Manifest V3 Background Service Worker.
 * Acts as the sole network egress, proxying fetches to the Hugging Face CDN
 * to bypass page-side CORS restrictions. */
import { installBackgroundNetHandler } from '../platform/net.js';

installBackgroundNetHandler();
