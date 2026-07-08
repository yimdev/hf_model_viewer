/* Extension entry: browser extension Popup.
 * Reuses the exact same business core and UI orchestration as the web;
 * network requests are proxied through the background service worker. */
import { mountApp } from '../ui/app.js';
import '../styles.css';

const root = document.getElementById('app');
mountApp(root);
