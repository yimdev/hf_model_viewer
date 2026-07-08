import { defineConfig } from 'vite';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { copyFileSync, mkdirSync, writeFileSync } from 'node:fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const target = process.env.BUILD_TARGET || 'web';

// Dual-form build pipeline:
//   BUILD_TARGET=web -> GitHub Pages static site (dist-web/), entry at root
//   BUILD_TARGET=ext -> browser extension bundle (dist-ext/): manifest + popup +
//                       background, all at root
// Each form gets its own root + absolute entry paths so HTML/JS land at the
// product root, matching manifest.json's default_popup / service_worker paths.
const root = target === 'web' ? resolve(__dirname, 'src/web') : resolve(__dirname, 'src/ext');
const entry = target === 'web'
  ? resolve(root, 'index.html')
  : { popup: resolve(root, 'popup.html'), background: resolve(root, 'background.js') };

export default defineConfig({
  root,
  base: target === 'web' ? './' : '',
  build: {
    outDir: target === 'web' ? resolve(__dirname, 'dist-web') : resolve(__dirname, 'dist-ext'),
    emptyOutDir: true,
    rollupOptions: {
      input: entry,
      output:
        target === 'ext'
          ? { entryFileNames: '[name].js', chunkFileNames: '[name].js', assetFileNames: '[name].[ext]' }
          : undefined,
    },
  },
  define: {
    // Inject the build target so runtime can branch by form (e.g. route
    // network requests through the background in extension form).
    'import.meta.env.BUILD_TARGET': JSON.stringify(target),
  },
  plugins: [
    {
      name: 'copy-static-assets',
      closeBundle() {
        if (target === 'ext') {
          const out = resolve(__dirname, 'dist-ext');
          mkdirSync(out, { recursive: true });
          copyFileSync(resolve(__dirname, 'src/ext/manifest.json'), resolve(out, 'manifest.json'));
        } else {
          // GitHub Pages runs Jekyll by default, which can mangle assets with
          // underscore prefixes; write .nojekyll to disable it. base is './'
          // so it works under the project subpath (/<repo>/).
          const out = resolve(__dirname, 'dist-web');
          mkdirSync(out, { recursive: true });
          writeFileSync(resolve(out, '.nojekyll'), '');
        }
      },
    },
  ],
});
