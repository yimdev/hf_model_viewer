/* ui/app.js — App orchestration
 * ------------------------------------------------------------
 * Two-column layout: left = config & calc controls, right = results.
 * Pipeline: application session -> Tensor Metadata Index -> tree + overview.
 * Language: strings come from i18n (t); switching re-paints the shell.
 * ------------------------------------------------------------ */

import '../styles.css';
import { createApplicationSession } from '../app/session.js';
import { renderTree } from './treeView.js';
import { renderChart } from './chart.js';
import { formatIngestionError } from './errors.js';
import { fmtNum, fmtGB, esc } from './format.js';
import { renderKVDetails } from './kvDetailsView.js';
import { t, getLang, setLang, onLangChange } from '../i18n.js';

function buildLayout() {
  const lang = getLang();
  const active = (l) => (lang === l ? ' active' : '');
  return `
<div class="app">
  <aside class="sidebar">
    <div class="brand">
      ${esc(t('brand.title'))}<small>${esc(t('brand.sub'))}</small>
      <div class="lang-toggle">
        <button class="lang-btn${active('zh')}" data-lang="zh">中文</button>
        <button class="lang-btn${active('en')}" data-lang="en">EN</button>
      </div>
    </div>

    <div class="search-row">
      <input id="repo" placeholder="${esc(t('ctl.repoPlaceholder'))}" />
      <button id="analyze">${esc(t('ctl.analyze'))}</button>
    </div>

    <details class="field">
      <summary>${esc(t('ctl.advanced'))}</summary>
      <input id="token" class="seq-input" style="margin-top:8px" placeholder="${esc(t('ctl.tokenPlaceholder'))}" />
    </details>

    <div class="field">
      <label>${esc(t('ctl.batchSize'))}<span class="bubble" id="batchVal">1</span></label>
      <input type="range" id="batch" min="1" max="128" value="1" />
    </div>

    <div class="field">
      <label>${esc(t('ctl.contextLength'))}</label>
      <div class="chips">
        <button data-seq="1024">1K</button>
        <button data-seq="8192">8K</button>
        <button data-seq="65536">64K</button>
        <button data-seq="131072">128K</button>
        <button data-seq="524288">512K</button>
        <button data-seq="1048576">1M</button>
      </div>
      <input type="number" id="seq" class="seq-input" value="8192" min="1024" max="131072" />
    </div>

    <div id="summary" class="summary" style="display:none"></div>
    <div id="status" class="status"></div>
  </aside>

  <main class="main">
    <section class="overview">
      <h2>${esc(t('ov.title'))}</h2>
      <div class="summary-grid" id="stats"><div class="empty">${esc(t('ctl.empty'))}</div></div>
      <div class="overview-chart" tabindex="0" aria-label="${esc(t('ov.title'))}">
        <div class="overview-chart-content">
          <div class="overview-chart-keys" role="list"></div>
          <div class="overview-chart-plot"><canvas id="chart"></canvas></div>
        </div>
      </div>
      <h3 class="overview-section-title">${esc(t('ov.kvTitle'))}</h3>
      <div id="kvdetails" class="kv-details"><div class="empty">${esc(t('ctl.empty'))}</div></div>
    </section>
    <section class="tree">
      <h2>${esc(t('ov.treeTitle'))}</h2>
      <div id="tree"><div class="empty">${esc(t('ov.treeEmpty'))}</div></div>
    </section>
  </main>
</div>`;
}

export function mountApp(rootEl) {
  const session = createApplicationSession();
  let state = null;
  let lastRepo = '';

  const $ = (id) => rootEl.querySelector('#' + id);

  function setStatus(msg, kind = '') {
    const statusEl = $('status');
    statusEl.className = 'status' + (kind ? ' ' + kind : '');
    statusEl.textContent = msg;
  }

  function recompute() {
    if (!state) return;
    const batch = parseInt($('batch').value, 10) || 1;
    const seq = parseInt($('seq').value, 10) || 8192;

    const sessionState = session.setWorkload({ batch, seq });
    const est = sessionState.estimate;

    renderChart($('chart'), est);
    renderKVDetails($('kvdetails'), est);

    // VRAM breakdown card (no GPU recommendation).
    const summaryEl = $('summary');
    summaryEl.style.display = '';
    const profile = est.profile;
    const maxCtx = state.maxContextLength;
    const maxCtxStr = maxCtx ? `${fmtNum(maxCtx)} tokens` : '—';
    summaryEl.innerHTML = `
      <div class="hw-note" style="font-size:13px">${esc(t('sum.total'))}<b>${fmtGB(est.vTotal)}</b> ｜ ${esc(t('sum.weights'))} ${fmtGB(est.vWeights)} ｜ KV ${est.complete ? fmtGB(est.vKV) : '—'} ｜ ${esc(t('sum.maxContext'))}${maxCtxStr}</div>
      ${profile ? `<div class="hw-note">${esc(t('sum.kvProfile'))}<span class="tag profile">${esc(profile.label)}</span> ｜ ${esc(t('sum.kvLayout'))}${esc(profile.layout.id)}@${esc(profile.layout.version)}</div>` : `<div class="hw-note incomplete">${esc(t('kv.totalUnknown'))}</div>`}
      ${est.note ? `<div class="hw-note dsa-note">${esc(est.note)}</div>` : ''}
    `;
  }

  function renderStats() {
    const { config, tensorMetadataIndex } = state;
    const tree = tensorMetadataIndex.summary;
    const arch = Array.isArray(config.architectures)
      ? config.architectures.join(', ')
      : config.model_type || '—';
    const moe = tree.isMoe ? t('stat.moeYes', { n: tree.numExperts }) : t('stat.no');
    $('stats').innerHTML = `
      <div class="stat">${esc(t('stat.totalParams'))}<b>${fmtNum(tree.totalParams)}</b></div>
      <div class="stat">${esc(t('stat.layers'))}<b>${tree.numLayers}</b></div>
      <div class="stat">${esc(t('stat.moe'))}<b>${moe}</b></div>
      <div class="stat">${esc(t('stat.arch'))}<b style="font-size:13px">${esc(arch)}</b></div>
      <div class="stat">${esc(t('stat.shards'))}<b>${state.shardCount ?? '—'}</b></div>
    `;
  }

  async function run() {
    const repoInput = $('repo');
    const repo = repoInput.value.trim();
    if (!repo) {
      setStatus(t('status.enterRepo'), 'error');
      return;
    }
    const analyzeBtn = $('analyze');
    analyzeBtn.disabled = true;
    setStatus(t('status.fetching'));
    try {
      session.setWorkload({
        batch: parseInt($('batch').value, 10) || 1,
        seq: parseInt($('seq').value, 10) || 8192,
      });
      const sessionState = await session.load(repo, {
        token: $('token').value.trim() || undefined,
      });
      if (sessionState.phase === 'error') throw sessionState.error;
      state = sessionState.model;
      lastRepo = repo;
      renderStats();

      // Context Length defaults to this model's max context length.
      const maxCtx = state.maxContextLength;
      const seqInput = $('seq');
      if (maxCtx) {
        seqInput.max = String(Math.max(maxCtx, 131072));
        seqInput.value = String(maxCtx);
      } else {
        seqInput.value = '8192';
      }
      rootEl.querySelectorAll('.chips button').forEach((x) => x.classList.remove('active'));

      renderTree($('tree'), state.tensorMetadataIndex.tensorNameTree);
      recompute();
      setStatus(t('status.done', { shards: state.shardCount, tensors: state.tensors.length }), 'ok');
    } catch (e) {
      setStatus(formatIngestionError(e), 'error');
    } finally {
      analyzeBtn.disabled = false;
    }
  }

  session.subscribe((next) => {
    if (next.phase !== 'loading') return;
    if (next.progress) {
      setStatus(t('status.shard', next.progress));
    } else {
      setStatus(t('status.fetching'));
    }
  });

  // ---- Shell render + event binding ----
  function bindShell() {
    const analyzeBtn = $('analyze');
    const repoInput = $('repo');
    const batchInput = $('batch');
    const seqInput = $('seq');

    analyzeBtn.addEventListener('click', run);
    repoInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') run();
    });

    batchInput.addEventListener('input', () => {
      $('batchVal').textContent = batchInput.value;
      recompute();
    });

    seqInput.addEventListener('input', () => recompute());
    rootEl.querySelectorAll('.chips button').forEach((c) =>
      c.addEventListener('click', () => {
        seqInput.value = c.dataset.seq;
        rootEl.querySelectorAll('.chips button').forEach((x) => x.classList.remove('active'));
        c.classList.add('active');
        recompute();
      }),
    );

    rootEl.querySelectorAll('.lang-btn').forEach((b) =>
      b.addEventListener('click', () => setLang(b.dataset.lang)),
    );
  }

  function paint() {
    rootEl.innerHTML = buildLayout();
    document.title = t('brand.title');
    document.documentElement.lang = getLang() === 'zh' ? 'zh-CN' : 'en';
    bindShell();
    if (state) {
      $('repo').value = lastRepo;
      renderStats();
      renderTree($('tree'), state.tensorMetadataIndex.tensorNameTree);
      recompute();
    }
  }

  // Re-paint the shell (and results, if any) when language changes.
  onLangChange(paint);

  paint();
}
