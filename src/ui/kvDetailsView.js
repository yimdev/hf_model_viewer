import { t } from '../i18n.js';
import { esc, fmtGB, fmtNum } from './format.js';

const MAX_SHOWN_DIFFERENCES = 12;

function diagnosticLabel(diagnostic) {
  if (!diagnostic) return t('kv.diag.unknown');
  const key = `kv.diag.${diagnostic.code}`;
  const translated = t(key);
  return translated === key ? diagnostic.code : translated;
}

function layerGroupLabel(group = {}) {
  const count = Number.isFinite(group.count) ? ` × ${group.count}` : '';
  if (Array.isArray(group.range)) {
    return `${group.label || t('kv.layers')} ${group.range[0]}–${group.range[1]}${count}`;
  }
  if (Array.isArray(group.indices)) {
    return `${group.label || t('kv.layers')} [${group.indices.join(', ')}]${count}`;
  }
  return `${group.label || '—'}${count}`;
}

function serialized(value) {
  const json = JSON.stringify(value);
  return json === undefined ? String(value) : json;
}

function assumptionLabel(assumption) {
  const key = `kv.assumption.${assumption.code}`;
  const translated = t(key);
  const label = translated === key ? assumption.code : translated;
  return assumption.assumedDtype ? `${label} (${assumption.assumedDtype})` : label;
}

export function renderKVDetails(container, estimate) {
  const provenance = estimate.provenance || {};
  const provenanceHtml = `
    <div class="kv-layout-id">
      ${esc(t('kv.repoId'))}<code>${esc(provenance.repoId || '—')}</code>
      ｜ ${esc(t('kv.currentCommit'))}<code>${esc(provenance.commitId || '—')}</code>
    </div>`;

  if (estimate.calculation.status !== 'computed') {
    const diagnostic = estimate.calculation.diagnostic;
    const issues = diagnostic?.details?.issues || [];
    const shownIssues = issues.slice(0, MAX_SHOWN_DIFFERENCES);
    container.innerHTML = `
      ${provenanceHtml}
      <div class="kv-unknown">
        <b>${esc(t('kv.unsupported'))}</b>
        <div>${esc(diagnosticLabel(diagnostic))}</div>
        ${shownIssues.length ? `<div class="kv-mismatch">${esc(t('kv.issues'))}${shownIssues.map((issue) => `<div><code>${esc(issue.configPath || issue.input || '—')}</code>: ${esc(issue.code)}</div>`).join('')}${issues.length > shownIssues.length ? `<div>+${issues.length - shownIssues.length}</div>` : ''}</div>` : ''}
      </div>`;
    return;
  }

  const rows = estimate.buffers.map((buffer) => `
    <tr>
      <td><b>${esc(buffer.label)}</b><code>${esc(buffer.id)}</code></td>
      <td>${esc(layerGroupLabel(buffer.layerGroup))}</td>
      <td class="num">${esc(fmtNum(buffer.elements))}</td>
      <td><code>${esc(buffer.dtype)}</code> × ${buffer.bytesPerElement}B</td>
      <td class="num">${esc(fmtNum(buffer.bytes))} B<br><span>${esc(fmtGB(buffer.gb))}</span></td>
      <td><code>${esc(buffer.formula)}</code></td>
    </tr>`).join('');
  const warnings = estimate.assurance?.warnings || [];
  const configWarning = warnings.find(
    (warning) => warning.code === 'config_mismatch',
  );
  const differences = configWarning?.differences || [];
  const shownDifferences = differences.slice(0, MAX_SHOWN_DIFFERENCES);
  const profileWarningHtml = estimate.assurance.status === 'warning' ? `
    <div class="kv-unknown">
      <b>${esc(t('kv.warning'))}</b>
      <div>${esc(t('kv.auditedCommit'))}<code>${esc(provenance.auditedCommitId || '—')}</code></div>
      ${shownDifferences.length ? `<div class="kv-mismatch">${esc(t('kv.configDifferences'))}${shownDifferences.map((difference) => `
        <div><code>${esc(difference.configPath)}</code>: <code>${esc(serialized(difference.auditedValue))}</code> → <code>${esc(serialized(difference.currentValue))}</code></div>`).join('')}${differences.length > shownDifferences.length ? `<div>+${differences.length - shownDifferences.length}</div>` : ''}</div>` : ''}
    </div>` : '';

  const profile = estimate.profile;
  const approximation = estimate.approximation;
  const identity = profile || approximation;
  const assumptions = approximation?.assumptions || [];
  const approximationWarningHtml = approximation ? `
    <div class="kv-approximate">
      <b>${esc(t('kv.genericWarning'))}</b>
      ${assumptions.length ? `<div>${esc(t('kv.assumptions'))}${assumptions.map((assumption) => `<div>• ${esc(assumptionLabel(assumption))}</div>`).join('')}</div>` : ''}
    </div>` : '';
  const status = estimate.assurance?.status || 'approximate';
  const statusClass = status === 'verified' ? 'verified' : status === 'approximate' ? 'approximate' : 'profile';
  container.innerHTML = `
    ${provenanceHtml}
    <div class="kv-profile-head">
      <div><b>${esc(identity.label)}</b><code>${esc(identity.id)}@${esc(identity.version)}</code></div>
      <span class="tag ${statusClass}">${esc(t(`kv.${status}`))}</span>
    </div>
    ${profileWarningHtml}
    ${approximationWarningHtml}
    ${profile ? `<div class="kv-layout-id">${esc(t('sum.kvLayout'))}<code>${esc(profile.layout.id)}@${esc(profile.layout.version)}</code></div>` : ''}
    <div class="kv-table-wrap"><table class="kv-table">
      <thead><tr><th>${esc(t('kv.buffer'))}</th><th>${esc(t('kv.layers'))}</th><th>${esc(t('kv.elements'))}</th><th>${esc(t('kv.dtype'))}</th><th>${esc(t('kv.bytes'))}</th><th>${esc(t('kv.formula'))}</th></tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr><th colspan="4">${esc(t('group.kv'))}</th><th class="num">${esc(fmtNum(estimate.buffers.reduce((sum, buffer) => sum + buffer.bytes, 0)))} B<br><span>${esc(fmtGB(estimate.vKV))}</span></th><th></th></tr></tfoot>
    </table></div>`;
}
