/* ui/treeView.js — Left-to-right Tensor Name Tree with Repeated Tensor Groups
 * ------------------------------------------------------------
 * Branches show cumulative dot-delimited prefixes. Single-child chains are
 * compressed into one visible row, while true branches remain independently
 * collapsible. Every branch starts closed so one interaction reveals one
 * additional level. Bottom-up equivalent subtrees share one visible Repeated
 * Tensor Group while statistics retain every original tensor.
 * ------------------------------------------------------------ */

import { fmtNum, fmtBytesGB, esc } from './format.js';
import { t } from '../i18n.js';

function collectTensorStats(tensors) {
  let params = 0;
  let bytes = 0;

  for (const tensor of tensors) {
    params += tensor.params;
    bytes += tensor.weightBytes;
  }

  return { params, bytes, tensors: tensors.length };
}

function pathLabel(prefix, currentNode) {
  const currentEnd = currentNode.prefix.length;
  const currentStart = currentEnd - currentNode.segment.length;
  const parent = prefix.slice(0, currentStart);
  const current = prefix.slice(currentStart, currentEnd);
  const descendant = prefix.slice(currentEnd);
  return `<span class="tensor-path-parent">${esc(parent)}</span><span class="tensor-path-current">${esc(current)}</span><span class="tensor-path-descendant">${esc(descendant)}</span>`;
}

function collapseSingleChild(node) {
  let current = node;
  let containsNumeric = node.numeric;
  let repetitionNode = node.repeatCount > 1 ? node : null;

  while (current.tensors.length === 0 && current.children.length === 1) {
    const next = current.children[0];
    const nextContainsNumeric = containsNumeric || next.numeric;
    if (nextContainsNumeric && next.children.length === 0) break;
    if (repetitionNode && next.repeatCount > 1) break;
    current = next;
    containsNumeric ||= current.numeric;
    if (current.repeatCount > 1) repetitionNode = current;
  }

  return { node: current, containsNumeric, repetitionNode };
}

function repeatBadge(node) {
  const ids = node.repeatIds.join(', ');
  return node.repeatCount > 1
    ? `<span class="tensor-repeat" title="${esc(t('tree.repeatIds', { count: node.repeatCount, ids }))}">×${fmtNum(node.repeatCount)}</span>`
    : '';
}

function tensorLeaf(node, depth, currentNode = node) {
  const tensor = node.tensors[0];
  const stats = collectTensorStats(node.tensors);
  return `
    <div class="tensor-leaf" style="--tree-depth:${depth}">
      <div class="tensor-path">${pathLabel(tensor.name, currentNode)}${repeatBadge(node)}</div>
      <code class="tensor-shape">${esc(tensor.shape.join('×'))}</code>
      <code class="tensor-dtype">${esc(tensor.dtype)}</code>
      <span class="tensor-number">${fmtNum(stats.params)}</span>
      <span class="tensor-number byte-cell">${fmtBytesGB(stats.bytes)}</span>
    </div>`;
}

function renderVisibleNode(start, depth) {
  const collapsed = collapseSingleChild(start);
  const node = collapsed.node;

  if (node.children.length === 0) {
    return tensorLeaf(node, depth, start);
  }

  const stats = node.stats;
  const directEntries = node.directChildCount;
  const numericClass = collapsed.containsNumeric ? ' numeric-branch' : '';
  const repetitionNode = collapsed.repetitionNode || node;
  const terminalRows = node.tensors.length > 0
    ? tensorLeaf(node, depth + 1)
    : '';
  const childRows = node.children
    .map((child) => renderVisibleNode(child, depth + 1))
    .join('');

  return `
    <details class="tensor-branch${numericClass}">
      <summary style="--tree-depth:${depth}">
        <span class="tensor-path">${pathLabel(node.prefix, start)}</span>
        <span class="tensor-child-count">(${directEntries})</span>
        ${repeatBadge(repetitionNode)}
        <span class="tensor-chevron" aria-hidden="true"></span>
        <span class="tensor-branch-meta">${fmtNum(stats.params)} ${esc(t('tree.paramsUnit'))} · <span class="byte-cell">${fmtBytesGB(stats.bytes)}</span></span>
      </summary>
      ${terminalRows}${childRows}
    </details>`;
}

export function renderTree(container, tree) {
  const rows = tree.children
    .map((child) => renderVisibleNode(child, 0))
    .join('');
  container.innerHTML = `
    <p class="tensor-tree-hint">${esc(t('tree.prefixHint'))}</p>
    <div class="tensor-tree-header">
      <span>${esc(t('tree.col.op'))}</span>
      <span>${esc(t('tree.col.shape'))}</span>
      <span>${esc(t('tree.col.dtype'))}</span>
      <span>${esc(t('tree.col.params'))}</span>
      <span>${esc(t('tree.col.vram'))}</span>
    </div>
    <div class="tensor-name-tree">${rows}</div>`;

  container.querySelectorAll('details.tensor-branch').forEach((branch) => {
    branch.addEventListener('toggle', () => {
      if (branch.open) return;
      branch.querySelectorAll('details.tensor-branch[open]').forEach((descendant) => {
        descendant.open = false;
      });
    });
  });
}
