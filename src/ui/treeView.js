/* ui/treeView.js — Left-to-right Tensor Name Tree
 * ------------------------------------------------------------
 * Branches show cumulative dot-delimited prefixes. Single-child chains are
 * compressed into one visible row, while true branches remain independently
 * collapsible. A Numeric Path Branch starts closed;
 * named branches start open. Leaves retain the original tensor metadata.
 * ------------------------------------------------------------ */

import { fmtNum, fmtBytesGB, esc } from './format.js';
import { tensorParams } from '../tree/buildTree.js';
import { t } from '../i18n.js';

function pCount(tensor) {
  return tensor.params != null ? tensor.params : tensorParams(tensor.shape);
}

function effBpp(name, effectiveBppByTensorName) {
  return effectiveBppByTensorName.get(name) ?? 2;
}

function tensorByte(tensor, effectiveBppByTensorName) {
  return pCount(tensor) * effBpp(tensor.name, effectiveBppByTensorName);
}

function pathLabel(prefix) {
  const split = prefix.lastIndexOf('.');
  const parent = split < 0 ? '' : prefix.slice(0, split + 1);
  const current = split < 0 ? prefix : prefix.slice(split + 1);
  return `<span class="tensor-path-parent">${esc(parent)}</span><span class="tensor-path-current">${esc(current)}</span>`;
}

function collapseSingleChild(node) {
  let current = node;
  let containsNumeric = node.numeric;

  while (current.tensors.length === 0 && current.children.length === 1) {
    const next = current.children[0];
    const nextContainsNumeric = containsNumeric || next.numeric;
    if (nextContainsNumeric && next.children.length === 0) break;
    current = next;
    containsNumeric ||= current.numeric;
  }

  return { node: current, containsNumeric };
}

function collectSubtreeStats(node, effectiveBppByTensorName, visitor) {
  let params = 0;
  let bytes = 0;
  let tensors = 0;

  for (const tensor of node.tensors) {
    const tensorBytes = tensorByte(tensor, effectiveBppByTensorName);
    params += pCount(tensor);
    bytes += tensorBytes;
    tensors += 1;
    visitor.tensor(tensor, tensorBytes);
  }
  for (const child of node.children) {
    const childStats = collectSubtreeStats(child, effectiveBppByTensorName, visitor);
    params += childStats.params;
    bytes += childStats.bytes;
    tensors += childStats.tensors;
  }

  const stats = { params, bytes, tensors };
  visitor.node(node, stats);
  return stats;
}

function createStats(tree, effectiveBppByTensorName) {
  const cache = new WeakMap();
  collectSubtreeStats(tree, effectiveBppByTensorName, {
    tensor() {},
    node(node, stats) {
      cache.set(node, stats);
    },
  });

  return (node) => cache.get(node);
}

function tensorLeaf(tensor, effectiveBppByTensorName, depth) {
  return `
    <div class="tensor-leaf" style="--tree-depth:${depth}">
      <div class="tensor-path">${pathLabel(tensor.name)}</div>
      <code class="tensor-shape">${esc(tensor.shape.join('×'))}</code>
      <code class="tensor-dtype">${esc(tensor.dtype)}</code>
      <span class="tensor-number">${fmtNum(pCount(tensor))}</span>
      <span class="tensor-number byte-cell" data-key="t:${esc(tensor.name)}">${fmtBytesGB(tensorByte(tensor, effectiveBppByTensorName))}</span>
    </div>`;
}

function renderVisibleNode(start, effectiveBppByTensorName, statsFor, depth) {
  const collapsed = collapseSingleChild(start);
  const node = collapsed.node;

  if (node.children.length === 0) {
    return node.tensors.map((tensor) => tensorLeaf(tensor, effectiveBppByTensorName, depth)).join('');
  }

  const stats = statsFor(node);
  const directEntries = node.directChildCount;
  const open = collapsed.containsNumeric ? '' : ' open';
  const numericClass = collapsed.containsNumeric ? ' numeric-branch' : '';
  const terminalRows = node.tensors.map((tensor) => tensorLeaf(tensor, effectiveBppByTensorName, depth + 1)).join('');
  const childRows = node.children
    .map((child) => renderVisibleNode(child, effectiveBppByTensorName, statsFor, depth + 1))
    .join('');

  return `
    <details class="tensor-branch${numericClass}"${open}>
      <summary style="--tree-depth:${depth}">
        <span class="tensor-path">${pathLabel(node.prefix)}</span>
        <span class="tensor-child-count">(${directEntries})</span>
        <span class="tensor-chevron" aria-hidden="true"></span>
        <span class="tensor-branch-meta">${fmtNum(stats.params)} ${esc(t('tree.paramsUnit'))} · <span class="byte-cell" data-key="p:${esc(node.prefix)}">${fmtBytesGB(stats.bytes)}</span></span>
      </summary>
      ${terminalRows}${childRows}
    </details>`;
}

function buildByteMap(tree, effectiveBppByTensorName) {
  const bytes = new Map();
  collectSubtreeStats(tree, effectiveBppByTensorName, {
    tensor(tensor, tensorBytes) {
      bytes.set(`t:${tensor.name}`, tensorBytes);
    },
    node(node, stats) {
      bytes.set(`p:${node.prefix}`, stats.bytes);
    },
  });
  return bytes;
}

export function renderTree(container, tree, effectiveBppByTensorName) {
  const statsFor = createStats(tree, effectiveBppByTensorName);
  const rows = tree.children
    .map((child) => renderVisibleNode(child, effectiveBppByTensorName, statsFor, 0))
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
}

/** Refresh byte totals without rebuilding the DOM, preserving open branches. */
export function updateTreeBytes(container, tree, effectiveBppByTensorName) {
  const byteMap = buildByteMap(tree, effectiveBppByTensorName);
  container.querySelectorAll('.byte-cell').forEach((element) => {
    const key = element.getAttribute('data-key');
    if (byteMap.has(key)) element.textContent = fmtBytesGB(byteMap.get(key));
  });
}
