import test from 'node:test';
import assert from 'node:assert/strict';

import { buildTensorNameTree, groupRepeatedTensorSubtrees } from '../../src/tree/index.js';
import { renderTree } from '../../src/ui/treeView.js';

function renderTensorNames(names) {
  const tensors = names.map((name) => ({
    name,
    shape: [8, 8],
    dtype: 'BF16',
  }));
  const tree = groupRepeatedTensorSubtrees(buildTensorNameTree(tensors));
  const container = {
    innerHTML: '',
    querySelectorAll: () => [],
  };

  renderTree(container, tree, new Map());
  return container.innerHTML;
}

test('highlights the next path segment when an expanded branch reveals a compressed chain', () => {
  const html = renderTensorNames([
    'language_model.model.layers.0.mlp.down_proj.weight',
    'language_model.model.layers.0.mlp.up_proj.weight',
  ]);

  assert.match(
    html,
    /<span class="tensor-path-parent">language_model\.model\.layers\.0\.mlp\.<\/span><span class="tensor-path-current">down_proj<\/span><span class="tensor-path-descendant">\.weight<\/span>/,
  );
});
