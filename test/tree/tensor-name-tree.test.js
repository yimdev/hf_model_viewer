import test from 'node:test';
import assert from 'node:assert/strict';

import { buildTensorNameTree, groupRepeatedTensorSubtrees } from '../../src/tree/index.js';

function child(node, segment) {
  return node.children.find((candidate) => candidate.segment === segment);
}

function descendant(node, ...segments) {
  return segments.reduce((current, segment) => child(current, segment), node);
}

test('builds cumulative tensor-name prefixes from left to right', () => {
  const tensors = [
    { name: 'shared.weight', shape: [32, 16], dtype: 'BF16' },
    { name: 'decoder.final_layer_norm.weight', shape: [16], dtype: 'BF16' },
    { name: 'decoder.block.0.layer.0.layer_norm.weight', shape: [16], dtype: 'BF16' },
    { name: 'decoder.block.0.layer.0.SelfAttention.q.weight', shape: [16, 16], dtype: 'BF16' },
    { name: 'decoder.block.0.layer.1.EncDecAttention.q.weight', shape: [16, 16], dtype: 'BF16' },
    { name: 'decoder.block.1.layer.0.SelfAttention.q.weight', shape: [16, 16], dtype: 'BF16' },
  ];

  const root = buildTensorNameTree(tensors);
  const decoder = child(root, 'decoder');
  const blocks = child(decoder, 'block');
  const block0 = child(blocks, '0');
  const block1 = child(blocks, '1');
  const block0Layers = child(block0, 'layer');
  const block0Layer0 = child(block0Layers, '0');

  assert.equal(root.prefix, '');
  assert.equal(root.directChildCount, 2);
  assert.equal(decoder.prefix, 'decoder');
  assert.equal(decoder.directChildCount, 2);
  assert.equal(blocks.prefix, 'decoder.block');
  assert.deepEqual(blocks.children.map((node) => node.segment), ['0', '1']);
  assert.equal(block0.numeric, true);
  assert.equal(block1.numeric, true);
  assert.equal(block0.prefix, 'decoder.block.0');
  assert.equal(block0Layers.directChildCount, 2);
  assert.equal(block0Layer0.prefix, 'decoder.block.0.layer.0');
  assert.equal(block0Layer0.directChildCount, 2);
});

test('preserves the first-seen order of path segments', () => {
  const root = buildTensorNameTree([
    { name: 'model.layers.10.weight', shape: [1], dtype: 'F32' },
    { name: 'model.layers.2.weight', shape: [1], dtype: 'F32' },
  ]);

  const layers = child(child(root, 'model'), 'layers');

  assert.deepEqual(layers.children.map((node) => node.segment), ['10', '2']);
});

test('keeps leaf metadata without assigning architecture semantics', () => {
  const tensor = {
    name: 'model.layers.0.self_attn.q_proj.weight',
    shape: [16, 16],
    dtype: 'BF16',
    params: 256,
  };
  const expertTensor = {
    name: 'model.layers.0.ffn.experts.1.up_proj.weight',
    shape: [16, 32],
    dtype: 'BF16',
  };

  const root = buildTensorNameTree([tensor, expertTensor]);
  const weight = descendant(root, 'model', 'layers', '0', 'self_attn', 'q_proj', 'weight');
  const expertWeight = descendant(root, 'model', 'layers', '0', 'ffn', 'experts', '1', 'up_proj', 'weight');

  assert.equal(weight.tensors[0], tensor);
  assert.equal(expertWeight.tensors[0], expertTensor);
  assert.deepEqual(weight.children, []);
  assert.deepEqual(Object.keys(weight).sort(), [
    'children',
    'directChildCount',
    'numeric',
    'prefix',
    'segment',
    'tensors',
  ]);
});

test('builds Repeated Tensor Groups from matching Numeric Path Branches', () => {
  const tensors = [
    { name: 'model.layers.0.self_attn.q.weight', shape: [8, 8], dtype: 'BF16' },
    { name: 'model.layers.0.self_attn.k.weight', shape: [8, 8], dtype: 'BF16' },
    { name: 'model.layers.0.mlp.up.weight', shape: [16, 8], dtype: 'BF16' },
    { name: 'model.layers.1.self_attn.q.weight', shape: [8, 8], dtype: 'BF16' },
    { name: 'model.layers.1.self_attn.k.weight', shape: [8, 8], dtype: 'BF16' },
    { name: 'model.layers.1.mlp.up.weight', shape: [16, 8], dtype: 'BF16' },
    { name: 'model.layers.2.self_attn.q.weight', shape: [4, 8], dtype: 'BF16' },
    { name: 'model.layers.2.self_attn.k.weight', shape: [8, 8], dtype: 'BF16' },
    { name: 'model.layers.2.mlp.up.weight', shape: [16, 8], dtype: 'BF16' },
    { name: 'model.layers.3.self_attn.q.weight', shape: [8, 8], dtype: 'F32' },
    { name: 'model.layers.3.self_attn.k.weight', shape: [8, 8], dtype: 'BF16' },
    { name: 'model.layers.3.mlp.up.weight', shape: [16, 8], dtype: 'BF16' },
  ];

  const grouped = groupRepeatedTensorSubtrees(buildTensorNameTree(tensors));
  const layers = descendant(grouped, 'model', 'layers');

  assert.equal(layers.directChildCount, 4);
  assert.equal(layers.children[0].repeatCount, 2);
  assert.deepEqual(layers.children[0].repeatIds, ['0', '1']);
  assert.deepEqual(layers.children[0].members.map((node) => node.prefix), [
    'model.layers.0',
    'model.layers.1',
  ]);
  assert.deepEqual(layers.children.slice(1).map((node) => node.repeatCount), [1, 1]);

  const repeatedAttention = descendant(layers.children[0], 'self_attn');
  const repeatedWeight = descendant(repeatedAttention, 'q', 'weight');
  assert.equal(repeatedAttention.directChildCount, 2);
  assert.equal(repeatedWeight.repeatCount, 1);
  assert.deepEqual(repeatedWeight.tensors.map((tensor) => tensor.name), [
    'model.layers.0.self_attn.q.weight',
    'model.layers.1.self_attn.q.weight',
  ]);
});

test('reports nested Repeated Tensor Groups within local Numeric Path Branches', () => {
  const tensors = [];
  for (let outerIndex = 0; outerIndex < 2; outerIndex += 1) {
    for (let innerIndex = 0; innerIndex < 3; innerIndex += 1) {
      tensors.push({
        name: `model.layers.${outerIndex}.mlp.experts.${innerIndex}.down.weight`,
        shape: [8, 8],
        dtype: 'BF16',
      });
    }
  }

  const grouped = groupRepeatedTensorSubtrees(buildTensorNameTree(tensors));
  const outerBranches = descendant(grouped, 'model', 'layers');
  const repeatedOuterBranch = outerBranches.children[0];
  const innerBranches = descendant(repeatedOuterBranch, 'mlp', 'experts');
  const repeatedInnerBranch = innerBranches.children[0];
  const nestedWeight = descendant(repeatedInnerBranch, 'down', 'weight');

  assert.equal(outerBranches.directChildCount, 2);
  assert.equal(repeatedOuterBranch.repeatCount, 2);
  assert.deepEqual(repeatedOuterBranch.repeatIds, ['0', '1']);
  assert.equal(innerBranches.directChildCount, 3);
  assert.equal(repeatedInnerBranch.repeatCount, 3);
  assert.deepEqual(repeatedInnerBranch.repeatIds, ['0', '1', '2']);
  assert.equal(repeatedInnerBranch.members.length, 6);
  assert.equal(nestedWeight.repeatCount, 1);
  assert.equal(nestedWeight.tensors.length, 6);
});
