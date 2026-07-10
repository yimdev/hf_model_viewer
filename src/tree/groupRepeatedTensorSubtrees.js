function tensorSignature(tensor) {
  return JSON.stringify([tensor.shape, tensor.dtype]);
}

function normalizedSegment(node) {
  return node.numeric ? ['numeric'] : ['named', node.segment];
}

function createSignatureReader() {
  const cache = new WeakMap();

  const signatureFor = (node) => {
    if (cache.has(node)) return cache.get(node);

    const tensorSignatures = node.tensors.map(tensorSignature).sort();
    const childSignatures = node.children.map(signatureFor).sort();
    const signature = JSON.stringify([
      normalizedSegment(node),
      tensorSignatures,
      childSignatures,
    ]);
    cache.set(node, signature);
    return signature;
  };

  return signatureFor;
}

function groupMembers(members, signatureFor) {
  const representative = members[0];
  const childBuckets = new Map();

  for (const member of members) {
    for (const child of member.children) {
      const signature = signatureFor(child);
      if (!childBuckets.has(signature)) childBuckets.set(signature, []);
      childBuckets.get(signature).push(child);
    }
  }

  const children = [...childBuckets.values()].map((childrenWithSameStructure) => (
    groupMembers(childrenWithSameStructure, signatureFor)
  ));

  return {
    segment: representative.segment,
    prefix: representative.prefix,
    numeric: representative.numeric,
    directChildCount: children.length,
    children,
    tensors: members.flatMap((member) => member.tensors),
    repeatCount: members.length,
    members,
  };
}

export function groupRepeatedTensorSubtrees(tree) {
  return groupMembers([tree], createSignatureReader());
}
