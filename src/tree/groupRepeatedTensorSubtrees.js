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

function bucketChildren(children, signatureFor) {
  const buckets = new Map();
  for (const child of children) {
    const signature = signatureFor(child);
    if (!buckets.has(signature)) buckets.set(signature, []);
    buckets.get(signature).push(child);
  }
  return buckets;
}

function compareNumericPathSegmentsAscending(left, right) {
  if (left.length !== right.length) return left.length - right.length;
  if (left === right) return 0;
  return left < right ? -1 : 1;
}

function groupMembers(members, signatureFor, repeatCount = 1, repeatIds = []) {
  const representative = members[0];
  const localBuckets = bucketChildren(representative.children, signatureFor);
  const children = [...localBuckets.entries()].map(([signature, localMembers]) => {
    const allMembers = members.flatMap((member) => (
      member.children.filter((child) => signatureFor(child) === signature)
    ));
    const localIds = localMembers.every((child) => child.numeric)
      ? localMembers
        .map((child) => child.segment)
        .sort(compareNumericPathSegmentsAscending)
      : [];
    return groupMembers(allMembers, signatureFor, localMembers.length, localIds);
  });

  return {
    segment: representative.segment,
    prefix: representative.prefix,
    numeric: representative.numeric,
    directChildCount: representative.directChildCount,
    children,
    tensors: members.flatMap((member) => member.tensors),
    repeatCount,
    repeatIds,
    members,
  };
}

export function groupRepeatedTensorSubtrees(tree) {
  return groupMembers([tree], createSignatureReader());
}
