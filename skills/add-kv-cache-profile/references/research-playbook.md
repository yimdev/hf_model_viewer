# KV Cache research playbook

Use this reference whenever a contribution introduces or changes a KV Cache Layout.

## Evidence hierarchy

Prefer evidence in this order:

1. Pinned official model configuration and model repository code.
2. Pinned upstream library code actually selected by `auto_map`, `architectures`, or the official loading path.
3. Pinned official kernels or inference implementation that allocate and update cache state.
4. The model paper or technical report for algorithm intent.
5. Independent runtime implementations for cross-checking shapes and transitions.

A paper-level equation is insufficient when runtime code stores additional state. A tensor shape in model weights is insufficient evidence for runtime cache semantics. A third-party implementation can expose a question but cannot by itself establish the audited Layout when authoritative code is available.

## Pin the source graph

Resolve a Hugging Face branch or tag through the model API and retain the returned 40-character SHA. Fetch files through commit-pinned URLs:

```text
https://huggingface.co/api/models/<org>/<repo>/revision/<revision>
https://huggingface.co/<org>/<repo>/resolve/<commit>/<path>
```

If the model delegates to another repository or package, pin that source independently. Record the relationship between the model commit, declared library version, and implementation commit. Treat moving branches, generated documentation, search snippets, and unpinned raw URLs as discovery aids only.

Start with `config.json`, repository file listings, `auto_map`, `architectures`, modeling files, cache utilities, attention modules, and custom kernels. Search for `past_key_values`, `use_cache`, `cache`, `update`, `state`, `conv`, `recurrent`, `compress`, `index`, `sliding`, and dtype conversions.

## Trace the state machine

Follow both prefill and decode:

1. Locate cache construction and initial shapes.
2. Trace the first prefill write.
3. Trace one decode update without assuming it is append-only.
4. Identify truncation, windows, chunk boundaries, compression residuals, and per-sequence state.
5. Determine which dtype each stored buffer actually uses after casts.
6. Determine whether empty sequences allocate state and whether ragged sequences interact.
7. Separate base-model state from optional MTP/speculation, vision preprocessing, workspace, and capacity reservation.
8. Identify the default cache implementation selected by the pinned framework version and separate its semantic payload from selectable static, quantized, offloaded, or paged allocation policies.

For hybrid models, audit the whole model. Do not compose a Profile from familiar attention labels without tracing layer selection and every mechanism's persistent state.

Record runtime override limits explicitly. Repository config can establish an audited cache dtype only when pinned code shows the stored state follows that value; loader or engine overrides that are absent from model provenance remain runtime assumptions.

## Evidence ledger

Maintain a table in working notes or the issue/PR description:

| State | Persistent shape | DType | Growth/lifetime | Config inputs | Pinned source | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Example key cache | `N × L × Hkv × S × D` | BF16 | grows per token | layers, heads, head dim | file and lines | include |

The ledger is complete when every allocation reachable from cache construction/update appears once and every decision has evidence. Preserve useful source links and derivations in the contribution handoff so reviewers can reproduce the audit.

## Independent arithmetic

Derive elements before bytes:

```text
elements(buffer) = product or sum of semantic dimensions
bytes(buffer) = elements(buffer) × stored dtype width
totalBytes = sum(bytes(buffer))
```

Test ordinary and boundary values. For a window or chunk size `K`, include `0`, `1`, `K-1`, `K`, and `K+1` where valid. For per-active-sequence state, compare an empty sequence, one active sequence, multiple active sequences, and a ragged batch. For layer schedules, verify the exact selected indices and counts. For compressed state, independently sum individual sequences rather than applying a uniform-batch shortcut.

The repository's `sequenceLengths` input accounts for independently resident or logically packed sequences by summing per-sequence Layout results. It does not describe the capacity of one rectangular padded framework tensor unless a Profile explicitly defines that policy. Preserve this distinction in formulas, notes, and the audit handoff.

Use exact integer arithmetic within JavaScript's safe-integer range. Treat unsafe intermediate or total values as an unknown calculation rather than a numeric estimate.
