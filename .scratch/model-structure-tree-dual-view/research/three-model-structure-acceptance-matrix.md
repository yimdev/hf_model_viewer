# Three-Model Structural Acceptance Corpus

## Purpose

This task asset identifies the smallest decision-useful tensor scenarios from the existing Hy3, GLM 5.2, and DeepSeek V4 Pro profile fixtures. It supports later recognition-contract, data-contract, prototype, and acceptance-test decisions; it is not production configuration and must not become a per-model presentation catalog.

The machine-readable corpus is [three-model-structure-acceptance.json](../fixtures/three-model-structure-acceptance.json). The broader checkpoint-path survey is [Repetition Axes in Model Tensor Paths](./tensor-path-repetition-axes.md).

## Provenance boundary

The corpus separates two kinds of input:

- `existing_fixture`: an exact `{ name, shape, dtype }` subset of a factory in `test/vram/profile-fixtures.js`.
- `contract_boundary_synthetic`: a deliberately small test-only scenario filling a domain-invariant gap in the existing fixtures. It is not evidence about any real checkpoint and must never affect KV Cache Profile matching.

The Structure Presentation Heuristic remains tensor-metadata-only. Config values below explain fixture coverage but are not expected recognition inputs.

## Full-fixture expectation matrix

| Pattern | Existing fixture evidence | Full-fixture expectation | Minimal corpus scenario |
| --- | --- | --- | --- |
| Uniform repeated layers | Hy3 emits `self_attn.q_proj.weight` with BF16 `[8192, 4096]` for `model.layers:0–80`. | One aggregate spans 81 sampled IDs; the unique `eh_proj` suffix spans only `model.layers:80`. The presentation layer must not silently drop the extra suffix. | `hy3_uniform_layers_and_unique_suffix` |
| Dense key with an extra terminal ID | GLM 5.2 emits `kv_a_proj_with_mqa.weight` for `model.layers:0–78`, while `kv_b_proj.weight` stops at `model.layers:77`. | Separate aggregate membership sets are 79 and 78 IDs even though both keys share an axis. Missing membership is information, not an error. | `glm52_sparse_indexer_bf16` |
| Sparse layer subset | GLM 5.2 indexer tensors occur on 21 IDs: `0, 1, 2`, then `6–74 / step 4`. | The aggregate exposes the exact sparse membership set; it must not expand the range to every intervening Layer ID. | `glm52_sparse_indexer_bf16` |
| Quantization companions | GLM 5.2 FP8 emits F8 weights and F32 `weight_scale_inv` tensors with different Shapes. | Base weights and companion tensors form distinct aggregates; each retains its own Shape, DType, parameters, and bytes. | `glm52_fp8_companion_tensors` |
| Interleaved same-key Shape split | DeepSeek V4 Pro emits `compressor.ape` as F32 `[128, 512]` on 31 IDs (`0, 1`, then odd `3–59`) and F32 `[4, 1024]` on 30 even IDs (`2–60`). | The same normalized key produces two aggregates because ordered Shape differs. Their Layer ID summaries expose the interleaving. | `deepseek_interleaved_shapes_and_mtp_axis` |
| Sparse interleaved key | DeepSeek V4 Pro indexer tensors occur only on the 30 even IDs `2–60`. | Indexer aggregates keep the even membership set and remain independent from the compressor aggregate. | `deepseek_interleaved_shapes_and_mtp_axis` |
| Auxiliary axis | DeepSeek V4 Pro contains `mtp.0.attn.wkv.weight` alongside `layers.<id>.attn.wkv.weight`. | `mtp:0` and `layers:0` are different Axis-Qualified Layer IDs and never merge, despite equal suffix, Shape, and DType. | `deepseek_interleaved_shapes_and_mtp_axis` |
| Routed-expert leaf and quantization leaf | The DeepSeek fixture contains only `layers.0.ffn.experts.0.w1.weight` and `.scale`. | The fixture proves path parsing and distinct leaf identity, but not repeated-expert collapse or per-layer multiplicity. | `deepseek_existing_single_expert_evidence` |
| Repeated routed experts | Not covered by the existing fixtures. | Two identical explicit Expert IDs share one representative aggregate with actual multiplicity two. | `synthetic_domain_boundary_gaps` |
| Shared expert and router | Not covered by the existing fixtures. | Shared-expert and router tensors remain ordinary layer tensors and receive no routed-expert multiplier. | `synthetic_domain_boundary_gaps` |
| Model-Level Tensor | Not covered by the existing fixtures. | Embedding and LM Head remain visible outside layer aggregation and receive no fabricated Layer ID. | `synthetic_domain_boundary_gaps` |
| Same-key DType split | The BF16 and FP8 GLM factories are separate checkpoint variants, so neither contains an internal DType drift. | Equal normalized key and Shape with different on-disk DType produce separate aggregates. | `synthetic_domain_boundary_gaps` |

## Minimality rationale

- Hy3 uses three far-apart IDs plus one unique suffix to prove uniform grouping without copying all 652 fixture tensors.
- GLM 5.2 uses both present and absent membership boundaries: a terminal extra ID, a dense key that stops earlier, and two representatives from the sparse indexer set.
- DeepSeek V4 Pro uses two representatives from each interleaved Shape family, a sparse indexer leaf, equal main/MTP suffixes, and the existing single-expert leaves.
- Synthetic entries exist only where the three fixtures cannot establish a required domain invariant. Keeping them in one explicitly marked scenario prevents accidental promotion to model evidence.

## Downstream requirements exposed by the corpus

1. Recognition tests need to consume both this corpus and the broader 22-case path survey; the three profile fixtures alone are insufficient for generic path recognition.
2. Aggregate membership is a set of Axis-Qualified Layer IDs, not a count or min/max range.
3. The data contract must represent a singleton aggregate, sparse membership, interleaved membership, explicit expert multiplicity, and Model-Level Tensors without special model branches.
4. The prototype must show the two DeepSeek `compressor.ape` aggregates adjacent enough for their Shape and Layer ID patterns to be compared.
5. Parameter and byte reconciliation must include every companion tensor and every model-level tensor exactly once.

No new domain term is resolved by this task, so `CONTEXT.md` requires no change. No ADR is warranted: this corpus records reversible test evidence rather than a hard-to-reverse architectural decision.
