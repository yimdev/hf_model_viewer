# Repetition Axes in Model Tensor Paths

## Research question

Which path patterns in real checkpoints provide reliable evidence for a layer repetition axis or a routed-expert repetition axis, and which numeric path segments must remain ordinary parts of a tensor key?

This note supports the dual-view model-structure effort. It surveys architectures beyond the three current KV-cache profiles (GLM 5.2, DeepSeek V4 Pro, and Hy3). It records checkpoint evidence first, then proposes conservative boundaries for a generic, model-independent presentation heuristic. It does not define architecture-specific display rules.

## Method and source boundary

- Sources are first-party model repositories, official checkpoint indexes/configurations, official Hugging Face Transformers source, and the Safetensors specification.
- Tensor metadata was read from Safetensors headers with HTTP range requests. The format makes tensor name, DType, and ordered Shape available in the header without downloading tensor data; this is an intended use of the format ([Safetensors format specification](https://github.com/huggingface/safetensors#format), [official metadata-parsing guide](https://huggingface.co/docs/safetensors/metadata_parsing)).
- Configuration values are used only to corroborate what an index contains. The recommended presentation heuristic remains based on tensor names, Shapes, and DTypes, not model identity or a manually maintained model catalog.
- Observed facts and recommendations are separated below. A recommendation is not evidence that every model follows a particular spelling.

## Observed evidence

### A layer axis has a qualified path, not just an integer

Qwen2.5-VL contains two independent repeated stacks in one checkpoint:

- `model.layers.0.self_attn.q_proj.weight` through `model.layers.35...`
- `visual.blocks.0.attn.qkv.weight` through `visual.blocks.31...`

The text configuration declares 36 hidden layers, while the vision configuration declares depth 32. The checkpoint also contains non-repeated vision tensors such as `visual.patch_embed.proj.weight` and `visual.merger.mlp.0.weight` ([Qwen2.5-VL config](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json), [checkpoint index](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/model.safetensors.index.json)). Header inspection confirms that the two repeated stacks have different Shapes even when both are attention projections: text `q_proj` is BF16 `[2048, 2048]`, while vision `qkv` is BF16 `[3840, 1280]` ([first checkpoint shard](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/model-00001-of-00002.safetensors)).

FLAN-T5 likewise has independent `encoder.block.<id>` and `decoder.block.<id>` stacks. Its config declares 24 blocks in each stack, and the checkpoint index carries both prefixes ([FLAN-T5-XL config](https://huggingface.co/google/flan-t5-xl/blob/main/config.json), [checkpoint index](https://huggingface.co/google/flan-t5-xl/blob/main/model.safetensors.index.json)).

Qwen3.5-35B-A3B adds a main `model.language_model.layers.<id>` stack and a separate `mtp.layers.0` auxiliary stack. The config declares 40 main layers and one MTP hidden layer ([Qwen3.5-35B-A3B config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json), [checkpoint index](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/model.safetensors.index.json)). The observed singleton `mtp.layers.0` is important: sibling-count evidence alone cannot recognize every legitimate axis.

**Observed conclusion:** equal bare integers do not imply the same layer. `model.layers:0`, `visual.blocks:0`, `encoder.block:0`, `decoder.block:0`, `model.language_model.layers:0`, and `mtp.layers:0` are different qualified identities.

### Numeric segments are not uniformly layer IDs

FLAN-T5 exposes the clearest counterexample to a regex such as `/\.(\d+)\./`:

```text
decoder.block.0.layer.0.SelfAttention.q.weight
decoder.block.0.layer.1.EncDecAttention.q.weight
decoder.block.0.layer.2.DenseReluDense.wi_0.weight
```

Here `decoder.block.0` identifies the repeated decoder block. The inner `layer.0`, `layer.1`, and `layer.2` values are fixed sublayer positions with different roles. Header metadata further distinguishes their Shapes: the two attention query weights are F32 `[2048, 2048]`, while `DenseReluDense.wi_0.weight` is F32 `[5120, 2048]` ([FLAN-T5 checkpoint index](https://huggingface.co/google/flan-t5-xl/blob/main/model.safetensors.index.json), [first checkpoint shard](https://huggingface.co/google/flan-t5-xl/blob/main/model-00001-of-00002.safetensors)). The `layer.<slot>` segment must remain in the normalized tensor key after only `block.<id>` is normalized.

Qwen2.5-VL has another fixed-position use:

```text
visual.merger.mlp.0.weight  # BF16 [5120, 5120]
visual.merger.mlp.2.weight  # BF16 [2048, 5120]
```

These are two positions inside one model-level merger MLP, not two repeated model layers. The differing Shapes are direct negative evidence for treating them as one repetition family ([Qwen2.5-VL first checkpoint shard](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/model-00001-of-00002.safetensors)).

Switch Transformers shows that even expert IDs are not always standalone numeric segments. Its official implementation stores experts in a `ModuleDict` under keys `expert_0`, `expert_1`, and so on, rather than `experts.0`, `experts.1` ([Transformers `SwitchTransformersExperts` source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L156-L173)). The first-party checkpoint config declares eight experts ([Switch Base 8 config](https://huggingface.co/google/switch-base-8/blob/main/config.json)).

**Observed conclusion:** neither “numeric segment” nor the lexical token `layer` is sufficient by itself. Position, qualified prefix, sibling structure, and Shape/DType evidence matter.

### Routed experts, shared experts, and routers are distinct structures

Mixtral serializes routed experts explicitly:

```text
model.layers.0.block_sparse_moe.experts.0.w1.weight
model.layers.0.block_sparse_moe.experts.7.w1.weight
model.layers.0.block_sparse_moe.gate.weight
```

The config declares 32 layers and 8 local experts. Header metadata shows expert 0 and expert 7 `w1` tensors are both BF16 `[14336, 4096]`; the router gate is BF16 `[8, 4096]` and has no Expert ID ([Mixtral config](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json), [checkpoint index](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/model.safetensors.index.json), [first checkpoint shard](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/model-00001-of-00019.safetensors)). This is direct evidence for two independent repetition axes in one path: layer and routed expert.

DeepSeek-V3 serializes routed experts, a shared expert, and a router side by side:

```text
model.layers.3.mlp.experts.0.gate_proj.weight
model.layers.3.mlp.experts.1.gate_proj.weight
model.layers.3.mlp.shared_experts.gate_proj.weight
model.layers.3.mlp.gate.weight
```

The config declares 256 routed experts, one shared expert, and three initial dense layers before the MoE layers. In the checkpoint header, both routed expert samples and the shared-expert projection are F8_E4M3 `[2048, 7168]`, while the router is BF16 `[256, 7168]` ([DeepSeek-V3 config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json), [checkpoint index](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/model.safetensors.index.json), [first checkpoint shard](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/model-00001-of-000163.safetensors)). Shape equality does not make the shared expert a routed-expert instance; the path semantics and absence of an Expert ID keep it separate.

Qwen3.5-35B-A3B demonstrates a third representation: the main language-model layers pack all routed experts into single tensors with no numeric Expert ID:

```text
model.language_model.layers.0.mlp.experts.gate_up_proj  # BF16 [256, 1024, 2048]
model.language_model.layers.0.mlp.experts.down_proj     # BF16 [256, 2048, 512]
```

The same checkpoint's MTP layer serializes experts individually as `mtp.layers.0.mlp.experts.0.down_proj.weight`, `...experts.1...`, and so on. Header metadata shows the packed main-layer tensor's leading dimension is 256, while each MTP expert projection is BF16 `[2048, 512]` ([Qwen3.5-35B-A3B checkpoint index](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/model.safetensors.index.json), [packed gate/up shard](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/model.safetensors-00006-of-00014.safetensors), [packed down shard](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/model.safetensors-00012-of-00014.safetensors), [MTP/shared-expert shard](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/model.safetensors-00014-of-00014.safetensors)).

**Observed conclusion:** “experts” in a path does not guarantee an Expert ID axis. Explicit per-expert tensors, embedded `expert_<id>` names, and packed expert tensors all exist. Packed tensors already include expert multiplicity in their Shape and must not receive an inferred multiplier.

### Auxiliary stacks and interleaved layer structures should remain visible

Qwen3.5's main text config lists a repeating hybrid pattern of linear-attention and full-attention layer types. The checkpoint reflects that pattern: `model.language_model.layers.0.linear_attn.*` exists while `model.language_model.layers.3.self_attn.*` exists. Both belong to the same main layer axis, but their normalized keys naturally cover different Layer ID subsets ([Qwen3.5-35B-A3B config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json), [checkpoint index](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/model.safetensors.index.json)).

The separate `mtp.layers.0` axis is not part of the `model.language_model.layers` ID space even though its tensor suffixes resemble main-layer suffixes. Model-level MTP tensors such as `mtp.fc.weight` and `mtp.norm.weight` have no numeric layer segment and therefore remain model-level tensors ([Qwen3.5-35B-A3B checkpoint index](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/model.safetensors.index.json)).

**Observed conclusion:** aggregation by normalized full path exposes interleaving without first classifying a tensor as attention or FFN. Auxiliary stacks require qualified axes; auxiliary tensors outside those stacks remain model-level.

### Quantization companions are independent tensors

The official Qwen2.5-7B GPTQ-Int4 checkpoint stores each quantized projection as multiple leaves:

```text
model.layers.0.self_attn.q_proj.qweight  # I32 [448, 3584]
model.layers.0.self_attn.q_proj.qzeros   # I32 [28, 448]
model.layers.0.self_attn.q_proj.scales   # F16 [28, 3584]
model.layers.0.self_attn.q_proj.g_idx    # I32 [3584]
model.layers.0.self_attn.q_proj.bias     # F16 [3584]
```

The config identifies GPTQ, 4 bits, and group size 128. Names come from the official index and Shapes/DTypes from the first checkpoint shard ([Qwen2.5 GPTQ config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4/blob/main/config.json), [checkpoint index](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4/blob/main/model.safetensors.index.json), [first checkpoint shard](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4/blob/main/model-00001-of-00002.safetensors)).

DeepSeek-V3 FP8 stores companion inverse scales next to weights:

```text
model.layers.3.mlp.experts.0.gate_proj.weight            # F8_E4M3 [2048, 7168]
model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv  # F32 [16, 56]
```

Both entries are real tensors and both contribute bytes/parameters according to their own metadata ([DeepSeek-V3 checkpoint index](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/model.safetensors.index.json), [first checkpoint shard](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/model-00001-of-000163.safetensors)).

**Observed conclusion:** companion suffixes are part of the full tensor path. They must not be discarded, folded into a base `weight`, or assigned the base weight's Shape/DType.

## Recommended generic heuristic boundaries

The following are recommendations inferred from the evidence. They are deliberately conservative: an unrecognized tensor should remain visible as model-level or unaggregated data instead of being silently assigned to the wrong axis.

### 1. Parse path structure before grouping

Tokenize the complete tensor name on `.` and retain the original name. Consider a standalone non-negative integer only as an axis candidate, never as an axis by default.

Candidate layer markers should be syntax-oriented rather than model-oriented—for example `layers`, `layer`, `blocks`, `block`, and compact stack markers such as `h`. The marker plus its complete prefix is the axis identity:

| Observed path fragment | Axis identity | Layer ID |
| --- | --- | --- |
| `model.layers.7` | `model.layers` | `model.layers:7` |
| `visual.blocks.7` | `visual.blocks` | `visual.blocks:7` |
| `decoder.block.7` | `decoder.block` | `decoder.block:7` |
| `model.language_model.layers.7` | `model.language_model.layers` | `model.language_model.layers:7` |
| `mtp.layers.0` | `mtp.layers` | `mtp.layers:0` |

Do not strip prefixes down to a generic word such as `layers`; doing so would merge independent stacks.

### 2. Score structural repetition; do not trust marker spelling alone

When multiple numeric candidates occur in one name, prefer the candidate supported by repeated sibling structures: replacing its numeric value should produce matching relative suffixes across multiple IDs, with matching Shape and DType for at least some leaves. This makes `decoder.block.<id>` stronger than the inner `decoder.block.0.layer.<slot>` family.

The exact scoring threshold belongs in the later axis-recognition contract/prototype. These are required boundaries:

- Never promote every numeric segment.
- Never promote a candidate solely because its preceding token is `layer`.
- Retain all unselected numeric segments in the Layer-Normalized Tensor Key.
- Permit a strongly marked singleton axis such as `mtp.layers.0`; a “two IDs minimum” rule would be a known false negative.
- If two candidates remain ambiguous, prefer no aggregation over a fabricated layer relationship.

### 3. Treat the routed-expert axis independently

Recognize an explicit routed-expert axis only when an Expert ID is present and sibling evidence supports repetition. At minimum, the parser should be able to represent both observed syntaxes:

- separate segment: `experts.<integer>`
- embedded token: `expert_<integer>`

The qualified expert axis is its complete path through the expert marker, scoped by the containing Layer ID. Normalize it independently from the layer segment.

Do not create a routed-expert axis for:

- `shared_expert` or `shared_experts` without an ID;
- router/gate tensors such as `mlp.gate.weight` or `shared_expert_gate.weight`;
- packed tensors such as `mlp.experts.down_proj` whose Shape already stores the expert dimension but whose name carries no Expert ID.

Count actual observed Expert IDs per Layer/aggregate. Do not assume a global count and do not multiply packed expert tensors.

### 4. Preserve full leaf identity, Shape, and DType

After normalizing only recognized layer and routed-expert IDs, aggregate by:

```text
normalized full tensor path + exactly ordered Shape + on-disk DType
```

This boundary keeps:

- hybrid layer variants in separate aggregates while exposing their Layer ID patterns;
- shared experts separate from routed experts even when their Shapes match;
- quantization companions separate from their base weights;
- anomalous experts or layers separate when Shape or DType differs.

Shape equality alone is not an axis detector, and equal parameter count with a different ordered Shape is not identity.

### 5. Keep model-level and unresolved tensors lossless

A tensor for which no layer axis is recognized remains a Model-Level Tensor. Numeric module slots such as `visual.merger.mlp.0` remain part of its full path. The view may preserve normal path hierarchy for readability, but it must not fabricate a Layer ID.

This fallback is also the safe behavior for unknown naming schemes. The heuristic organizes structure; it does not infer model architecture, KV-cache behavior, or supported-model status.

## Reusable sample inventory

The following compact fixture inventory is derived from the cited checkpoint headers. It is intended for later recognition-contract tests. `layerAxis: null` means the tensor should remain model-level. `expertAxis: null` means no routed-expert multiplier is permitted. The inventory is intentionally decision-focused rather than exhaustive.

```json
[
  {
    "case": "qwen_vl_text_layer",
    "name": "model.layers.0.self_attn.q_proj.weight",
    "dtype": "BF16",
    "shape": [2048, 2048],
    "layerAxis": "model.layers",
    "layerId": 0,
    "expertAxis": null
  },
  {
    "case": "qwen_vl_visual_layer",
    "name": "visual.blocks.0.attn.qkv.weight",
    "dtype": "BF16",
    "shape": [3840, 1280],
    "layerAxis": "visual.blocks",
    "layerId": 0,
    "expertAxis": null
  },
  {
    "case": "qwen_vl_visual_layer_same_key_other_id",
    "name": "visual.blocks.31.attn.qkv.weight",
    "dtype": "BF16",
    "shape": [3840, 1280],
    "layerAxis": "visual.blocks",
    "layerId": 31,
    "expertAxis": null
  },
  {
    "case": "qwen_vl_fixed_numeric_module_slot_0",
    "name": "visual.merger.mlp.0.weight",
    "dtype": "BF16",
    "shape": [5120, 5120],
    "layerAxis": null,
    "layerId": null,
    "expertAxis": null
  },
  {
    "case": "qwen_vl_fixed_numeric_module_slot_2",
    "name": "visual.merger.mlp.2.weight",
    "dtype": "BF16",
    "shape": [2048, 5120],
    "layerAxis": null,
    "layerId": null,
    "expertAxis": null
  },
  {
    "case": "t5_decoder_block_with_inner_slot",
    "name": "decoder.block.0.layer.0.SelfAttention.q.weight",
    "dtype": "F32",
    "shape": [2048, 2048],
    "layerAxis": "decoder.block",
    "layerId": 0,
    "retainedNumericSegments": ["layer.0"],
    "expertAxis": null
  },
  {
    "case": "t5_decoder_other_block_same_key",
    "name": "decoder.block.1.layer.0.SelfAttention.q.weight",
    "dtype": "F32",
    "shape": [2048, 2048],
    "layerAxis": "decoder.block",
    "layerId": 1,
    "retainedNumericSegments": ["layer.0"],
    "expertAxis": null
  },
  {
    "case": "t5_decoder_cross_attention_slot",
    "name": "decoder.block.0.layer.1.EncDecAttention.q.weight",
    "dtype": "F32",
    "shape": [2048, 2048],
    "layerAxis": "decoder.block",
    "layerId": 0,
    "retainedNumericSegments": ["layer.1"],
    "expertAxis": null
  },
  {
    "case": "t5_encoder_separate_axis",
    "name": "encoder.block.0.layer.0.SelfAttention.q.weight",
    "dtype": "F32",
    "shape": [2048, 2048],
    "layerAxis": "encoder.block",
    "layerId": 0,
    "retainedNumericSegments": ["layer.0"],
    "expertAxis": null
  },
  {
    "case": "mixtral_routed_expert_0",
    "name": "model.layers.0.block_sparse_moe.experts.0.w1.weight",
    "dtype": "BF16",
    "shape": [14336, 4096],
    "layerAxis": "model.layers",
    "layerId": 0,
    "expertAxis": "model.layers.0.block_sparse_moe.experts",
    "expertId": 0
  },
  {
    "case": "mixtral_routed_expert_7",
    "name": "model.layers.0.block_sparse_moe.experts.7.w1.weight",
    "dtype": "BF16",
    "shape": [14336, 4096],
    "layerAxis": "model.layers",
    "layerId": 0,
    "expertAxis": "model.layers.0.block_sparse_moe.experts",
    "expertId": 7
  },
  {
    "case": "mixtral_router_not_expert",
    "name": "model.layers.0.block_sparse_moe.gate.weight",
    "dtype": "BF16",
    "shape": [8, 4096],
    "layerAxis": "model.layers",
    "layerId": 0,
    "expertAxis": null
  },
  {
    "case": "deepseek_routed_expert",
    "name": "model.layers.3.mlp.experts.0.gate_proj.weight",
    "dtype": "F8_E4M3",
    "shape": [2048, 7168],
    "layerAxis": "model.layers",
    "layerId": 3,
    "expertAxis": "model.layers.3.mlp.experts",
    "expertId": 0
  },
  {
    "case": "deepseek_shared_expert_not_routed",
    "name": "model.layers.3.mlp.shared_experts.gate_proj.weight",
    "dtype": "F8_E4M3",
    "shape": [2048, 7168],
    "layerAxis": "model.layers",
    "layerId": 3,
    "expertAxis": null
  },
  {
    "case": "deepseek_fp8_companion",
    "name": "model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv",
    "dtype": "F32",
    "shape": [16, 56],
    "layerAxis": "model.layers",
    "layerId": 3,
    "expertAxis": "model.layers.3.mlp.experts",
    "expertId": 0
  },
  {
    "case": "qwen35_packed_experts",
    "name": "model.language_model.layers.0.mlp.experts.down_proj",
    "dtype": "BF16",
    "shape": [256, 2048, 512],
    "layerAxis": "model.language_model.layers",
    "layerId": 0,
    "expertAxis": null,
    "packedMultiplicityAlreadyInShape": true
  },
  {
    "case": "qwen35_mtp_explicit_expert_0",
    "name": "mtp.layers.0.mlp.experts.0.down_proj.weight",
    "dtype": "BF16",
    "shape": [2048, 512],
    "layerAxis": "mtp.layers",
    "layerId": 0,
    "expertAxis": "mtp.layers.0.mlp.experts",
    "expertId": 0
  },
  {
    "case": "qwen35_mtp_explicit_expert_1",
    "name": "mtp.layers.0.mlp.experts.1.down_proj.weight",
    "dtype": "BF16",
    "shape": [2048, 512],
    "layerAxis": "mtp.layers",
    "layerId": 0,
    "expertAxis": "mtp.layers.0.mlp.experts",
    "expertId": 1
  },
  {
    "case": "qwen25_gptq_qweight",
    "name": "model.layers.0.self_attn.q_proj.qweight",
    "dtype": "I32",
    "shape": [448, 3584],
    "layerAxis": "model.layers",
    "layerId": 0,
    "expertAxis": null
  },
  {
    "case": "qwen25_gptq_qzeros",
    "name": "model.layers.0.self_attn.q_proj.qzeros",
    "dtype": "I32",
    "shape": [28, 448],
    "layerAxis": "model.layers",
    "layerId": 0,
    "expertAxis": null
  },
  {
    "case": "qwen25_gptq_scales",
    "name": "model.layers.0.self_attn.q_proj.scales",
    "dtype": "F16",
    "shape": [28, 3584],
    "layerAxis": "model.layers",
    "layerId": 0,
    "expertAxis": null
  },
  {
    "case": "qwen25_gptq_group_index",
    "name": "model.layers.0.self_attn.q_proj.g_idx",
    "dtype": "I32",
    "shape": [3584],
    "layerAxis": "model.layers",
    "layerId": 0,
    "expertAxis": null
  }
]
```

## Decision summary

1. A Layer ID must be qualified by the complete path of its recognized repetition axis.
2. A bare numeric path segment is insufficient evidence. Fixed sublayer/module positions must survive normalization.
3. A routed-expert axis is independent of the layer axis and exists only when an actual Expert ID is encoded; shared experts, routers, and packed experts are not routed-expert instances.
4. Quantization companions are ordinary tensors with distinct full keys, Shapes, and DTypes.
5. The safest generic fallback is lossless non-aggregation. It is better to show an unfamiliar structure verbatim than to invent a layer or expert relationship.
6. The observed samples support a generic, tensor-metadata-only heuristic. They do not support per-model presentation branches.

## Questions deliberately left for the recognition-contract ticket

- The exact confidence score/threshold for choosing among multiple numeric candidates.
- Whether and how to expose more than one nested non-expert repetition axis in a single tensor path.
- How a singleton axis competes with an ambiguous fixed module slot when lexical evidence is weak.
- The internal data representation for packed expert multiplicity versus explicit Expert IDs.

These are contract/prototype decisions, not facts that primary-source research can settle alone.
