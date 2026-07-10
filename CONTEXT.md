# LLM VRAM Estimation

This context defines the domain language for model-inference VRAM estimation, with particular emphasis on distinguishing model identity, attention mechanisms, and the KV Cache that remains GPU-resident.

## Language

**Model Class Identifier**:
A model class name declared in `config.architectures[]` as capable of loading a set of pretrained weights; it is not a complete identity for a KV Cache Layout.
_Avoid_: Architect, layout identifier

**Model Architecture**:
A model-level architecture category that determines complete KV Cache storage semantics; it is not synonymous with an individual attention mechanism such as MHA, GQA, or MLA.
_Avoid_: Model Class Identifier, attention architecture

**Architecture Profile**:
A specifically identified and manually reviewed model-architecture variant; each Profile maps uniquely to one dedicated complete KV Cache Layout, while aliases that differ only by class name may share a Profile.
_Avoid_: Generic architecture guess, layout template

**KV Cache Layout**:
The complete set of cache structures that a Model Architecture requires to remain GPU-resident during inference; a hybrid-mechanism model owns one dedicated complete Layout rather than a composition of existing Layouts.
_Avoid_: Hybrid layout, layout composition

**KV Cache Buffer Primitive**:
A precisely defined resident cache structure within a complete Layout; it may reuse representation and accounting mechanisms but cannot replace an Architecture Profile's complete Layout definition and end-to-end verification.
_Avoid_: Generic layout, shared formula as proof of correctness

**Verified Layout**:
A dedicated Layout backed by a first-party implementation, an independent VRAM derivation, Profile-level end-to-end golden tests, and, for complex cases, cross-checks against official cache Shapes or measured VRAM; a Layout that does not meet this evidence threshold is not supported.
_Avoid_: Speculative support, primitive-only test coverage

**Effective KV Cache Payload**:
All KV Cache data that model semantics require to remain GPU-resident for a specified batch and context, including model-defined compression, windows, indexers, and cache DType but excluding framework capacity reservations, allocator fragmentation, and offload policy.
_Avoid_: Framework allocation, CUDA reserved memory

**Complete VRAM Estimate**:
A total VRAM result available only when weights, a Verified KV Cache Payload, and fixed overhead are all known; if any required component is unknown, the total must remain unknown.
_Avoid_: Treating unknown KV as zero, partial total VRAM

**Architecture Layout Catalog**:
An explicit, manually reviewed or specifically verified catalog that maps a Model Class Identifier, through model-specific identification, to one Architecture Profile and its unique KV Cache Layout; entries must not be inferred from naming patterns or generic heuristics.
_Avoid_: Automatic architecture guessing, rule-based routing

**Unsupported Model Architecture**:
A Model Architecture that is not explicitly included in the Architecture Layout Catalog; its KV Cache VRAM usage is unknown and must not be estimated through a generic formula or heuristic fallback.
_Avoid_: Guessed result, default MHA

**Layer View**:
Organizes tensors by axis-qualified Layer ID and presents each layer's tensor structure separately, preserving topology differences between layers.
_Avoid_: Current view, old view

**Cross-Layer Aggregate View**:
Organizes tensor structures that repeat across multiple Layers while retaining the axis-qualified Layer IDs covered by each aggregate, making recurring patterns and structural variants visible.
_Avoid_: Key view, compressed view

**Structure Presentation Heuristic**:
Automatically organizes Layer and Cross-Layer Aggregate Views using parsed tensor names, exact Shapes, and DTypes only; it does not determine model architecture, KV Cache layout, or verified support status.
_Avoid_: Architecture inference, KV Cache layout inference

**Layer-Normalized Tensor Key**:
The full tensor path after replacing the recognized Layer ID segment with a placeholder; module path segments remain intact and are not preclassified into display categories such as attention or FFN.
_Avoid_: Key, tensor shorthand, module category

**Cross-Layer Aggregate Identity**:
The combination of a Layer-Normalized Tensor Key, an exactly ordered Shape, and the on-disk DType; a difference in any component creates a separate aggregate.
_Avoid_: Parameter-count-only grouping, tensor-name-only grouping

**Routed-Expert Repetition Axis**:
The Expert ID in a routed-expert path is a repetition dimension independent of the Layer ID; aggregates retain the actual expert multiplicity for each layer, and only experts with identical aggregate identities share a representative structure.
_Avoid_: Global expert multiplier, counting shared experts as routed experts

**Model-Level Tensor**:
A tensor that does not belong to any Layer repetition axis; it remains visible in both structure views but does not participate in cross-layer aggregation or receive a fabricated Layer ID.
_Avoid_: Other outside Layers, layerless aggregate

**Axis-Qualified Layer ID**:
A Layer identifier composed of a repetition-path axis and a numeric index within that axis, such as `model.layers:7` or `mtp:0`; equal numeric indices do not imply the same Layer.
_Avoid_: Bare Layer index, cross-axis Layer ID
