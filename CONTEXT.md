# LLM VRAM Estimation

This context defines the domain language for model-inference VRAM estimation, with particular emphasis on distinguishing model identity, attention mechanisms, and the KV Cache that remains GPU-resident.

## Language

**Model Class Identifier**:
A model class name declared in `config.architectures[]` as capable of loading a set of pretrained weights; it is not a complete identity for a KV Cache Layout.
_Avoid_: Architect, layout identifier

**Model Repository Identifier**:
The canonical Hugging Face `repoId` resolved by ingestion; it selects one Architecture Profile in the Architecture Layout Catalog, independently of the Model Class Identifier.
_Avoid_: User-entered alias, Model Class Identifier

**Model Architecture**:
A model-level architecture category that determines complete KV Cache storage semantics; it is not synonymous with an individual attention mechanism such as MHA, GQA, or MLA.
_Avoid_: Model Class Identifier, attention architecture

**Architecture Profile**:
A standardized definition of one dedicated complete KV Cache Layout algorithm, its Model Repository Identifiers, and one audited commit plus audited values for every config input the algorithm consumes; calculation uses the current revision's valid config inputs even when they differ from the audited baseline.
_Avoid_: Generic architecture guess, layout template, checkpoint-wide tensor signature

**KV Cache Layout**:
The complete set of cache structures that a Model Architecture requires to remain GPU-resident during inference; a hybrid-mechanism model owns one dedicated complete Layout rather than a composition of existing Layouts.
_Avoid_: Hybrid layout, layout composition

**KV Cache Buffer Primitive**:
A precisely defined resident cache structure within a complete Layout; it may reuse representation and accounting mechanisms but cannot replace an Architecture Profile's complete Layout definition and end-to-end verification.
_Avoid_: Generic layout, shared formula as proof of correctness

**Verified Layout**:
A dedicated Layout whose current Model Repository Identifier, immutable commit, and every algorithm-dependent config input equal the Architecture Profile's audited baseline; other valid revisions may still produce a warning calculation but are not Verified.
_Avoid_: Treating a warning calculation as Verified, primitive-only test coverage

**Profile Assurance**:
The verification state of an Architecture Profile calculation; it is `verified` only when the current immutable commit and every algorithm-dependent config input match the audited baseline, otherwise it is `warning` with structured commit and config differences.
_Avoid_: Calculation availability, Complete VRAM Estimate completeness

**Effective KV Cache Payload**:
All KV Cache data that model semantics require to remain GPU-resident for a specified batch and context, including model-defined compression, windows, indexers, and cache DType but excluding framework capacity reservations, allocator fragmentation, and offload policy.
_Avoid_: Framework allocation, CUDA reserved memory

**Complete VRAM Estimate**:
A total VRAM result available only when weights and an Effective KV Cache Payload are both known; if either required component is unknown, the total must remain unknown, while Profile Assurance independently communicates whether a known result is Verified or warning-only.
_Avoid_: Treating unknown KV as zero, partial total VRAM

**Architecture Layout Catalog**:
An explicit catalog that maps each canonical Model Repository Identifier to exactly one Architecture Profile and its unique KV Cache Layout; duplicate repository registrations and invalid Profile definitions fail during catalog initialization.
_Avoid_: Model Class Identifier routing, automatic architecture guessing, rule-based routing

**Unsupported Model Architecture**:
A Model Architecture whose canonical Model Repository Identifier is not explicitly included in the Architecture Layout Catalog; its KV Cache VRAM usage is unknown and must not be estimated through a generic formula or heuristic fallback.
_Avoid_: Guessed result, default MHA

**Tensor Name Tree**:
A lossless presentation of parsed tensor metadata as a left-to-right prefix tree of dot-delimited tensor names; branch rows show cumulative prefixes and direct child counts, while leaves retain the original Shape, DType, parameters, and bytes.
_Avoid_: Layer View, architecture tree

**Numeric Path Branch**:
A Tensor Name Tree branch whose accumulated path contains a standalone non-negative integer segment; it is independently collapsible and carries no inferred Layer or Expert meaning.
_Avoid_: Layer ID, Expert ID, architecture inference

**Repeated Tensor Group**:
A bottom-up display group of numeric sibling subtrees whose normalized relative keys, terminal Shapes, and terminal DTypes are identical; one representative is displayed with its repeat count and IDs relative to the current parent, while parameters and bytes include every original tensor across ancestor repetitions. The public `groupRepeatedTensorSubtrees` operation creates these groups.
_Avoid_: Architecture-based grouping, representative-only VRAM, shape-only grouping

**Tensor Name Pattern**:
A presentation-only aggregation identity for tensors whose dot-delimited names differ only in standalone non-negative integer segments, represented by `*`; unlike a Repeated Tensor Group, it does not require matching terminal Shapes or DTypes.
_Avoid_: Repeated Tensor Group, Tensor category, architecture grouping

**Tensor Metadata Index**:
A lossless derived index of parsed tensor metadata that provides authoritative parameter counts, DTypes, weight bytes, Tensor Name Patterns, and subtree totals; it may produce a Tensor Name Tree and Repeated Tensor Groups as presentation projections without assigning model-architecture semantics.
_Avoid_: Model architecture inference, presentation tree as accounting authority
