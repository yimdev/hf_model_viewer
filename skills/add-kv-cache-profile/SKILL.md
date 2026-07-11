---
name: add-kv-cache-profile
description: Add and audit dedicated KV Cache Architecture Profiles for Hugging Face models in hf_view. Use when researching a model's cache semantics, implementing or registering support for a new model repository, or validating a Profile's formulas, provenance, and tests.
---

# Add KV Cache Profile

Build an evidence chain from immutable upstream code to an independently checked Effective KV Cache Payload. A Profile is complete only when every GPU-resident decode state is accounted for and the audited revision passes the repository's assurance protocol.

## 1. Establish the contract

Read `CONTEXT.md`, `src/vram/kv/profile-definition.js`, `src/vram/kv/profile-primitives.js`, `src/vram/kv/catalog.js`, and the closest existing Profile and tests. Read [references/profile-contract.md](references/profile-contract.md) for the current extension points and required validation matrix.

Identify the canonical Model Repository Identifier, exact model variant, and default framework/cache implementation whose Effective KV Cache Payload the Profile represents. Do not route by Model Class Identifier: one class may load repositories with different Layouts. Treat static capacity, quantized caches, offload, and other selectable implementations as runtime policies unless the model requires them.

Complete this step when the target repository, intended Profile boundary, and all project invariants are explicit.

## 2. Audit an immutable revision

For a new or changed Layout, read [references/research-playbook.md](references/research-playbook.md) before gathering evidence.

Resolve the requested revision to a 40-character commit. Inspect the pinned `config.json`, model code, cache object or update path, and relevant kernels. Trace prefill and one-token decode far enough to enumerate every persistent GPU-resident state. Use official model repositories and upstream implementation repositories as primary evidence; use papers and third-party runtimes to cross-check ambiguous semantics.

Maintain an evidence ledger while investigating. For every candidate state, record its shape, dtype, lifetime, update rule, config dependencies, source location, and inclusion or exclusion rationale. Include token-growing buffers, windowed or compressed history, indexer state, causal-convolution history, recurrent matrices, and other persistent state. Exclude weights, temporary workspaces, allocator capacity, fragmentation, offload policy, and optional speculative modules unless the base runtime requires them.

Complete this step only when every persistent decode state and every algorithm-dependent config input is accounted for by pinned evidence. If the evidence cannot establish a complete Layout, report the gap and leave the repository on the Generic KV Cache Estimate path.

## 3. Derive an independent oracle

Write the Layout equations from the evidence ledger before translating them into Profile code. Define workload semantics for uniform batches, ragged `sequenceLengths`, zero-length sequences, active-sequence state, maximum context, window boundaries, and compression or chunk transitions. State whether ragged lengths represent independently resident or logically packed sequences; do not silently substitute the physical allocation of one padded framework batch.

Hand-calculate golden vectors that exercise one ordinary point and every semantic boundary. Use a separate calculation method or a small disposable derivation so expected totals are not produced by the Profile implementation itself. Check dimensions and dtype widths for every buffer, then sum bytes.

Complete this step when each buffer has an independently derived formula and at least one golden total, with boundary vectors for every piecewise rule.

## 4. Implement the Profile

Create one file in `src/vram/kv/profiles/` that owns the complete Layout. Declare only config inputs consumed by the algorithm, validate cross-field invariants, and calculate from current valid config values. Store the pinned values of every declared input under each repository audit. Give repository aliases separate audited commits; share a Profile only when pinned evidence proves they use the same Layout.

Represent each resident structure with `makeBuffer(...)`. Make `layerGroup`, dtype, byte width, element count, and displayed formula agree. Apply `validateSequenceWorkload(...)` with the model's actual workload policy. Return explicit notes for material inclusions and exclusions.

Register the Profile explicitly in `src/vram/kv/catalog.js`. Add the pinned config to `test/vram/profile-fixtures.js`; keep fixture values traceable to the audited revision and include only tensor metadata needed by affected integration tests.

Complete this step when the catalog selects exactly one Profile for every added repository and the Profile computes all oracle vectors from current config inputs.

## 5. Prove the evidence chain

Add tests from the required matrix in [references/profile-contract.md](references/profile-contract.md). Compare per-buffer IDs, bytes, and dtypes as well as the total. Exercise ragged and zero workloads according to the model's semantics, every piecewise boundary, config drift, missing or invalid inputs, a changed commit, and safe-integer failure. Test repository aliases independently.

Run the focused KV/Profile tests, then `npm test` and `npm run build`. Update `README.md` when the supported-model list or public calculation description changes. Update `CONTEXT.md` only when the contribution introduces a stable domain concept that existing vocabulary cannot express. Keep all project contributions in English.

Complete this step only when every ledger entry maps to a buffer or explicit exclusion, all golden vectors pass, protocol behavior remains intact, the full test suite passes, and the production build succeeds.

## 6. Hand off the audit

Summarize the immutable revisions and primary sources, the complete buffer inventory, equations and dtype choices, workload semantics, exclusions, golden vectors, and commands run. Identify any framework-specific assumptions. Distinguish a computed result from a Verified Layout: only the exact audited commit and baseline inputs earn `verified`; valid drift earns `warning`.
