# Architecture Profile contract

Read this reference before implementing or reviewing a Profile. Source code remains authoritative when the contract evolves.

## Extension points

- `src/vram/kv/profiles/<model>.js`: owns one complete KV Cache Layout.
- `src/vram/kv/catalog.js`: explicitly maps each registered Model Repository Identifier through the Profile's `repositories` entries.
- `src/vram/kv/profile-definition.js`: validates Profile identity, immutable provenance, config inputs, drift, and output safety.
- `src/vram/kv/profile-primitives.js`: provides buffer construction and sequence-workload validation.
- `test/vram/profile-fixtures.js`: holds audited config fixtures and only the tensor metadata needed by integration tests.
- `test/vram/kv-profile.test.js`: holds model Layout golden vectors and workload semantics.
- `test/vram/profile-protocol.test.js`: protects assurance and provenance behavior shared by Profiles.
- `test/vram/estimate-vram.test.js`: covers end-to-end weight plus Effective KV Cache Payload totals when the new model changes that surface.

## Definition shape

`defineArchitectureProfile(...)` requires:

- `id`, `version`, and `label` for the Architecture Profile.
- `layout: { id, version }` for the complete KV Cache Layout.
- One or more `repositories` with canonical `repoId`, 40-character `auditedCommitId`, and `baselineInputs` exactly matching the declared config-input names.
- `configInputs` descriptors with a config path and validator.
- `calculateLayout({ inputs, workload })`, returning `{ buffers, note }` or a structured range error.
- Optional `validateInputs(inputs)` for cross-field and model-specific invariants.

`profileConfigInput.positiveInteger(path)` and `.array(path)` cover common values. Define a local descriptor when a model needs enums, dtypes, booleans, nullable values, or other semantics. Include every value that can change the algorithm's buffer count, shape, dtype, growth rule, or workload limit. Omit descriptive values that the calculation does not consume.

Current valid inputs drive calculation. Baseline inputs establish assurance; they are not constants that replace current config. An exact commit and exact inputs produce `verified`. Valid commit or config drift produces `warning` and still calculates. Invalid provenance or invalid required inputs produces `unknown`.

## Buffer contract

Create every resident structure through `makeBuffer(...)` with:

- a stable `id` and human-readable `label`;
- an exact `layerGroup` count plus indices or range;
- an integer element count;
- a semantic dtype and matching byte width;
- a formula that describes the same element count shown to users.

Supported semantic dtype widths currently live in `profile-primitives.js`. Extend that single source of truth only when pinned evidence requires a new stored dtype.

Use `validateSequenceWorkload(...)` for uniform `{ batch, seq }` and ragged `{ sequenceLengths }` inputs. Select its zero-batch and empty-list policy from model semantics. Piecewise per-sequence layouts should use `workload.entries`; token-linear layouts may use `workload.tokenCount`. Count active sequences explicitly for fixed recurrent state.

Document the default framework/cache implementation used to establish the semantic Layout. Keep selectable allocation policies outside the Effective KV Cache Payload. Interpret `sequenceLengths` as independent or logically packed sequences unless the Profile explicitly defines a different policy; a padded runtime tensor's unused capacity is not semantic payload.

## Required validation matrix

Every new Profile must demonstrate:

- exact audited revision and inputs produce `computed` plus `verified`;
- exact per-buffer IDs, bytes, dtypes, and total match an independent oracle;
- a valid changed algorithm input changes the result and produces structured config drift;
- a different valid commit produces commit drift;
- every required input missing or invalid yields `invalid_profile_config`;
- ordinary uniform, ragged, and zero workloads follow documented Layout semantics;
- window, compression, chunk, or recurrent boundaries have golden vectors;
- maximum context and impossible unsafe arithmetic fail with the expected diagnostic;
- every repository alias is selected and verified against its own audited commit;
- catalog initialization remains unique and full `npm test` plus `npm run build` pass.

Add only applicable integration assertions, but never remove a relevant row because another Profile already covers the shared protocol. A primitive-only unit test does not verify a complete Layout.

## Version and documentation changes

Use a new Profile/Layout identity for a different complete Layout. Bump semantic versions when behavior or audit meaning changes while identity remains valid. Keep Profile and Layout versions independently meaningful.

Update `README.md` for the public supported-model list and material calculation semantics. Add to `CONTEXT.md` only when the implementation resolves a durable domain term; use existing vocabulary otherwise. Keep formulas, labels, notes, tests, documentation, and contribution summaries in English.
