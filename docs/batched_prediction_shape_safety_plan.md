# Plan: preserve dataset independence in batched prediction

## Summary

`TabPFNRegressor.predict_batched` and
`TabPFNClassifier.predict_proba_batched` currently validate the shapes of the
raw datasets, preprocess each dataset independently, and then pass every
preprocessed dataset to `meta_dataset_collator` in one group. Independent
preprocessors can produce different feature counts even when all raw arrays
have the same shape. Constant-feature removal is one common way this happens.

`meta_dataset_collator` pads tensors to the largest shape in its input. That is
intentional for its fine-tuning callers, but it is not safe for prediction:
TabPFN has no feature-padding mask, so each padded zero column is interpreted as
a real feature. A dataset's prediction can consequently depend on which other
datasets are present in the same batch.

The fix should group datasets by their shapes *after all CPU and GPU
preprocessing*, run one fused prediction per compatible group, and restore the
original input order. The inference path should also assert that no padding is
needed before it calls the general-purpose collator.

## Evidence and impact

The failure reproduces in both public batched APIs with the stock v3
checkpoints.

In one regression reproduction, four equal-shaped raw datasets produced 21,
11, 21, and 20 transformed columns. A batch-size-one call matched serial
`fit`/`predict` exactly. Pairing the 11-column dataset with a 21-column dataset
padded ten features and changed its prediction by 19.3%; repeating that same
11-column dataset twice changed it by only 0.0007%. Pairing the 20-column
dataset with a 21-column dataset changed it by 7.0%, while the unpadded
21-column member was stable.

With the stock v3 regressor and two estimators, a 14-column transformed dataset
changed by 5.8% on L4 when paired with a 25-column dataset, compared with 0.022%
in duplicate- and matched-width controls. The issue also reproduced on RTX.

For three-class classification with the stock v3 classifier, padding a
12/14-column target alongside a 22/25-column dataset changed an output class
probability by 0.0127; the matched-width control changed it by 0.000069.

The exact error depends on model version, estimator ensemble, precision, GPU,
and data. The correctness violation does not: callers are promised independent
predictions, but an unmasked padded feature axis makes results depend on batch
composition.

## Scope and non-goals

This change should cover:

- `TabPFNRegressor.predict_batched`;
- `TabPFNClassifier.predict_proba_batched`;
- shared helpers needed to identify and safely collate compatible
  preprocessed datasets;
- regression tests for stock architectures, including v3, while keeping the
  helpers architecture-independent so other supported architectures receive
  the fix automatically;
- API documentation and a changelog fragment.

This change should not:

- change `meta_dataset_collator`'s existing padding semantics for fine-tuning;
- add padding masks to every TabPFN architecture;
- relax the current public requirement that raw train arrays share a shape and
  raw test arrays share a shape;
- change preprocessing, constant-feature detection, ensemble construction,
  target transforms, probability balancing, or temperature tuning;
- promise bitwise equality across batch sizes or GPU architectures. Compatible
  fused kernels can still differ from singleton kernels within the existing
  numerical tolerance.

## Proposed design

### 1. Define a post-preprocessing shape signature

Add a private helper near the batch dataclasses/collation utilities that derives
an immutable signature from one `ClassifierBatch` or `RegressorBatch` before
collation.

The signature should include, for every ensemble member and in ensemble order:

- the complete `X_context` shape;
- the complete `X_query` shape;
- the `y_context` shape.

Also include the non-ensemble `y_query` shape. Context and query row counts are
currently protected by raw-shape validation, but including complete shapes
makes the invariant explicit and protects future preprocessing changes.

Categorical indices and ensemble configs do not need to be equal across
datasets: the batched executor already carries their per-dataset values. The
existing executor validation that one estimator slot cannot select multiple
underlying models remains authoritative.

The signature must be computed from tensors after `_maybe_run_gpu_preprocessing`.
CPU-only shapes are insufficient because GPU preprocessing is allowed to alter
the final feature axis.

### 2. Group preprocessed items stably

Add a small stable grouping helper, or equivalent local logic shared by the two
APIs:

1. enumerate `(original_index, item)` pairs;
2. compute each item's shape signature;
3. append the pair to an insertion-ordered mapping keyed by signature;
4. preserve item order inside each group.

The result is a sequence of compatible groups plus the original positions used
to stitch predictions back together. Stable grouping makes behavior
deterministic and prevents output reordering when signatures are interleaved,
for example `A, B, A, C, B`.

Do not group only on a single width. Preprocessing configuration can differ by
ensemble member, so the key must be the full vector of per-estimator shapes.

### 3. Add an inference-safe collator boundary

Introduce a private inference helper such as
`_collate_same_shape_for_batched_inference(items)`. It should:

- reject an empty group;
- compare the post-preprocessing signatures and raise an internal-error-style
  exception if more than one is present;
- delegate to `meta_dataset_collator` only after proving padding is a no-op.

Keeping this assertion adjacent to collation is important. Grouping alone can
regress silently if a future caller supplies an incomplete signature. The
helper documents that unmasked feature padding is forbidden for inference,
while leaving the generic collator available to fine-tuning code that relies on
padding.

### 4. Run each regressor shape group independently

Refactor the section of `predict_batched` after `items` are built:

1. Keep constant-target results in the existing preallocated result list.
2. Store each non-constant item's original input index and its own raw-space bar
   distribution together, rather than relying on parallel lists whose positions
   assume one global fused batch.
3. Stable-group the non-constant items by post-preprocessing signature.
4. For each group:
   - inference-safely collate the group;
   - call `fit_from_preprocessed` on the worker;
   - iterate ensemble outputs exactly as today;
   - translate and accumulate logits per group member using that member's
     fitted `RegressorEnsembleConfig`;
   - decode with that member's raw-space bar distribution;
   - write the result at its original input index.
5. Retain the final assertion that every result position was filled.

The worker may replace its batched executor for every group. Model weights are
already loaded on the worker and `fit_from_preprocessed(..., no_refit=True)` is
designed to reuse them. Verify this explicitly with a test or spy so grouping
does not reload the checkpoint per group.

Do not combine logits across groups. `n_estimators` should be counted and
validated independently per group, even though every group is expected to have
the same ensemble size.

### 5. Run each classifier shape group independently

Apply the same structure to `predict_proba_batched`:

1. build all independently preprocessed `ClassifierBatch` items;
2. stable-group them by the shared shape-signature helper;
3. inference-safely collate one group at a time;
4. call `fit_from_preprocessed` and `forward` for that group;
5. convert to NumPy and assign every group lane to its original index;
6. return one stacked `(n_datasets, n_test, n_classes)` array.

Preallocate a result list rather than a NumPy array so dtype/shape validation
can happen after all groups finish. Before stacking, assert that no slot is
empty and that every probability array has the expected test-row/class shape.

The existing shared-class-set validation must remain before preprocessing.
Per-dataset class permutations remain attached to the grouped configs exactly
as in the current single-batch path.

### 6. Update contracts and comments

Update both public docstrings to distinguish two constraints:

- callers must still supply equal raw train/test shapes;
- if preprocessing produces heterogeneous model-input shapes, TabPFN handles
  them as separate fused groups internally.

Remove or rewrite comments that claim the one global collator is harmless for
same-shaped raw inputs. State explicitly that raw shape does not imply
post-preprocessing shape because fitted transforms can remove constants or
choose data-dependent output widths.

Add a `fixed` changelog fragment describing batch-composition-dependent
predictions in `predict_batched` and `predict_proba_batched`.

## Test plan

### Shared helper tests

Add focused unit tests for the new utilities:

- identical complete signatures form one group;
- differing feature widths in any ensemble member form separate groups;
- differing context/query/y row shapes form separate groups;
- interleaved signatures retain first-seen group order and original item order;
- the inference-safe collator rejects heterogeneous signatures before calling
  `pad_tensors`;
- the inference-safe collator returns the same structure as
  `meta_dataset_collator` for a homogeneous group;
- `meta_dataset_collator` itself continues padding ragged fine-tuning inputs,
  protecting its existing contract.

### Regressor interface tests

Construct at least three raw datasets with identical `(n_rows, n_features)`
shapes but different constant-feature patterns, so fitted preprocessing yields
two or more shape signatures. Interleave the signatures in the input list.

Assert that:

- `predict_batched` matches independent `fit`/`predict` references within the
  established device tolerance for every dataset;
- output order matches input order after grouping;
- datasets with the same signature share one fused call;
- datasets with different signatures do not share a fused call;
- a duplicate dataset remains equivalent to its singleton reference;
- all supported output types (`mean`, quantiles, `main`, and `full`) retain
  their types and shapes;
- `full` output keeps each dataset's own criterion/bar distribution;
- constant-target datasets are filled analytically and do not disturb the
  grouping/index mapping;
- fitted target transforms, including the `n_preprocessing_jobs > 1` path,
  remain associated with the correct dataset;
- the estimator remains unmodified after the call;
- existing rejection of unequal raw train/test shapes remains unchanged.

Use deterministic dummy model specs where they exercise preprocessing and the
batched executor faithfully. Include one integration test with a stock
checkpoint in the GPU suite because the bug depends on real fitted
preprocessing widths.

### Classifier interface tests

Use balanced shuffled labels with at least three classes so probability changes
are not hidden by saturation near 0 or 1. Construct the same heterogeneous
constant-feature patterns as the regressor test.

Assert that:

- `predict_proba_batched` matches independent `fit`/`predict_proba` references;
- result shape remains `(n_datasets, n_test, n_classes)`;
- rows sum to one after regrouping;
- input order is restored for interleaved signatures;
- homogeneous signatures are fused and heterogeneous signatures are separated;
- class permutations/configs remain aligned with their datasets;
- DataFrame, categorical, missing-value, and GPU-preprocessing paths still
  behave like serial prediction;
- differing class sets, balancing, tuning, float64, and unequal raw shapes keep
  their current errors;
- the estimator remains unmodified.

### Architecture and device coverage

The implementation belongs to the package-level batching path and should not
branch on architecture version. Validate at least:

- stock/default v3 regressor and classifier;
- one older supported architecture if its preprocessing path differs;
- CPU for deterministic unit/interface coverage;
- CUDA smoke coverage on L4 and RTX, using tolerances rather than bitwise
  equality.

The CUDA smoke should include singleton, duplicate, matched-width, and
heterogeneous-width controls and record the post-preprocessing signatures in
failure messages.

## Performance and memory validation

Grouping trades one unsafe large fused call for one fused call per unique shape
signature. Correctness takes priority, but measure the impact explicitly:

- record number and sizes of shape groups for representative local-series and
  synthetic workloads;
- compare total runtime and peak GPU memory with the current homogeneous case;
- confirm homogeneous workloads still execute exactly one fused forward per
  estimator and do not acquire extra tensor copies beyond signature bookkeeping;
- confirm model weights are loaded once, not once per group;
- test a worst case where every dataset has a unique signature and document
  that it degrades to several safe smaller batches rather than serial model
  reloads;
- retain existing OOM handling at the group level and include the group's model
  input shape in any enhanced diagnostic message.

No warning should be emitted merely because multiple groups are needed. This is
a supported correctness path, not an exceptional caller error.

## Alternatives considered

### Reject heterogeneous post-preprocessing shapes

This is safe and a reasonable minimal hotfix, but it needlessly rejects raw
equal-shaped inputs that the API can support by issuing multiple fused calls.
It may be useful as the inference-collator invariant, but not as the final
public behavior.

### Disable data-dependent constant-feature removal

This changes preprocessing semantics and serial predictions, and other fitted
transforms can still produce data-dependent widths. It treats one trigger
rather than the batching invariant.

### Teach every architecture to mask padded feature columns

This is substantially broader, requires mask propagation through feature
grouping and attention, and risks checkpoint-dependent behavior. Grouping
requires no architecture changes and exactly preserves each dataset's serial
feature set.

### Change `meta_dataset_collator` to reject all ragged inputs

The collator is also a fine-tuning utility whose padding behavior is intentional.
Changing it globally would be a backwards-incompatible fix in the wrong layer.

## Acceptance criteria

The implementation is complete when:

1. Neither public batched prediction API ever sends feature-padded tensors into
   a model forward pass.
2. Equal raw shapes with heterogeneous post-preprocessing shapes are handled by
   separate stable groups without changing output order.
3. Regression and classification heterogeneous-width reproductions match
   independent serial predictions within existing numerical tolerances.
4. Homogeneous batches remain fused in one call per estimator.
5. Fine-tuning collation retains its existing padding behavior.
6. Constant targets, target transforms, class permutations, output variants,
   and estimator immutability remain covered and passing.
7. Targeted CPU tests, the existing regressor/classifier interface suites, lint,
   type checks required by CI, and L4/RTX CUDA smoke tests pass.
8. Documentation and the changelog state that compatibility is determined after
   preprocessing and handled internally.

## Suggested implementation sequence

1. Add failing regressor and classifier reproductions using equal raw shapes and
   heterogeneous constant-feature patterns.
2. Add shape-signature, stable-grouping, and inference-safe-collation helpers
   with unit tests.
3. Refactor `predict_batched` to execute and stitch shape groups.
4. Refactor `predict_proba_batched` onto the same helpers.
5. Expand edge-case tests for constants, transforms, classes, output types, and
   order preservation.
6. Run targeted CPU suites and formatting/static checks.
7. Run stock v3 CUDA integration tests on L4 and RTX.
8. Benchmark homogeneous and multi-signature workloads, document the expected
   performance trade-off, update docstrings/changelog, and move the PR out of
   draft once all acceptance criteria pass.
