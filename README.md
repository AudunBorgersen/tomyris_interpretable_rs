# ToMyRiS Tsetlin Machine Recommender System
Repo accompanying the paper "ToMyRiS: Tsetlin Machines for Interpretable Embedding-based Recommender Systems".
TMs are trained via experiments/coalesced_baseline.py, and CF performance is evaluated using experiments/clause_cf_eval.py
Run configurations for TM training runs are located in configs/
Runs are saved to the runs/ folder, CF evaluations are carried out on the saved runs.

The remainder of the README details the modifications we've made to the original TMU repo (github.com/cair/tmu)

## Modifications to TMU
This implementation is built upon the TMU repository, which represents a consolidated implementation of several versions of the TM.

We've made a few modifications towards its functionality. All of them live in the
coalesced classifier (`tmu/tmu/models/classification/coalesced_classifier.py`).

### Target-literal masking (`mask_target_literal_p`)
For certain datasets, such as MovieLens 1M, no item may be interacted with twice.
Under our masked item-prediction formulation the target item's column is zeroed in
every training input. Its positive literal is therefore trivially always false and
its negated literal trivially always true. The former causes Type I feedback to
penalise clauses that capture tight item clusters which happen to include the
target item itself (self-inclusion), discouraging otherwise sensible co-taste
clauses. The new `mask_target_literal_p` parameter excludes the target item's
literals from feedback — the negated literal always, the positive literal with
probability `mask_target_literal_p` — applied to the target class in the first
phase of the update and to the sampled negative class in the second.

### Feedback-masked literal subsampling (`literal_feedback_drop_p`)
Clause antecedents draw their literals almost exclusively from the most popular
items: a literal only survives Type Ib erosion if it is frequently True among
the samples its clause matches, which puts an effective popularity bar on
inclusion. Word2vec-style subsampling of the *inputs* (zeroing popular bits)
flattens this frequency distribution, but at a structural cost we measured
directly: every zeroed bit fabricates a false negative ("user did not watch X"
about a user who did), and a k-literal conjunction is falsified whenever any of
its k members is dropped — a multiplicative penalty that collapses mean clause
size from ~2.9 literals to ~1.0.

`literal_feedback_drop_p` (an optional per-feature probability vector on
`TMCoalescedClassifier`) implements the same popularity flattening with
missing-not-false semantics. For each training sample, each *present* feature
(bit = 1) is hidden with its per-feature probability by clearing its bit — both
polarities — in that sample's `literal_active` mask. The existing feedback
kernels then guarantee, identically on CPU and CUDA (every `inc`/`dec` is gated
bitwise by `literal_active`), that the hidden literal receives no Type Ia
reward, no Type Ib erosion, and no Type II introduction for that sample.
Because only present bits are ever masked, clause evaluation is unchanged — the
encoded input still carries the 1, so conjunctions containing the hidden item
keep firing and their other literals train normally. Evidence is hidden, never
inverted.

Masks are redrawn per sample (fresh each epoch, since `fit` is called per
epoch). The implementation lives entirely in the coalesced classifier's
`update()` — it reuses the per-sample `literal_active` machinery introduced for
`mask_target_literal_p` — and requires no C/CUDA changes. With a zero or absent
vector the RNG stream is untouched, so existing runs reproduce exactly. Only
flat (non-convolutional) inputs are supported. Driven from
`experiments/coalesced_baseline.py` via `subsample_mode: feedback` +
`input_subsample_t`.

### Cached weight matrix
The unmodified implementation reconstructs the full (n_classes × n_clauses) weight
matrix from the individual per-class weight banks on *every training sample*, an
O(n_classes × n_clauses) operation that dominated per-epoch runtime. We instead
keep a contiguous weight array, rebuilt once per epoch and updated only on the two
rows that change per sample (the target and the sampled negative class). The change
is behaviour-preserving — identical weights and RNG stream — and roughly halves
per-epoch time in CPU profiling.

### BLAS class sums and cheaper negative sampling
Profiling (`scripts/profile_coalesced.py`) showed ~72% of remaining per-update time
went to the class-sum matvec `weight_matrix @ clause_outputs`: the matrix was
`int32`, and numpy has no BLAS path for integer dtypes, so the product ran in a
slow generic loop (~10.7 ms vs ~0.23 ms via BLAS at 3647 classes × 500 clauses).
Three changes, all in the coalesced classifier:

- **Float64 weight matrix.** The cached weight matrix is now `float64` so the
  per-update matvec hits BLAS. Float64 represents `int32` weights exactly, so
  class sums are unchanged, not just approximately equal.
- **Inverse-CDF negative sampling.** Focused negative sampling used
  `rng.choice(n, p=...)`, which re-validates and re-normalises the probability
  vector on every call. Replaced with one `cumsum` + `searchsorted`, consuming
  the same single uniform draw with the same selection probabilities. Likewise
  `rng.choice(2)` → `rng.randint(2)` (what `choice` delegates to internally).
- **Vectorised `predict()`.** Prediction ran a Python loop of `number_of_classes`
  `np.dot` calls per test sample; now it is one BLAS matmul per batch, which cuts
  per-epoch evaluation cost as well.

Verified bit-exact against the previous implementation (identical TA states,
weight banks, and class sums after two epochs with the same seed). Net effect in
CPU profiling: 14.5 → 4.7 ms/update (~3×); the remaining time is dominated by the
Type I feedback C kernel.

### Other fixes
- **`output_balancing` class count.** With output balancing enabled, the per-round
  batch trigger compared against `number_of_classes`. Cold items (classes that
  never appear as labels) mean fewer distinct classes are ever observed, so the
  trigger could be unreachable and a round never complete. It now triggers on the
  number of *active* classes (`len(np.unique(Y))`) when balancing is on.
- **No-negation compatibility.** When `feature_negation=False`, the negated-literal
  bits are cleared with an explicit `np.uint32` mask so `literal_active` stays a
  `uint32` array rather than being promoted to a Python int.
