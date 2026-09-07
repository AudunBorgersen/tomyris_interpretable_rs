# Copyright (c) 2023 Ole-Christoffer Granmo
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
import typing

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tmu.models.base import MultiWeightBankMixin, SingleClauseBankMixin, TMBaseModel
from tmu.util.encoded_data_cache import DataEncoderCache
from tmu.weight_bank import WeightBank
import numpy as np


class TMCoalescedClassifier(TMBaseModel, SingleClauseBankMixin, MultiWeightBankMixin):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        type_i_ii_ratio=1.0,
        type_iii_feedback=False,
        focused_negative_sampling=False,
        output_balancing=False,
        d=200.0,
        platform="CPU",
        patch_dim=None,
        feature_negation=True,
        boost_true_positive_feedback=1,
        reuse_random_feedback=0,
        max_positive_clauses=None,
        max_included_literals=None,
        mask_target_literal_p=0.0,
        literal_feedback_drop_p=None,
        number_of_state_bits_ta=8,
        number_of_state_bits_ind=8,
        weighted_clauses=False,
        clause_drop_p=0.0,
        literal_drop_p=0.0,
        seed=None,
    ):
        super().__init__(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            type_i_ii_ratio=type_i_ii_ratio,
            type_iii_feedback=type_iii_feedback,
            focused_negative_sampling=focused_negative_sampling,
            output_balancing=output_balancing,
            d=d,
            platform=platform,
            patch_dim=patch_dim,
            feature_negation=feature_negation,
            boost_true_positive_feedback=boost_true_positive_feedback,
            reuse_random_feedback=reuse_random_feedback,
            max_included_literals=max_included_literals,
            number_of_state_bits_ta=number_of_state_bits_ta,
            number_of_state_bits_ind=number_of_state_bits_ind,
            weighted_clauses=weighted_clauses,
            clause_drop_p=clause_drop_p,
            literal_drop_p=literal_drop_p,
            seed=seed,
        )
        SingleClauseBankMixin.__init__(self)
        MultiWeightBankMixin.__init__(self, seed=seed)

        # These data structures cache the encoded data for the training and test sets. It also makes a fast-check if
        # training data has changed, and only re-encodes if it has.
        self.test_encoder_cache = DataEncoderCache(seed=self.seed)
        self.train_encoder_cache = DataEncoderCache(seed=self.seed)

        self.max_positive_clauses = max_positive_clauses
        self.mask_target_literal_p = mask_target_literal_p

        # Optional (number_of_features,) array: per-feature probability that a
        # *present* feature (bit = 1) is hidden from feedback for a given
        # training sample. Hidden literals are cleared from literal_active for
        # that sample only: their TAs receive no Type I reward, no Type I
        # erosion and no Type II introduction, while clause evaluation is
        # unaffected (the bit is still 1 in the encoded input, so conjunctions
        # containing it keep firing). This hides evidence without fabricating
        # absence — the principled alternative to zeroing input bits, which
        # reads as "user disliked X" and collapses clauses to single literals.
        # Fresh Bernoulli draws are made per sample. See README "Modifications
        # to TMU".
        self.literal_feedback_drop_p = literal_feedback_drop_p
        self._feedback_drop_cols = None

    def init_clause_bank(self, X: np.ndarray, Y: np.ndarray):
        clause_bank_type, clause_bank_args = self.build_clause_bank(X=X)
        self.clause_bank = clause_bank_type(**clause_bank_args)

    def init_weight_bank(self, X: np.ndarray, Y: np.ndarray):
        self.number_of_classes = int(np.max(Y) + 1)
        self.weight_banks.set_clause_init(
            WeightBank,
            dict(
                weights=self.rng.choice([-1, 1], size=self.number_of_clauses).astype(
                    np.int32
                )
            ),
        )
        self.weight_banks.populate(list(range(self.number_of_classes)))

    def init_after(self, X: np.ndarray, Y: np.ndarray):
        if self.max_included_literals is None:
            self.max_included_literals = self.clause_bank.number_of_literals

        if self.max_positive_clauses is None:
            self.max_positive_clauses = self.number_of_clauses

    def update(self, target, e, encoded_X_train):
        # Feedback-masked literal subsampling: hide a Bernoulli draw of the
        # eligible *present* features from this sample's feedback. Only bits
        # that are 1 in the input are ever masked, so clause evaluation is
        # unchanged (a present literal satisfies its clause with or without
        # the mask) — the mask solely silences TA updates for those literals.
        base_literal_active = self.literal_active
        if self._feedback_drop_cols is not None:
            present = self._feedback_drop_present[e]
            if present.any():
                cand = self._feedback_drop_cols[present]
                dropped = cand[
                    self.rng.rand(cand.size) < self._feedback_drop_p[present]
                ]
                if dropped.size:
                    base_literal_active = self.literal_active.copy()
                    # Clear both polarities: the negated literal of a hidden
                    # feature is the same unknown fact. (Inert when
                    # feature_negation=False — that half is already inactive.)
                    lits = np.concatenate(
                        [dropped, dropped + self.clause_bank.number_of_features]
                    )
                    bits = np.uint32(1) << (lits % 32).astype(np.uint32)
                    np.bitwise_and.at(base_literal_active, lits // 32, ~bits)

        clause_outputs = self.clause_bank.calculate_clause_outputs_update(
            base_literal_active, encoded_X_train, e
        )

        class_sum = np.dot(
            self.clause_active * self.weight_banks[target].get_weights(), clause_outputs
        ).astype(np.int32)
        class_sum = np.clip(class_sum, -self.T, self.T)
        update_p = (self.T - class_sum) / (2 * self.T)

        # randint(2) is what choice(2) delegates to internally — same draw,
        # without the per-call Python overhead of choice().
        type_iii_feedback_selection = self.rng.randint(2)

        # The target column is masked to 0 in every training input (build_samples
        # formulation).  Two spurious signals arise from this:
        #   1. The negated target literal (~X[target]) is trivially always 1 →
        #      exclude it so the TM cannot reward that vacuous signal.
        #   2. The positive target literal (X[target]) is trivially always 0 →
        #      exclude it so Type I does not penalise clauses that capture tight
        #      item clusters (e.g. "likes EP IV ∧ EP V ∧ EP VI") solely because
        #      the target item itself appears absent in its own training samples.
        neg_ta = self.clause_bank.number_of_features + target
        target_masked_literal_active = base_literal_active.copy()
        target_masked_literal_active[neg_ta // 32] &= ~np.uint32(1 << (neg_ta % 32))
        if (
            self.mask_target_literal_p > 0.0
            and self.rng.rand() < self.mask_target_literal_p
        ):
            target_masked_literal_active[target // 32] &= ~np.uint32(1 << (target % 32))

        self.clause_bank.type_i_feedback(
            update_p=update_p * self.type_i_p,
            clause_active=self.clause_active
            * (self.weight_banks[target].get_weights() >= 0),
            literal_active=target_masked_literal_active,
            encoded_X=encoded_X_train,
            e=e,
        )

        self.clause_bank.type_ii_feedback(
            update_p=update_p * self.type_ii_p,
            clause_active=self.clause_active
            * (self.weight_banks[target].get_weights() < 0),
            literal_active=target_masked_literal_active,
            encoded_X=encoded_X_train,
            e=e,
        )

        if (
            self.weight_banks[target].get_weights() >= 0
        ).sum() < self.max_positive_clauses:
            self.weight_banks[target].increment(
                clause_output=clause_outputs,
                update_p=update_p,
                clause_active=self.clause_active,
                positive_weights=True,
            )
            self._weight_matrix[target] = self.weight_banks[target].get_weights()

        if self.type_iii_feedback and type_iii_feedback_selection == 0:
            self.clause_bank.type_iii_feedback(
                update_p=update_p,
                clause_active=self.clause_active
                * (self.weight_banks[target].get_weights() >= 0),
                literal_active=target_masked_literal_active,
                encoded_X=encoded_X_train,
                e=e,
                target=1,
            )

            self.clause_bank.type_iii_feedback(
                update_p=update_p,
                clause_active=self.clause_active
                * (self.weight_banks[target].get_weights() < 0),
                literal_active=target_masked_literal_active,
                encoded_X=encoded_X_train,
                e=e,
                target=0,
            )

        # All class sums for this sample, read straight from the maintained
        # cache instead of rebuilding the weight matrix on every update.
        # The matvec runs in float64 because numpy has no BLAS path for
        # integer dtypes (~90x slower); float64 is exact for int32 weights.
        active_outputs = (self.clause_active * clause_outputs).astype(np.float64)
        self.update_ps = np.clip(self._weight_matrix @ active_outputs, -self.T, self.T)
        self.update_ps = (self.T + self.update_ps) / (2 * self.T)
        self.update_ps[target] = 0.0

        cumulative_ps = np.cumsum(self.update_ps)
        total_p = cumulative_ps[-1]
        if total_p == 0:
            return

        if self.focused_negative_sampling:
            # Inverse-CDF sampling; same single uniform draw and selection
            # probabilities as rng.choice(n, p=...) without its per-call
            # validation/normalization overhead.
            not_target = min(
                int(
                    np.searchsorted(
                        cumulative_ps, self.rng.rand() * total_p, side="right"
                    )
                ),
                self.number_of_classes - 1,
            )
            update_p = self.update_ps[not_target]
        else:
            not_target = self.rng.randint(self.number_of_classes)
            while not_target == target:
                not_target = self.rng.randint(self.number_of_classes)
            update_p = self.update_ps[not_target]

        # Phase 2: mirror the Phase 1 masking for not_target.  Clauses for class j
        # that include literal j are useless for recommendation (FOR) or actively
        # harmful (AGAINST — fires on j's own fans and penalises j for them).
        # Independent draw keeps the two decisions decorrelated.
        not_target_literal_active = base_literal_active
        if (
            self.mask_target_literal_p > 0.0
            and self.rng.rand() < self.mask_target_literal_p
        ):
            not_target_literal_active = base_literal_active.copy()
            not_target_literal_active[not_target // 32] &= ~np.uint32(
                1 << (not_target % 32)
            )

        self.clause_bank.type_i_feedback(
            update_p=update_p * self.type_i_p,
            clause_active=self.clause_active
            * (self.weight_banks[not_target].get_weights() < 0),
            literal_active=not_target_literal_active,
            encoded_X=encoded_X_train,
            e=e,
        )

        self.clause_bank.type_ii_feedback(
            update_p=update_p * self.type_ii_p,
            clause_active=self.clause_active
            * (self.weight_banks[not_target].get_weights() >= 0),
            literal_active=not_target_literal_active,
            encoded_X=encoded_X_train,
            e=e,
        )

        if self.type_iii_feedback and type_iii_feedback_selection == 1:
            self.clause_bank.type_iii_feedback(
                update_p=update_p,
                clause_active=self.clause_active
                * (self.weight_banks[not_target].get_weights() < 0),
                literal_active=not_target_literal_active,
                encoded_X=encoded_X_train,
                e=e,
                target=1,
            )

            self.clause_bank.type_iii_feedback(
                update_p=update_p,
                clause_active=self.clause_active
                * (self.weight_banks[not_target].get_weights() >= 0),
                literal_active=not_target_literal_active,
                encoded_X=encoded_X_train,
                e=e,
                target=0,
            )

        self.weight_banks[not_target].decrement(
            clause_output=clause_outputs,
            update_p=update_p,
            clause_active=self.clause_active,
            negative_weights=True,
        )
        self._weight_matrix[not_target] = self.weight_banks[not_target].get_weights()

    def fit(self, X, Y, shuffle=True, **kwargs):
        self.init(X, Y)

        # Per-fit structures for feedback-masked literal subsampling: the
        # eligible feature columns (drop_p > 0), their probabilities, and a
        # boolean presence matrix restricted to those columns so update() can
        # cheaply find which eligible features are present in sample e.
        self._feedback_drop_cols = None
        if self.literal_feedback_drop_p is not None:
            if self.patch_dim is not None:
                raise ValueError(
                    "literal_feedback_drop_p is only supported for flat "
                    "(non-convolutional) inputs."
                )
            p = np.asarray(self.literal_feedback_drop_p, dtype=np.float64)
            if p.shape != (self.clause_bank.number_of_features,):
                raise ValueError(
                    f"literal_feedback_drop_p must have shape "
                    f"({self.clause_bank.number_of_features},), got {p.shape}."
                )
            cols = np.flatnonzero(p > 0)
            if cols.size:
                self._feedback_drop_cols = cols
                self._feedback_drop_p = p[cols]
                self._feedback_drop_present = X[:, cols] != 0

        # Cached (n_classes, n_clauses) copy of every per-class weight vector,
        # kept in sync with the weight banks inside update(). Rebuilt here each
        # fit() so it reflects weights accumulated over previous epochs. This
        # replaces an O(n_classes x n_clauses) rebuild that previously ran on
        # every training sample (the dominant per-epoch cost).
        # float64 so the per-update class-sum matvec hits BLAS (exact for
        # int32 weights); rows are written back as int32 inside update().
        self._weight_matrix = np.array(
            [self.weight_banks[i].get_weights() for i in range(self.number_of_classes)],
            dtype=np.float64,
        )

        encoded_X_train = self.train_encoder_cache.get_encoded_data(
            X, encoder_func=lambda x: self.clause_bank.prepare_X(X)
        )

        Ym = np.ascontiguousarray(Y).astype(np.uint32)

        # Drops clauses randomly based on clause drop probability
        self.clause_active = (
            self.rng.rand(self.number_of_clauses) >= self.clause_drop_p
        ).astype(np.int32)

        # Literals are dropped based on literal drop probability
        self.literal_active = np.zeros(
            self.clause_bank.number_of_ta_chunks, dtype=np.uint32
        )
        literal_active_integer = (
            self.rng.rand(self.clause_bank.number_of_literals) >= self.literal_drop_p
        )
        for k in range(self.clause_bank.number_of_literals):
            if literal_active_integer[k] == 1:
                ta_chunk = k // 32
                chunk_pos = k % 32
                self.literal_active[ta_chunk] |= 1 << chunk_pos

        if not self.feature_negation:
            for k in range(
                self.clause_bank.number_of_literals // 2,
                self.clause_bank.number_of_literals,
            ):
                ta_chunk = k // 32
                chunk_pos = k % 32
                self.literal_active[ta_chunk] &= ~np.uint32(1 << chunk_pos)

        self.literal_active = self.literal_active.astype(np.uint32)

        self.update_ps = np.empty(self.number_of_classes)

        shuffled_index = np.arange(X.shape[0])
        if shuffle:
            self.rng.shuffle(shuffled_index)

        class_observed = np.zeros(self.number_of_classes, dtype=np.uint32)
        example_indexes = np.zeros(self.number_of_classes, dtype=np.uint32)
        # Cold items (never appear as labels) would make the output_balancing
        # trigger unreachable if we compared against number_of_classes.
        n_active_classes = len(np.unique(Ym))
        example_counter = 0
        for e in shuffled_index:
            if self.output_balancing:
                if class_observed[Ym[e]] == 0:
                    example_indexes[Ym[e]] = e
                    class_observed[Ym[e]] = 1
                    example_counter += 1
            else:
                example_indexes[example_counter] = e
                example_counter += 1

            if example_counter == (
                n_active_classes if self.output_balancing else self.number_of_classes
            ):
                example_counter = 0

                for i in range(self.number_of_classes):
                    class_observed[i] = 0
                    batch_example = example_indexes[i]
                    self.update(Ym[batch_example], batch_example, encoded_X_train)
        return

    def predict(
        self, X, clip_class_sum=False, return_class_sums: bool = False, **kwargs
    ):

        # Caching the encoded test set if it's not cached already
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(X)
            self.X_test = X.copy()

        # Clause outputs per sample (C call, reused buffer — copy each row),
        # then one BLAS matmul against all class weights at once instead of a
        # number_of_classes Python loop of np.dot per sample.
        clause_outputs = np.empty(
            (X.shape[0], self.number_of_clauses), dtype=np.float64
        )
        for e in range(X.shape[0]):
            clause_outputs[e] = self.clause_bank.calculate_clause_outputs_predict(
                self.encoded_X_test, e
            )

        weights = np.array(
            [self.weight_banks[i].get_weights() for i in range(self.number_of_classes)],
            dtype=np.float64,
        )
        class_sums = (clause_outputs @ weights.T).astype(np.int32)
        if clip_class_sum:
            class_sums = np.clip(class_sums, -self.T, self.T)

        # Find the class with the maximum sum for each sample
        max_classes = np.argmax(class_sums, axis=1)

        if return_class_sums:
            return max_classes, class_sums
        else:
            return max_classes

    def compute_class_sums(
        self, encoded_X_test: np.array, ith_sample: int, clip_class_sum: bool
    ) -> typing.List[int]:
        """The following function evaluates the resulting class sum votes.

        Args:
            ith_sample (int): The index of the sample
            clip_class_sum (bool): Wether to clip class sums

        Returns:
            list[int]: list of all class sums
        """
        class_sums = []
        clause_outputs = self.clause_bank.calculate_clause_outputs_predict(
            encoded_X_test, ith_sample
        )
        for i in range(self.number_of_classes):
            class_sum = np.dot(
                self.weight_banks[i].get_weights(), clause_outputs
            ).astype(np.int32)

            if clip_class_sum:
                class_sum = np.clip(class_sum, -self.T, self.T)
            class_sums.append(class_sum)
        return class_sums

    """def predict(self, X, clip_class_sum=False, return_class_sums: bool = False, **kwargs):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(X)
            self.X_test = X.copy()

        Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))

        for e in range(X.shape[0]):
            max_class_sum = -self.T
            max_class = 0
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test, e)
            for i in range(self.number_of_classes):
                class_sum = np.dot(self.weight_banks[i].get_weights(), clause_outputs).astype(np.int32)

                if clip_class_sum:
                    class_sum = np.clip(class_sum, -self.T, self.T)

                if class_sum > max_class_sum:
                    max_class_sum = class_sum
                    max_class = i
            Y[e] = max_class
        return Y"""

    def predict_individual(self, X):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(X)
            self.X_test = X.copy()

        Y = np.ascontiguousarray(
            np.zeros((X.shape[0], self.number_of_classes), dtype=np.uint32)
        )

        for e in range(X.shape[0]):
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(
                self.encoded_X_test, e
            )
            for i in range(self.number_of_classes):
                class_sum = np.dot(
                    self.weight_banks[i].get_weights(), clause_outputs
                ).astype(np.int32)
                class_sum = np.clip(class_sum, -self.T, self.T)
                Y[e, i] = class_sum >= 0
        return Y

    def clause_precision(self, the_class, X, Y):
        clause_outputs = self.transform(X)
        weights = self.weight_banks[the_class].get_weights()

        positive_clause_outputs = (weights >= 0)[
            :, np.newaxis
        ].transpose() * clause_outputs
        true_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(
            axis=0
        )
        false_positive_clause_outputs = positive_clause_outputs[Y != the_class].sum(
            axis=0
        )

        positive_clause_outputs = (weights < 0)[
            :, np.newaxis
        ].transpose() * clause_outputs
        true_positive_clause_outputs += positive_clause_outputs[Y != the_class].sum(
            axis=0
        )
        false_positive_clause_outputs += positive_clause_outputs[Y == the_class].sum(
            axis=0
        )

        return np.where(
            true_positive_clause_outputs + false_positive_clause_outputs == 0,
            0,
            1.0
            * true_positive_clause_outputs
            / (true_positive_clause_outputs + false_positive_clause_outputs),
        )

    def clause_recall(self, the_class, X, Y):
        clause_outputs = self.transform(X)
        weights = self.weight_banks[the_class].get_weights()

        positive_clause_outputs = (weights >= 0)[
            :, np.newaxis
        ].transpose() * clause_outputs
        true_positive_clause_outputs = (
            positive_clause_outputs[Y == the_class].sum(axis=0)
            / Y[Y == the_class].shape[0]
        )

        positive_clause_outputs = (weights < 0)[
            :, np.newaxis
        ].transpose() * clause_outputs
        true_positive_clause_outputs += (
            positive_clause_outputs[Y != the_class].sum(axis=0)
            / Y[Y != the_class].shape[0]
        )

        return true_positive_clause_outputs

    def get_weights(self, the_class):
        return self.weight_banks[the_class].get_weights()

    def get_weight(self, the_class, clause):
        return self.weight_banks[the_class].get_weights()[clause]

    def set_weight(self, the_class, clause, weight):
        self.weight_banks[the_class].get_weights()[clause] = weight

    def number_of_include_actions(self, clause):
        return self.clause_bank.number_of_include_actions(clause)

    def __getstate__(self):
        state = self.__dict__.copy()
        # encoded_X_test holds a pycuda DeviceAllocation after the first predict()
        # call and cannot be pickled. Drop it so the next predict() re-encodes.
        state.pop("encoded_X_test", None)
        state["X_test"] = np.zeros(0, dtype=np.uint32)
        # _weight_matrix is a derived cache, rebuilt at the start of each fit();
        # no need to persist it (and it would bloat checkpoints).
        state.pop("_weight_matrix", None)
        # Feedback-drop structures are rebuilt per fit() from
        # literal_feedback_drop_p; the presence matrix in particular is
        # (n_samples, n_eligible) and would bloat checkpoints.
        state.pop("_feedback_drop_present", None)
        state.pop("_feedback_drop_p", None)
        state["_feedback_drop_cols"] = None
        return state
