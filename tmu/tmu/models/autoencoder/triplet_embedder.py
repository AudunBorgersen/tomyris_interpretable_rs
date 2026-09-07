# Copyright (c) 2023 Ole-Christoffer Granmo
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tmu.models.autoencoder.autoencoder import TMAutoEncoder
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.metrics.pairwise import cosine_similarity


class CooccurrenceTripletSampler:
    """
    Samples triplets (anchor, positive, negative) based on word co-occurrence statistics.

    Positive pairs: words with high co-occurrence
    Negative pairs: words with low co-occurrence
    """

    def __init__(self, co_occurrence_matrix, vocab_size, k_top=10, k_bottom=10):
        """
        Args:
            co_occurrence_matrix: Matrix where [i,j] = co-occurrence count of words i and j
            vocab_size: Number of words in vocabulary
            k_top: Sample positive from top-k most co-occurring words
            k_bottom: Sample negative from bottom-k least co-occurring words
        """
        self.co_occurrence = co_occurrence_matrix
        self.vocab_size = vocab_size
        self.k_top = k_top
        self.k_bottom = k_bottom
        self.rng = np.random.default_rng()

    def sample_from_top_k(self, scores, k):
        """Sample from top-k highest scoring indices"""
        if len(scores) <= k:
            return self.rng.integers(0, len(scores))

        # Get top-k indices (excluding zeros for co-occurrence)
        nonzero_indices = np.where(scores > 0)[0]
        if len(nonzero_indices) == 0:
            # No co-occurrence found, random sample
            return self.rng.integers(0, len(scores))

        if len(nonzero_indices) <= k:
            return self.rng.choice(nonzero_indices)

        top_k_indices = np.argpartition(scores, -k)[-k:]
        return self.rng.choice(top_k_indices)

    def sample_from_bottom_k(self, scores, k):
        """Sample from bottom-k lowest scoring indices"""
        if len(scores) <= k:
            return self.rng.integers(0, len(scores))

        # For negatives, we want low co-occurrence (including zeros)
        bottom_k_indices = np.argpartition(scores, k)[:k]
        return self.rng.choice(bottom_k_indices)

    def sample(self):
        """
        Sample a triplet (anchor, positive, negative)

        Returns:
            tuple: (anchor_idx, positive_idx, negative_idx)
        """
        # Random anchor
        anchor = self.rng.integers(0, self.vocab_size)

        # Positive: high co-occurrence with anchor
        cooccur_scores = self.co_occurrence[anchor]

        # Ensure we don't sample the anchor itself
        cooccur_scores_copy = cooccur_scores.copy()
        cooccur_scores_copy[anchor] = 0

        positive = self.sample_from_top_k(cooccur_scores_copy, self.k_top)

        # Negative: low co-occurrence with anchor
        negative = self.sample_from_bottom_k(cooccur_scores_copy, self.k_bottom)

        # Ensure positive != negative != anchor
        if positive == anchor:
            positive = (anchor + 1) % self.vocab_size
        if negative == anchor or negative == positive:
            negative = (anchor + 2) % self.vocab_size

        return (anchor, positive, negative)


class TMTripletEmbedder(TMAutoEncoder):
    """
    Text embedder using Tsetlin Machine with triplet loss.

    Combines standard autoencoder training with triplet-based contrastive learning
    to create semantically meaningful word embeddings.

    Training strategy:
    1. Warmup phase: Standard autoencoder learns basic word-context associations
    2. Triplet phase: Refines embeddings to push similar words together and dissimilar words apart
    """

    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        output_active,
        margin=1.0,
        triplet_ratio=0.5,
        k_top=10,
        k_bottom=10,
        **kwargs,
    ):
        """
        Args:
            number_of_clauses: Number of clauses (logical patterns)
            T: Threshold for voting
            s: Specificity parameter
            output_active: Array of word indices to learn embeddings for
            margin: Triplet margin (default: 1.0)
            triplet_ratio: Fraction of training for triplet phase (default: 0.5 = 50%)
            k_top: Sample positive from top-k co-occurring words
            k_bottom: Sample negative from bottom-k non-co-occurring words
            **kwargs: Additional parameters passed to TMAutoEncoder
        """
        # Set reduced Type II feedback by default (unless explicitly overridden)
        if "type_i_ii_ratio" not in kwargs:
            kwargs["type_i_ii_ratio"] = 2.0

        super().__init__(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            output_active=output_active,
            **kwargs,
        )

        self.margin = margin
        self.triplet_ratio = triplet_ratio
        self.k_top = k_top
        self.k_bottom = k_bottom

        # Will be initialized during fit()
        self.co_occurrence_matrix = None
        self.triplet_sampler = None

    def build_co_occurrence_matrix(self, X):
        """
        Build word co-occurrence matrix from training corpus.

        Counts how often word pairs appear together in the same document.
        Only computes co-occurrence for words in output_active.

        Args:
            X: Document-term matrix (sparse CSR format)

        Returns:
            co_occurrence_matrix: Matrix where [i,j] = co-occurrence count for output_active[i] and output_active[j]
        """
        X_csr = csr_matrix(X.reshape(X.shape[0], -1))

        # Extract only columns for words in output_active
        X_active = X_csr[:, self.output_active]

        # For each target word, count which other target words appear in the same documents
        # Co-occurrence = X_active^T @ X_active
        # Result: num_target_words x num_target_words matrix
        co_occurrence = X_active.T @ X_active

        # Zero out diagonal (word co-occurring with itself)
        co_occurrence.setdiag(0)

        # Convert to dense for easier sampling
        return co_occurrence.toarray()

    def _clause_similarity(self, weights_a, weights_b):
        """
        Calculate similarity between two weight vectors based on clause agreement.

        Counts how many clauses have weights with the same sign.

        Args:
            weights_a: Weight vector for word A
            weights_b: Weight vector for word B

        Returns:
            similarity: Number of agreeing clauses (higher = more similar)
        """
        # Count clauses where weights have same sign
        agreement = np.sum((weights_a * weights_b) > 0)
        return agreement

    def _triplet_update(self, anchor_idx, positive_idx, negative_idx):
        """
        Update embeddings based on triplet constraint.

        Goal: Make anchor-positive more similar than anchor-negative by at least margin.

        Strategy:
        - If constraint violated: align anchor-positive, separate anchor-negative
        - Alignment: Train both words on their typical contexts
        - Separation: Train anchor on contexts where negative is absent

        Args:
            anchor_idx: Index of anchor word
            positive_idx: Index of positive word (similar to anchor)
            negative_idx: Index of negative word (dissimilar to anchor)
        """
        # Get current weight vectors
        W_a = self.weight_banks[anchor_idx].get_weights()
        W_p = self.weight_banks[positive_idx].get_weights()
        W_n = self.weight_banks[negative_idx].get_weights()

        # Calculate similarities (clause agreement)
        sim_ap = self._clause_similarity(W_a, W_p)
        sim_an = self._clause_similarity(W_a, W_n)

        # Check triplet violation: sim(anchor, positive) should be > sim(anchor, negative) + margin
        triplet_loss = max(0, self.margin + sim_an - sim_ap)

        if triplet_loss > 0:
            # Generate clause and literal activations
            clause_active = self.activate_clauses()
            literal_active = self.activate_literals()

            # ALIGN: Push anchor-positive together
            # Train both words on their typical contexts (reinforces shared patterns)
            for word_idx in [anchor_idx, positive_idx]:
                Xu, Yu = self.clause_bank.produce_autoencoder_example(
                    encoded_X=self.encoded_X_train,
                    target=word_idx,
                    target_true_p=self.feature_true_probability[
                        self.output_active[word_idx]
                    ],
                    accumulation=self.accumulation,
                )
                self.update(word_idx, Yu, Xu, clause_active, literal_active)

            # SEPARATE: Pull anchor-negative apart
            # Train anchor on contexts where negative is ABSENT (creates divergent patterns)
            Xu_neg, Yu_neg = self.clause_bank.produce_autoencoder_example(
                encoded_X=self.encoded_X_train,
                target=negative_idx,
                target_true_p=1.0
                - self.feature_true_probability[self.output_active[negative_idx]],
                accumulation=self.accumulation,
            )
            # Invert target to get opposite contexts
            self.update(anchor_idx, 1 - Yu_neg, Xu_neg, clause_active, literal_active)

    def fit(self, X, number_of_examples=2000, shuffle=True, **kwargs):
        """
        Two-phase training: autoencoder warmup + triplet refinement.

        Phase 1: Warmup with standard autoencoder (builds initial embeddings)
        Phase 2: Triplet refinement (pushes similar words together, dissimilar apart)

        Args:
            X: Training data (document-term matrix)
            number_of_examples: Total training examples
            shuffle: Whether to shuffle class indices
        """
        # Prepare data matrices (needed for both phases)
        X_csr = csr_matrix(X.reshape(X.shape[0], -1))
        X_csc = csc_matrix(X.reshape(X.shape[0], -1)).sorted_indices()

        # Build co-occurrence matrix for triplet generation
        print("Building co-occurrence matrix...")
        self.co_occurrence_matrix = self.build_co_occurrence_matrix(X)

        # Initialize triplet sampler
        self.triplet_sampler = CooccurrenceTripletSampler(
            self.co_occurrence_matrix,
            len(self.output_active),
            k_top=self.k_top,
            k_bottom=self.k_bottom,
        )

        # Calculate training split
        warmup_examples = int(number_of_examples * (1 - self.triplet_ratio))
        triplet_examples = number_of_examples - warmup_examples

        # Phase 1: Autoencoder warmup
        print(f"Phase 1: Autoencoder warmup ({warmup_examples} examples)...")
        super().fit(X, number_of_examples=warmup_examples, shuffle=shuffle, **kwargs)

        # Ensure encoded_X_train is prepared for Phase 2
        # (should already be done by super().fit(), but double-check)
        if not hasattr(self, "encoded_X_train") or self.encoded_X_train is None:
            self.encoded_X_train = self.clause_bank.prepare_X_autoencoder(
                X_csr, X_csc, self.output_active
            )
            self.X_train = np.concatenate((X_csr.indptr, X_csr.indices))

        # Phase 2: Triplet refinement
        print(f"Phase 2: Triplet refinement ({triplet_examples} examples)...")
        for iteration in range(triplet_examples):
            anchor, positive, negative = self.triplet_sampler.sample()
            self._triplet_update(anchor, positive, negative)

            if (iteration + 1) % 100 == 0:
                print(f"  Triplet iteration {iteration + 1}/{triplet_examples}")

        print("Training complete!")
        return

    def evaluate_triplet_accuracy(self, num_samples=1000):
        """
        Measure triplet accuracy: what % of triplets satisfy the constraint?

        For randomly sampled triplets, checks if:
        distance(anchor, positive) < distance(anchor, negative)

        Args:
            num_samples: Number of triplets to sample

        Returns:
            accuracy: Fraction of satisfied triplets (0-1)
        """
        if self.triplet_sampler is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        correct = 0
        total = 0

        for _ in range(num_samples):
            anchor, positive, negative = self.triplet_sampler.sample()

            # Get embeddings (weight vectors)
            emb_a = self.weight_banks[anchor].get_weights()
            emb_p = self.weight_banks[positive].get_weights()
            emb_n = self.weight_banks[negative].get_weights()

            # Calculate distances (L2 norm)
            dist_ap = np.linalg.norm(emb_a - emb_p)
            dist_an = np.linalg.norm(emb_a - emb_n)

            if dist_ap < dist_an:
                correct += 1
            total += 1

        return correct / total

    def get_embedding_similarity_matrix(self, threshold=0):
        """
        Return cosine similarity matrix of all word embeddings.

        Args:
            threshold: Weight threshold for sparsity (default: 0 = use all weights)

        Returns:
            similarity_matrix: Pairwise cosine similarities (vocab_size x vocab_size)
        """
        # Extract all embeddings
        embeddings = np.array(
            [self.weight_banks[i].get_weights() for i in range(len(self.output_active))]
        )

        # Apply threshold for sparsity
        if threshold > 0:
            embeddings = np.where(np.abs(embeddings) >= threshold, embeddings, 0)

        # Calculate cosine similarity
        sim_matrix = cosine_similarity(embeddings)

        return sim_matrix

    def export_embeddings(self, word_list=None):
        """
        Export embeddings for external analysis (t-SNE, UMAP, etc.).

        Args:
            word_list: Optional list of word strings (for labeling)

        Returns:
            embeddings: numpy array (vocab_size x number_of_clauses)
            OR dict mapping words to vectors if word_list provided
        """
        embeddings = np.array(
            [self.weight_banks[i].get_weights() for i in range(len(self.output_active))]
        )

        if word_list is not None:
            return {word: embeddings[i] for i, word in enumerate(word_list)}

        return embeddings
