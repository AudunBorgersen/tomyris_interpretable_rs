import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from tmu.models.autoencoder.triplet_embedder import TMTripletEmbedder
from tmu.data import IMDB
from tmu.tools import BenchmarkTimer
import logging
import os
import argparse

os.environ["CUDA_PROFILE"] = "1"

_LOGGER = logging.getLogger(__name__)

target_words = [
    'awful',
    'terrible',
    'lousy',
    'abysmal',
    'crap',
    'outstanding',
    'brilliant',
    'excellent',
    'superb',
    'magnificent',
    'marvellous',
    'truck',
    'plane',
    'car',
    'cars',
    'motorcycle',
    'scary',
    'frightening',
    'terrifying',
    'horrifying',
    'funny',
    'comic',
    'hilarious',
    'witty'
]

# Define semantic groups for validation (NOT used in training)
sentiment_groups = {
    'negative': ['awful', 'terrible', 'lousy', 'abysmal', 'crap'],
    'positive': ['outstanding', 'brilliant', 'excellent', 'superb', 'magnificent', 'marvellous'],
    'vehicles': ['truck', 'plane', 'car', 'cars', 'motorcycle'],
    'fear': ['scary', 'frightening', 'terrifying', 'horrifying'],
    'humor': ['funny', 'comic', 'hilarious', 'witty']
}


def evaluate_clustering(tm, target_words, clause_weight_threshold=0):
    """
    Evaluate emergent clustering: Do semantically similar words cluster together?

    This is the key validation that the triplet embedder learns meaningful semantics.
    """
    _LOGGER.info("\n=== EMERGENT CLUSTERING EVALUATION ===")

    # Get embeddings
    profile = np.array(
        [np.where(tm.get_weights(i) >= clause_weight_threshold, tm.get_weights(i), 0) for i in
         range(len(target_words))])
    similarity = cosine_similarity(profile)

    # Calculate within-group vs between-group similarity
    results = {}

    for group_name, group_words in sentiment_groups.items():
        # Get indices of words in this group
        group_indices = [i for i, word in enumerate(target_words) if word in group_words]

        if len(group_indices) < 2:
            continue

        # Within-group similarity
        within_sims = []
        for i in range(len(group_indices)):
            for j in range(i + 1, len(group_indices)):
                idx_i = group_indices[i]
                idx_j = group_indices[j]
                within_sims.append(similarity[idx_i, idx_j])

        within_avg = np.mean(within_sims) if within_sims else 0.0

        # Between-group similarity (with other groups)
        between_sims = []
        for idx_i in group_indices:
            for idx_j in range(len(target_words)):
                if idx_j not in group_indices:
                    between_sims.append(similarity[idx_i, idx_j])

        between_avg = np.mean(between_sims) if between_sims else 0.0

        results[group_name] = {
            'within': within_avg,
            'between': between_avg,
            'ratio': within_avg / between_avg if between_avg > 0 else 0
        }

        _LOGGER.info(f"{group_name.upper()}: within={within_avg:.3f}, between={between_avg:.3f}, ratio={within_avg/between_avg:.3f}")

    # Overall success metric: average ratio across groups
    avg_ratio = np.mean([r['ratio'] for r in results.values()])
    _LOGGER.info(f"\nOVERALL CLUSTERING QUALITY: {avg_ratio:.3f} (>1.0 = success)")

    return results, avg_ratio


def metrics(args):
    return dict(
        triplet_accuracy=[],
        clustering_quality=[],
        train_time=[],
        test_time=[],
        precision=[],
        recall=[],
        args=vars(args)
    )


def main(args):
    experiment_results = metrics(args)

    # load IMDB dataset
    dataloader = IMDB(num_words=args.NUM_WORDS, index_from=args.INDEX_FROM)
    dataset = dataloader.get()
    train_x = dataset["x_train"]
    train_y = dataset["y_train"]
    test_x = dataset["x_test"]
    test_y = dataset["y_test"]

    # get word to indice and id_to_word mappings
    word_to_id = dataloader.get_word_index()
    word_to_id = {k: (v + args.INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value: key for key, value in word_to_id.items()}

    _LOGGER.info("Producing bit representation...")

    def produce_documents(dsx, dsy):
        docs = []
        for i in range(dsy.shape[0]):
            terms = [id_to_word[word_id].lower() for word_id in dsx[i]]
            docs.append(terms)
        return docs

    training_documents = produce_documents(train_x, train_y)
    testing_documents = produce_documents(test_x, test_y)

    # Create vectorizer
    def tokenizer(s):
        return s

    vectorizer_X = CountVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,
        lowercase=False,
        binary=True
    )

    X_train = vectorizer_X.fit_transform(training_documents)
    feature_names = vectorizer_X.get_feature_names_out()

    output_active = np.empty(len(target_words), dtype=np.uint32)
    for i in range(len(target_words)):
        target_word = target_words[i]
        target_id = vectorizer_X.vocabulary_[target_word]
        output_active[i] = target_id

    # Create TMTripletEmbedder
    tm = TMTripletEmbedder(
        number_of_clauses=args.clauses,
        T=args.T,
        s=args.s,
        output_active=output_active,
        margin=args.margin,
        triplet_ratio=args.triplet_ratio,
        max_included_literals=args.max_included_literals,
        accumulation=args.accumulation,
        feature_negation=args.feature_negation,
        platform=args.device,
        output_balancing=args.output_balancing,
        type_i_ii_ratio=args.type_i_ii_ratio
    )

    benchmark_train = BenchmarkTimer()
    benchmark_test = BenchmarkTimer()
    benchmark_total = BenchmarkTimer()

    _LOGGER.info(f"Training over {args.epochs} epochs:")
    for e in range(args.epochs):
        with benchmark_total:
            _LOGGER.info(f"\n{'='*60}")
            _LOGGER.info(f"Epoch {e + 1}")
            _LOGGER.info(f"{'='*60}")

            with benchmark_train:
                tm.fit(X_train, number_of_examples=args.number_of_examples)
            experiment_results["train_time"].append(benchmark_train.elapsed())

            with benchmark_test:
                # Evaluate triplet accuracy
                _LOGGER.info("\nEvaluating triplet accuracy...")
                triplet_acc = tm.evaluate_triplet_accuracy(num_samples=args.triplet_eval_samples)
                experiment_results["triplet_accuracy"].append(triplet_acc)
                _LOGGER.info(f"Triplet Accuracy: {triplet_acc:.2%}")

                # Evaluate emergent clustering
                clustering_results, clustering_quality = evaluate_clustering(
                    tm, target_words, clause_weight_threshold=args.clause_weight_threshold
                )
                experiment_results["clustering_quality"].append(clustering_quality)

                # Calculate precision and recall
                _LOGGER.info("\nCalculating precision and recall...")
                precision = [tm.clause_precision(i, True, X_train, number_of_examples=500) for i in
                             tqdm(range(len(target_words)), desc="Precision")]

                recall = [tm.clause_recall(i, True, X_train, number_of_examples=500) for i in
                          tqdm(range(len(target_words)), desc="Recall")]

                experiment_results["precision"].append(precision)
                experiment_results["recall"].append(recall)

                # Show clause interpretations
                if args.show_clauses:
                    _LOGGER.info("\n=== CLAUSE INTERPRETATIONS ===")
                    for j in range(min(args.clauses, 10)):  # Show first 10 clauses
                        clause_info = " ".join(
                            [f"{target_words[i]}:W{tm.get_weight(i, j)}:P{precision[i][j]:.2f}:R{recall[i][j]:.2f}"
                             for i in range(len(target_words))])
                        literals = ["{}{}({})".format("¬" if k >= tm.clause_bank.number_of_features else "",
                                                      feature_names[k % tm.clause_bank.number_of_features],
                                                      tm.clause_bank.get_ta_state(j, k)) for k in
                                    range(tm.clause_bank.number_of_literals) if tm.get_ta_action(j, k) == 1]
                        _LOGGER.info(f"Clause #{j} {clause_info}")
                        _LOGGER.info(f"  {' ∧ '.join(literals[:10])}")  # Show first 10 literals

                # Show word similarity matrix
                _LOGGER.info("\n=== WORD SIMILARITY MATRIX ===")
                similarity_matrix = tm.get_embedding_similarity_matrix(threshold=args.clause_weight_threshold)

                for i in range(len(target_words)):
                    sorted_index = np.argsort(-similarity_matrix[i, :])
                    similarity_info = " ".join(
                        [f"{target_words[sorted_index[j]]}({similarity_matrix[i, sorted_index[j]]:.2f})" for j in
                         range(1, min(6, len(target_words)))])  # Show top 5 similar words
                    _LOGGER.info(f"{target_words[i]}: {similarity_info}")

            experiment_results["test_time"].append(benchmark_test.elapsed())

        _LOGGER.info(f"\n{'='*60}")
        _LOGGER.info(f"Epoch {e+1} Summary:")
        _LOGGER.info(f"Total time: {benchmark_total.elapsed():.2f}s")
        _LOGGER.info(f"Training time: {benchmark_train.elapsed():.2f}s")
        _LOGGER.info(f"Testing time: {benchmark_test.elapsed():.2f}s")
        _LOGGER.info(f"Triplet Accuracy: {triplet_acc:.2%}")
        _LOGGER.info(f"Clustering Quality: {clustering_quality:.3f}")
        _LOGGER.info(f"{'='*60}\n")

    return experiment_results


def default_args(**kwargs):
    parser = argparse.ArgumentParser(description='TMTripletEmbedder Demo')

    # Model architecture
    parser.add_argument('--clauses', type=int, default=200, help='Number of clauses')
    parser.add_argument('--T', type=int, default=400, help='Threshold (T)')
    parser.add_argument('--s', type=float, default=5.0, help='Specificity (s)')
    parser.add_argument('--max_included_literals', type=int, default=5, help='Max included literals per clause')
    parser.add_argument('--feature_negation', type=bool, default=True, help='Feature negation')

    # Triplet-specific parameters
    parser.add_argument('--margin', type=float, default=1.0, help='Triplet margin')
    parser.add_argument('--triplet_ratio', type=float, default=0.5, help='Fraction of training for triplet phase (0.5 = 50%)')
    parser.add_argument('--type_i_ii_ratio', type=float, default=2.0, help='Type I to Type II feedback ratio')

    # Training parameters
    parser.add_argument('--number_of_examples', type=int, default=2000, help='Number of training examples per epoch')
    parser.add_argument('--accumulation', type=int, default=25, help='Context accumulation')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--output_balancing', type=float, default=0.5, help='Output balancing')

    # Dataset parameters
    parser.add_argument('--NUM_WORDS', type=int, default=10000, help='Number of words in vocabulary')
    parser.add_argument('--INDEX_FROM', type=int, default=2, help='Index from')

    # Evaluation parameters
    parser.add_argument('--clause_weight_threshold', type=int, default=0, help='Clause weight threshold for embeddings')
    parser.add_argument('--triplet_eval_samples', type=int, default=1000, help='Number of triplets to sample for evaluation')
    parser.add_argument('--show_clauses', type=bool, default=False, help='Show clause interpretations')

    # Hardware
    parser.add_argument('--device', type=str, default="CPU", help='Device (CPU or CUDA)')

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    results = main(default_args())

    _LOGGER.info("\n" + "="*60)
    _LOGGER.info("FINAL RESULTS")
    _LOGGER.info("="*60)
    _LOGGER.info(f"Final Triplet Accuracy: {results['triplet_accuracy'][-1]:.2%}")
    _LOGGER.info(f"Final Clustering Quality: {results['clustering_quality'][-1]:.3f}")
    _LOGGER.info(f"Average Training Time: {np.mean(results['train_time']):.2f}s")
    _LOGGER.info("="*60)
