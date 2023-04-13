from argparse import ArgumentParser, Namespace, _SubParsersAction
from importlib.metadata import version
from pathlib import Path

from nlp_final.args import AlphabeticalOrderHelpFormatter, authors


def getArgs() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="LUC COMP 429 (NLP) Final: Victorian Author Document Classifier",
        usage=None,
        description="A program to train and test models meant to classify Victorian era documents.",
        epilog=f"Authors: {', '.join(authors)}",
        formatter_class=AlphabeticalOrderHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"NLP Final: {version('nlp-final')}",
    )

    subparsers: _SubParsersAction = parser.add_subparsers(
        title="Operation Mode",
        description="Options to run the program in model training or model infrence mode",
        dest="mode",
        required=True,
    )

    trainingMode: ArgumentParser = subparsers.add_parser(
        name="train",
        help="Set the program to to run in model training mode",
        formatter_class=AlphabeticalOrderHelpFormatter,
    )

    trainingSubparsers: _SubParsersAction = trainingMode.add_subparsers(
        title="Model Type Training",
        description="Options to specify what type of model to train",
        dest="modelType",
        required=True,
    )

    vectorizer: ArgumentParser = trainingSubparsers.add_parser(
        name="vectorizer",
        help="Set the program to train different types of vectorizers",
        formatter_class=AlphabeticalOrderHelpFormatter,
    )

    vectorizer.add_argument(
        "--tf-idf",
        action="store_true",
        help="Train a TF-IDF vectorizer",
        dest="vectorizerTrainTFIDF",
    )

    vectorizer.add_argument(
        "--word2vec",
        action="store_true",
        help="Train a Word2Vec vectorizer",
        dest="vectorizerTrainWord2Vec",
    )

    vectorizer.add_argument(
        "--doc2vec",
        action="store_true",
        help="Train a Doc2Vec vectorizer",
        dest="vectorizerTrainDoc2Vec",
    )

    vectorizer.add_argument(
        "--fasttext",
        action="store_true",
        help="Train a FastText vectorizer",
        dest="vectorizerTrainFastText",
    )

    vectorizer.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of training jobs to run in parallel (default: 1)",
        dest="vectorizerJobs",
    )

    vectorizer.add_argument(
        "-o",
        "--output",
        default=Path("model"),
        type=Path,
        required=False,
        help="Directory to save the vectorizers to (default: ./model/)",
        dest="vectorizerOutput",
    )

    trainingMode.add_argument(
        "-o",
        "--output",
        default=Path("model"),
        type=Path,
        required=False,
        help="Directory to save models to",
    )

    trainingMode.add_argument(
        "--training-dataset",
        type=Path,
        required=True,
        help="Training dataset to use",
    )

    infrenceMode: ArgumentParser = subparsers.add_parser(
        name="infrence",
        help="Set the program to to run in infrence training mode",
        formatter_class=AlphabeticalOrderHelpFormatter,
    )

    return parser.parse_args()
