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
        required=True,
    )

    trainingMode: ArgumentParser = subparsers.add_parser(
        name="train",
        help="Set the program to to run in model training mode",
        formatter_class=AlphabeticalOrderHelpFormatter,
    )

    trainingMode.add_argument(
        "--training-dataset",
        type=Path,
        required=True,
        help="Training dataset to use",
    )

    trainingMode.add_argument(
        "--naive-bayes",
        action="store_true",
        help="Train a naive bayes classifier",
    )

    trainingMode.add_argument(
        "--vectorizer",
        nargs=1,
        default="tf-idf",
        type=str,
        choices=["tf-idf", "word2vec", "fasttext"],
        required=True,
        help="Vectorizer model to use",
    )

    trainingMode.add_argument(
        "-o",
        "--output",
        default=Path("../model"),
        type=Path,
        required=False,
        help="Directory to save the classifiers to",
        dest="trainingOutput",
    )

    return parser.parse_args()
