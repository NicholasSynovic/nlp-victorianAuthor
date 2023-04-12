from argparse import ArgumentParser, Namespace
from pathlib import Path

from . import AlphabeticalOrderHelpFormatter, authors


def getArgs() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="LUC COMP 429 (NLP) Final: Victorian Author Document Classifier",
        usage="A program to train and test models meant to classify Victorian era documents.",
        description=None,
        epilog=f"Authors: {', '.join(authors)}",
        formatter_class=AlphabeticalOrderHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        choices=["train", "test"],
        required=False,
        help="Either train models or test models",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=Path("model"),
        type=Path,
        required=False,
        help="Model folder to store/ read models from",
    )

    parser.add_argument(
        "--training-dataset",
        type=Path,
        required=True,
        help="Training dataset to use",
    )

    # parser.add_argument(
    #     "--testing-dataset",
    #     type=Path,
    #     required=False,
    #     help="Training dataset to use",
    # )

    return parser.parse_args()
