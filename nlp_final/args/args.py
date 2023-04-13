from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path

from . import AlphabeticalOrderHelpFormatter, authors


def getArgs() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="LUC COMP 429 (NLP) Final: Victorian Author Document Classifier",
        usage=None,
        description="A program to train and test models meant to classify Victorian era documents.",
        epilog=f"Authors: {', '.join(authors)}",
        formatter_class=AlphabeticalOrderHelpFormatter,
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
