from argparse import ArgumentParser, Namespace

from . import AlphabeticalOrderHelpFormatter, authors


def getArgs() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="LUC COMP 429 (NLP) Final: Victorian Author Document Classifier",
        usage="A program to train and test models meant to classify Victorian era documents.",
        description=None,
        epilog=f"Authors: {','.join(authors)}",
        formatter_class=AlphabeticalOrderHelpFormatter,
    )
    return parser.parse_args()
