from argparse import Namespace
from os import mkdir

from sklearnex import patch_sklearn

from nlp_final.args import args
from nlp_final.train import train


def makeDir(args: Namespace) -> None:
    try:
        try:
            mkdir(path=args.vectorizerOutput)
        except FileExistsError:
            pass
    except AttributeError:
        try:
            mkdir(path=args.classifierOutput)
        except FileExistsError:
            pass


def main() -> None:
    userArgs: Namespace = args.getArgs()

    patch_sklearn()

    match userArgs.mode:
        case "train":
            makeDir(args=userArgs)
            train.main(args=userArgs)
        case "infrence":
            pass


if __name__ == "__main__":
    main()
