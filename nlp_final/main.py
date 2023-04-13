from argparse import Namespace
from os import mkdir

from sklearnex import patch_sklearn

from nlp_final.args import args
from nlp_final.train import train


def main() -> None:
    userArgs: Namespace = args.getArgs()

    patch_sklearn()

    match userArgs.mode:
        case "train":
            try:
                mkdir(path=userArgs.output)
            except FileExistsError:
                pass

            train.main(args=userArgs)
        case "infrence":
            pass


if __name__ == "__main__":
    main()
