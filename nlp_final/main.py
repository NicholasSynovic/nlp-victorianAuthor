from argparse import Namespace
from os import mkdir

from nlp_final.args import args
from nlp_final.train import train


def main() -> None:
    userArgs: Namespace = args.getArgs()

    match userArgs.mode:
        case "train":
            try:
                mkdir(path=userArgs.output)
            except FileExistsError:
                pass

            train.main(args=userArgs)
        case "test":
            pass


if __name__ == "__main__":
    main()
