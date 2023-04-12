from argparse import Namespace

from args.args import getArgs


def main() -> None:
    args: Namespace = getArgs()

    match args.mode:
        case "train":
            pass
        case "test":
            pass


if __name__ == "__main__":
    main()
