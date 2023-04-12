from argparse import ArgumentDefaultsHelpFormatter
from typing import List

authors: List[str] = [
    "Giorgio Montenegro <gmontenegro@luc.edu>",
    "Nicholas M. Synovic <nicholas.synovic@gmail.com>",
    "Nelson Stefan <snelson9@luc.edu>",
]


class AlphabeticalOrderHelpFormatter(ArgumentDefaultsHelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.dest)
        super(AlphabeticalOrderHelpFormatter, self).add_arguments(actions)
