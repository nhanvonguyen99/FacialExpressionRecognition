import numpy as np


class facs2emotion:
    def __init__(self, facsUp, facsLow):
        self.facsUp = facsUp
        self.facsLow = facsLow

    def declare(self):
        if 4 in self.facsUp and 2 in self.facsLow:
            return 1

        if 0 in self.facsUp and 2 in self.facsUp and 3 in self.facsLow:
            return 2

        if 0 in self.facsUp and 1 in self.facsUp and 3 in self.facsUp and 4 in self.facsLow and 9 in self.facsLow:
            return 3

        if 0 in self.facsUp and 1 in self.facsUp and 2 in self.facsUp \
                and 3 in self.facsUp and 5 in self.facsUp and 3 in self.facsLow:
            return 4

        if 2 in self.facsUp and 3 in self.facsUp and 5 in self.facsLow and 5 in self.facsUp:
            return 5

        if 3 in self.facsLow and 4 in self.facsLow:
            return 6

        return 0
