import unittest
from refactoring.MOO.MOFEA import MOFEA


class TestMOFEA(unittest.TestCase):
    def setUp(self) -> None:
        dim = 20
        n_obj = 3