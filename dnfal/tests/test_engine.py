import unittest
from dnfal.engine import AdaptiveRoi


class TestAdaptiveRoi(unittest.TestCase):

    def setUp(self) -> None:
        self.adaptive_roi = AdaptiveRoi()

    def test_add_box(self):
        adaptive_roi = AdaptiveRoi(thresh=0)
        box = (0, 0, 1, 1)
        self.adaptive_roi.add_box(box)
        self.assertEqual(None, adaptive_roi.roi)


if __name__ == '__main__':
    unittest.main()
