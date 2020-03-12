import unittest
from os import path
import cv2 as cv

from dnfal.engine import AdaptiveRoi
from dnfal.engine import (
    FaceDetector,
    FaceMarker,
    FaceAligner,
    FaceEncoder,
    FrameAnalyzer
)
from cvtlib.files import list_files

DETECTOR_WEIGHTS_PATH = 'models/weights_face_detector.pth'
MARKER_WEIGHTS_PATH = 'models/weights_face_marker.npy'
ENCODER_WEIGHTS_PATH = 'models/weights_face_encoder.pth'

CURR_DIR = path.dirname(path.abspath(__file__))
DATA_DIR = path.join(CURR_DIR, 'data')
IMAGE_EXT = ('.jpeg', '.jpg', '.png')


class TestAdaptiveRoi(unittest.TestCase):

    def setUp(self) -> None:
        self.adaptive_roi = AdaptiveRoi()

    def test_add_box(self):
        adaptive_roi = AdaptiveRoi(thresh=0)
        box = (0, 0, 1, 1)
        self.adaptive_roi.add_box(box)
        self.assertEqual(None, adaptive_roi.roi)
        
        
class TestFrameAnalyzer(unittest.TestCase):

    def setUp(self) -> None:

        detection_min_height = 24
        detection_min_score = 0.8
        marking_min_score = 0.6
        max_frame_size = 1024
        store_frames = True
        align_max_deviation = None
        detection_face_padding = 0.2

        face_detector = FaceDetector(
            weights_path=DETECTOR_WEIGHTS_PATH,
            min_height=detection_min_height,
            min_score=detection_min_score,
            force_cpu=True
        )

        face_marker: FaceMarker = FaceMarker(
            weights_path=MARKER_WEIGHTS_PATH,
            force_cpu=True
        )

        face_aligner = FaceAligner()

        face_encoder = FaceEncoder(
            weights_path=ENCODER_WEIGHTS_PATH,
            force_cpu=True
        )

        self.frame_analyzer = FrameAnalyzer(
            detector=face_detector,
            marker=face_marker,
            aligner=face_aligner,
            encoder=face_encoder,
            detection_only=False,
            max_frame_size=max_frame_size,
            store_frames=store_frames,
            max_deviation=align_max_deviation,
            marking_min_score=marking_min_score,
            face_padding=detection_face_padding
        )

        self.image_paths = list_files(DATA_DIR, IMAGE_EXT, recursive=True)

    def test_find_faces(self):

        for image_path in self.image_paths:
            image = cv.imread(image_path)
            faces, _ = self.frame_analyzer.find_faces(image)
            self.assertGreater(len(faces), 0)


if __name__ == '__main__':
    unittest.main()
