from cvtlib.video import VideoCapture

from .alignment import FaceAligner, FaceMarker
from .detection import FaceDetector
from .encoding import FaceEncoder
from dnfal.genderage import GenderAgePredictor
from .engine import FrameAnalyzer, VideoAnalyzer, FaceMatcher
from .loggers import logger, config_logger
from .settings import Settings


class FacesVision:
    def __init__(self, se: Settings):

        self.settings: Settings = se

        self.init_logger(se)

        # noinspection PyTypeChecker
        self._face_detector: FaceDetector = None
        # noinspection PyTypeChecker
        self._face_marker: FaceMarker = None
        # noinspection PyTypeChecker
        self._face_aligner: FaceAligner = None
        # noinspection PyTypeChecker
        self._face_encoder: FaceEncoder = None
        # noinspection PyTypeChecker
        self._face_matcher: FaceMatcher = None
        # noinspection PyTypeChecker
        self._genderage_predictor: GenderAgePredictor = None
        # noinspection PyTypeChecker
        self._frame_analyzer: FrameAnalyzer = None
        # noinspection PyTypeChecker
        self._video_capture: VideoCapture = None
        # noinspection PyTypeChecker
        self._video_analyzer: VideoAnalyzer = None

    @property
    def face_marker(self):
        if self._face_marker is None:
            self._face_marker = FaceMarker(
                weights_path=self.settings.marker_weights_path,
                force_cpu=self.settings.force_cpu
            )
            logger.info('Face marker created.')
        return self._face_marker

    @property
    def face_aligner(self):
        if self._face_aligner is None:
            se = self.settings
            self._face_aligner = FaceAligner(out_size=se.face_align_size)
            logger.info('Face aligner created.')
        return self._face_aligner

    @property
    def face_encoder(self):
        if self._face_encoder is None:
            self._face_encoder = FaceEncoder(
                weights_path=self.settings.encoder_weights_path,
                force_cpu=self.settings.force_cpu
            )
            logger.info('Face encoder created.')
        return self._face_encoder

    @property
    def face_detector(self):
        if self._face_detector is None:
            se = self.settings
            self._face_detector = FaceDetector(
                weights_path=se.detector_weights_path,
                min_height=se.detection_min_height,
                min_score=se.detection_min_score,
                nms_thresh=se.detection_nms_thresh,
                force_cpu=se.force_cpu
            )
            logger.info('Face detector created.')
        return self._face_detector

    @property
    def face_matcher(self):
        if self._face_matcher is None:
            self._face_matcher = FaceMatcher()
            logger.info('Face matcher created.')
        return self._face_matcher

    @property
    def genderage_predictor(self):
        if self._genderage_predictor is None:
            se = self.settings
            self._genderage_predictor = GenderAgePredictor(
                se.genderage_weights_path,
                force_cpu=se.force_cpu
            )
            logger.info('Gender-age predictor created.')
        return self._genderage_predictor

    @property
    def frame_analyzer(self):
        if self._frame_analyzer is None:
            se = self.settings

            face_marker = None
            face_aligner = None
            face_encoder = None

            if not se.detection_only or se.video_mode == VideoAnalyzer.MODE_HUNT:
                face_marker = self.face_marker
                face_aligner = self.face_aligner
                face_encoder = self.face_encoder

            self._frame_analyzer = FrameAnalyzer(
                detector=self.face_detector,
                marker=face_marker,
                aligner=face_aligner,
                encoder=face_encoder,
                detection_only=se.detection_only,
                max_frame_size=se.max_frame_size,
                store_frames=se.store_face_frames,
                max_deviation=se.align_max_deviation,
                marking_min_score=se.marking_min_score,
                face_padding=se.detection_face_padding
            )
            logger.info('Frame analyzer created.')

        return self._frame_analyzer

    @property
    def video_capture(self):
        if self._video_capture is None:
            se = self.settings
            self._video_capture = VideoCapture(
                src=se.video_capture_source,
                auto_grab=se.video_real_time
            )
            logger.info('Video capture created.')
        return self._video_capture

    @property
    def video_analyzer(self):
        if self._video_analyzer is None:
            se = self.settings
            self._video_analyzer = VideoAnalyzer(
                frame_analyzer=self.frame_analyzer,
                video_capture=self.video_capture,
                real_time=se.video_real_time,
                detect_interval=se.video_detect_interval,
                faces_memory=se.video_face_memory,
                similarity_thresh=se.similarity_thresh,
                mode=se.video_mode,
                hunt_embeddings=se.video_hunt_embeddings,
                hunt_keys=se.video_hunt_keys,
                start_at=se.video_start_at,
                stop_at=se.video_stop_at,
                roi_adapt=se.video_roi_adapt
            )
            logger.info('Video analyzer created.')

        return self._video_analyzer

    def update_settings(self, se: Settings):

        self.settings = se
        self.init_logger(se)

        # noinspection PyTypeChecker
        self._face_detector: FaceDetector = None
        # noinspection PyTypeChecker
        self._face_marker: FaceMarker = None
        # noinspection PyTypeChecker
        self._face_aligner: FaceAligner = None
        # noinspection PyTypeChecker
        self._face_encoder: FaceEncoder = None
        # noinspection PyTypeChecker
        self._face_matcher: FaceMatcher = None
        # noinspection PyTypeChecker
        self._genderage_predictor: GenderAgePredictor = None
        # noinspection PyTypeChecker
        self._frame_analyzer: FrameAnalyzer = None
        # noinspection PyTypeChecker
        self._video_capture: VideoCapture = None
        # noinspection PyTypeChecker
        self._video_analyzer: VideoAnalyzer = None

    @staticmethod
    def init_logger(se: Settings):
        if se.logging_file or se.log_to_console:
            config_logger(
                level=se.logging_level,
                file_path=se.logging_file,
                to_console=se.log_to_console
            )
