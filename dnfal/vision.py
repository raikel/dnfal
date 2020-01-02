from cvtlib.video import VideoCapture

from .alignment import FaceAligner, FaceMarker
from .detection import FaceDetector
from .encoding import FaceEncoder
from .engine import FrameAnalyzer, VideoAnalyzer, FaceMatcher
from .loggers import logger, config_logger
from .settings import Settings


class FacesVision:
    def __init__(self, se: Settings):

        self.settings: Settings = se

        self.init_logger(se)

        logger.info('Starting building FacesVision.')

        self.face_detector = FaceDetector(
            weights_path=se.detector_weights_path,
            min_height=se.detection_min_height,
            min_score=se.detection_min_score,
            nms_thresh=se.detection_nms_thresh,
            force_cpu=se.force_cpu
        )
        logger.info('Building FacesVision: Face detector created.')

        # noinspection PyTypeChecker
        self.face_marker: FaceMarker = None
        # noinspection PyTypeChecker
        self.face_aligner: FaceAligner = None
        # noinspection PyTypeChecker
        self.face_encoder: FaceEncoder = None

        if not se.detection_only or se.video_mode == VideoAnalyzer.MODE_HUNT:
            self.face_marker: FaceMarker = FaceMarker(
                weights_path=se.marker_weights_path,
                force_cpu=se.force_cpu
            )
            logger.info('Building FacesVision: Face aligner created.')

            self.face_aligner = FaceAligner()
            logger.info('Building FacesVision: Face marker created.')

            self.face_encoder = FaceEncoder(
                weights_path=se.encoder_weights_path,
                force_cpu=se.force_cpu
            )
            logger.info('Building FacesVision: Face encoder created.')

        self.frame_analyzer = FrameAnalyzer(
            detector=self.face_detector,
            marker=self.face_marker,
            aligner=self.face_aligner,
            encoder=self.face_encoder,
            detection_only=se.detection_only,
            max_frame_size=se.max_frame_size,
            store_frames=se.store_face_frames,
            max_deviation=se.align_max_deviation,
            marking_min_score=se.marking_min_score,
            face_padding=se.detection_face_padding
        )

        logger.info('Building FacesVision: Frame analyzer created.')

        self.face_matcher = FaceMatcher()

        logger.info('Building FacesVision: Face matcher created.')

        # noinspection PyTypeChecker
        self.video_capture: VideoCapture = None
        # noinspection PyTypeChecker
        self.video_analyzer: VideoAnalyzer = None

        self.init_video_analyzer(se)

    @staticmethod
    def init_logger(se: Settings):
        if se.logging_file or se.log_to_console:
            config_logger(
                level=se.logging_level,
                file_path=se.logging_file,
                to_console=se.log_to_console
            )

    def init_video_analyzer(self, se: Settings):
        if se.video_capture_source is not None:
            self.video_capture = VideoCapture(src=se.video_capture_source)

            self.video_analyzer = VideoAnalyzer(
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

            logger.info('Building FacesVision: Video analyzer created.')

    def update_settings(self, se: Settings):

        self.settings = se

        self.init_logger(se)

        self.face_detector.min_size = se.detection_min_height
        self.face_detector.score_thresholds = se.detection_min_score
        self.face_detector.nms_thresholds = se.detection_nms_thresh
        self.face_detector.face_padding = se.detection_face_padding

        self.face_aligner.max_nose_deviation = se.align_max_deviation

        self.frame_analyzer.max_frame_size = se.max_frame_size
        self.frame_analyzer.store_frames = se.store_face_frames

        self.init_video_analyzer(se)