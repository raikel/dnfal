import json
from typing import Tuple, List
from validator_collection import validators
from .engine import VideoAnalyzer
from . import loggers


class InvalidSetting(Exception):
    pass


class InvalidJsonSettings(Exception):
    pass


class InvalidYamlSettings(Exception):
    pass


VIDEO_MODES = (VideoAnalyzer.MODE_ALL, VideoAnalyzer.MODE_HUNT)

LOGGING_LEVELS = (
    loggers.LEVEL_DEBUG,
    loggers.LEVEL_INFO,
    loggers.LEVEL_WARNING,
    loggers.LEVEL_ERROR,
    loggers.LEVEL_CRITICAL
)


class Settings:
    """Settings for FacesVision instance.

    See also
    --------
    vision.FacesVision
    """

    # noinspection PyTypeChecker
    def __init__(self):
        self._force_cpu: bool = False
        self._detector_weights_path: str = ''
        self._marker_weights_path: str = ''
        self._encoder_weights_path: str = ''
        self._detection_min_height: int = 64
        self._detection_min_score: float = 0.9
        self._marking_min_score: float = 0.95
        self._detection_nms_thresh: float = 0.6
        self._detection_face_padding: float = 0.2
        self._detection_only: bool = False
        self._align_max_deviation: Tuple[float, float] = None
        self._max_frame_size: int = 0
        self._similarity_thresh: float = 0.5
        self._video_capture_source: [str, int, None] = None
        self._video_real_time: bool = True
        self._video_capture_resize_to: int = 0
        self._video_detect_interval: float = 1
        self._video_mode: str = 'all'
        self._video_hunt_embeddings: list = None
        self._video_hunt_keys: List[int] = None
        self._video_start_at: float = 0
        self._video_stop_at: float = -1
        self._video_roi_adapt: bool = False
        self._store_face_frames: bool = False
        self._faces_time_memory: float = 30
        self._logging_level: str = 'info'
        self._logging_file: str = ''
        self._log_to_console: bool = False

        self._fields = [attr[1::] for attr in self.__dict__]

    @property
    def force_cpu(self):
        """Use CPU instead GPU even if GPU is available

        Returns
        -------
        force_cpu : bool, (default=False)
        """
        return self._force_cpu

    @force_cpu.setter
    def force_cpu(self, val: bool):
        self._force_cpu = bool(validators.numeric(val))

    @property
    def detector_weights_path(self):
        """Path to file containing the model weights of face detector

        Returns
        -------
        detector_weights_path : str
        """
        return self._detector_weights_path

    @detector_weights_path.setter
    def detector_weights_path(self, val: str):
        self._detector_weights_path = validators.string(val)

    @property
    def marker_weights_path(self):
        """Path to file containing the model weights of face marker

        Returns
        -------
        marker_weights_path : str
        """
        return self._marker_weights_path

    @marker_weights_path.setter
    def marker_weights_path(self, val: str):
        self._marker_weights_path = validators.string(val)

    @property
    def encoder_weights_path(self):
        """Path to file containing the model weights of face encoder

        Returns
        -------
        encoder_weights_path : str
        """
        return self._encoder_weights_path

    @encoder_weights_path.setter
    def encoder_weights_path(self, val: str):
        self._encoder_weights_path = validators.string(val)

    @property
    def detection_min_height(self):
        """Minimum height in pixels of face to be detected

        Setting the detection minimum height to a reasonable value can help to
        save considerable computing power. Sometimes it is not necessary to
        detect very small faces because it will not be possible to recognize
        them.

        Returns
        -------
        detection_min_height : positive int (default=64)
        """
        return self._detection_min_height

    @detection_min_height.setter
    def detection_min_height(self, val: (float, int)):
        self._detection_min_height = int(validators.numeric(val, minimum=1))

    @property
    def detection_min_score(self):
        """Minimum confidence score of detected faces

        Faces with scores lower than the `detection_min_score` will be
        discarded.

        Returns
        -------
        detection_min_score : float in range [0, 1], (default=0.9)
        """
        return self._detection_min_score

    @detection_min_score.setter
    def detection_min_score(self, val: float):
        self._detection_min_score = validators.float(val, minimum=0, maximum=1)

    @property
    def marking_min_score(self):
        """Minimum marking score of a face after landmark detection

        Face analysis includes three steps: detection, alignment and
        encoding. At the alignment step, face landmarks are computed and a
        marking score between 0 and 1 is assigned to each face. Lower scores
        mean lower face alignment confidence. Faces with marking score lower
        than the `marking_min_score` will be discarded.

        Returns
        -------
        marking_min_score : float in range [0, 1], (default=0.95)
        """
        return self._marking_min_score

    @marking_min_score.setter
    def marking_min_score(self, val: float):
        self._marking_min_score = validators.float(val, minimum=0, maximum=1)

    @property
    def detection_nms_thresh(self):
        """Non-maximum suppression threshold of detected faces

        At detection step, from the set of face boxes which overlaps more than
        the corresponding NMS threshold the one with the highest score will be
        selected and the others discarded.

        Returns
        -------
        detection_nms_thresh : float in range [0, 1], (default=0.7)
        """
        return self._detection_nms_thresh

    @detection_nms_thresh.setter
    def detection_nms_thresh(self, val: float):
        self._detection_nms_thresh = validators.float(val, minimum=0, maximum=1)

    @property
    def detection_face_padding(self):
        """Relative padding of detected faces

        After a face is detected, the corresponding face image is extracted
        from the original image and sent to a face alignment stage. Padding the
        detected face box can help to obtain better alignment results. Padding
        value is a number between 0 and 1, relative to the face box height.

        Returns
        -------
        detection_face_padding : float in range [0, 1], (default=0.2)
        """
        return self._detection_face_padding

    @detection_face_padding.setter
    def detection_face_padding(self, val: float):
        self._detection_face_padding = validators.float(val, minimum=0, maximum=1)

    @property
    def detection_only(self):
        """Only detect faces, do no perform alignment or encoding

        Face analysis includes three steps: detection, alignment and
        encoding. If `detection_only = True`, only the detection step is done.

        Returns
        -------
        detection_only : bool, (default=False)
        """
        return self._detection_only

    @detection_only.setter
    def detection_only(self, val: bool):
        self._detection_only = bool(validators.numeric(val))

    @property
    def align_max_deviation(self):
        """Maximum deviation of the nose in a detected face

        Face analysis includes three steps: detection, alignment and
        encoding. At the alignment step, the nose deviation is computed. The
        nose deviation is a tuple where the first and second elements
        represents the horizontal and vertical deviation of nose landmark,
        respectively. If this deviation is higher than `align_max_deviation`
        (element-wise comparision), the face is discarded. For a deviation
        lower than (0.4, 0.3), the face is approximately frontal in both
        planes. Set it to `None` to allow any deviation.

        Returns
        -------
        align_max_deviation : tuple of length=2 or None, (default=None)
        """
        return self._align_max_deviation

    @align_max_deviation.setter
    def align_max_deviation(self, val: Tuple[float, float]):
        val = validators.iterable(
            val, minimum_length=2, maximum_length=2, allow_empty=True
        )
        self._align_max_deviation = val if val is None else tuple([
            validators.numeric(i, minimum=0) for i in val
        ])

    @property
    def max_frame_size(self):
        """Maximum frame size

        During video analysis, when a face is detected a Face object is created.
        Optionally, the frame where the face was detected can be stored in the
        Face object. If a frame is stored, it is resized so that is maximum
        dimension is lowest than `max_frame_size`. Set it to a value lower than
        or equal to zero to do not resize.

        Returns
        -------
        max_frame_size : int, (default=0)
        """
        return self._max_frame_size

    @max_frame_size.setter
    def max_frame_size(self, val: (float, int)):
        self._max_frame_size = int(validators.numeric(val))

    @property
    def similarity_thresh(self):
        """Similarity threshold for face matching

        During face matching, two faces are considered to belong to the same
        person if the similarity between them is higher than
        `similarity_thresh`. A similarity of `1` means that the two faces are
        exactly the same, while a similarity of `0` means that the two faces
        are very different. Similarity between two faces if computed on the
        basis of the distance between their embeddings vectors.

        Returns
        -------
        similarity_thresh : float in range [0, 1], (default=0.5)
        """
        return self._similarity_thresh

    @similarity_thresh.setter
    def similarity_thresh(self, val: (float, int)):
        self._similarity_thresh = validators.numeric(val, minimum=0, maximum=1)

    ############################################################################
    # VIDEO ANALYSIS SETTINGS
    ############################################################################

    @property
    def video_capture_source(self):
        """Video analysis source

        For a string value, it must be a valid path to a video file of any
        supported format by `cv2.VideoCapture` class of opencv or a valid url
        video stream. For an int value, it refers to a connected camera,
        enumerated starting from zero. Set it to `None` to do initialize video
        processing engine.

        Returns
        -------
        video_capture_source : int, str or None (default=None)
        """
        return self._video_capture_source

    @video_capture_source.setter
    def video_capture_source(self, val: (str, int, None)):
        try:
            self._video_capture_source = validators.integer(
                val, coerce_value=True, allow_empty=True
            )
        except TypeError:
            self._video_capture_source = validators.string(
                val, allow_empty=True
            )

    @property
    def video_real_time(self):
        """Whether the video capture is from a camera or not

        Video analysis can be done from a real time source, like a camera, or
        from a non-real time source, like a video file. Although the video
        analysis for both types of sources is similar, some differences exists.
        For example, for a real time source when the algorithm finish analyzing
        one frame, it waits (sleeps) a time equal to `video_detect_interval`
        before processing a new frame. For a video file, the algorithms jumps
        directly into the next target frame after analyzing the current frame.

        Returns
        -------
        video_real_time : bool, (default=True)
        """
        return self._video_real_time

    @video_real_time.setter
    def video_real_time(self, val: bool):
        self._video_real_time = bool(validators.numeric(val))

    @property
    def video_capture_resize_to(self):
        """Size in pixels to which video frames are resized before processing

        After capturing the video frames, they are scaled so that its maximum
        dimension are equal to `video_capture_resize_to`. Set it to a value
        lower than or equal to zero to do not perform any resizing.

        Returns
        -------
        video_capture_resize_to : int, (default=0)
        """
        return self._video_capture_resize_to

    @video_capture_resize_to.setter
    def video_capture_resize_to(self, val: int):
        self._video_capture_resize_to = int(validators.numeric(val))

    @property
    def video_detect_interval(self):
        """Interval in seconds to analyze frames in video processing

        In order to save processing time, not all frames are analyzed during
        video processing. For a real time source, when the algorithm finish
        analyzing one frame, it waits (sleeps) a time equal to
        `video_detect_interval` before processing a new frame. Similarly, for a
        video file, the algorithms jumps directly into the next target frame
        located `video_detect_interval` seconds beyond. Setting it to a
        reasonable value can help to save considerable processing time, taking
        into account the (typical) low movement speed of people.

        Returns
        -------
        video_detect_interval : positive float, (default=1)
        """
        return self._video_detect_interval

    @video_detect_interval.setter
    def video_detect_interval(self, val: (int, float)):
        self._video_detect_interval = validators.numeric(val, minimum=0)

    @property
    def video_face_memory(self):
        """Time memory in seconds of faces during video analysis

        During video analysis, the algorithm try to match detected faces and
        group matched faces into Subject objects. A face is added to a Subject
        object only if no face have been added to that subject in the previous
        `face_memory` seconds. This setting helps to control the growth without
        limits of the number of faces added to a subject. Also, as face images
        belonging to the same person in successive nearby frames may be very
        similar, it also avoid storing redundant information.

        Returns
        -------
        video_face_memory : positive float, (default=30)
        """
        return self._faces_time_memory

    @video_face_memory.setter
    def video_face_memory(self, val: (float, int)):
        self._faces_time_memory = validators.numeric(val, minimum=0)

    @property
    def video_mode(self):
        """Video analysis mode

        Video analysis can be executed in two different modes: "all" or "hunt".
        In the "all" mode, the algorithm tries to register all the detected
        faces, grouping them in Subject objects. In the "hunt" mode the
        algorithm tries to match the detected faces with a predefined list of
        face embeddings and to register only matched faces.

        Note that if `video_mode` is set to "hunt", parameter `detection_only`
        will be ignored if set to `True`, because face encoding is needed to
        match against hunted faces.

        Returns
        -------
        video_mode : string, (default="all")
        """
        return self._video_mode

    @video_mode.setter
    def video_mode(self, val: str):
        val = validators.string(val)
        if val not in VIDEO_MODES:
            raise ValueError(f'"{val}" is not one of {VIDEO_MODES}.')
        self._video_mode = val

    @property
    def video_hunt_embeddings(self):
        """Embeddings vectors to hunt for

        If `video_mode = "hunt"` the algorithm tries to match the detected
        faces against a set of embeddings vectors in `hunt_embeddings` that
        represents the hunted faces, and to register only matched faces. Each
        embedding vector is of length 512, as returned by encoding.FaceEncoder.

        Returns
        -------
        video_hunt_embeddings : list or None, (default=None)
        """
        return self._video_hunt_embeddings

    @video_hunt_embeddings.setter
    def video_hunt_embeddings(self, val: List[float]):
        val = validators.iterable(val, allow_empty=True)
        self._video_hunt_embeddings = val if val is None else [
            validators.numeric(i) for i in val
        ]

    @property
    def video_hunt_keys(self):
        """List of face keys associated to the hunt embeddings

        If `video_mode = "hunt"` the algorithm tries to match the detected
        faces against a set of embeddings vectors in `hunt_embeddings` that
        represents the hunted faces, and to register only matched faces. Each
        embedding vector of a hunted face must be associated to face key in
        `video_hunt_keys`, which is a list of integers numbers of length
        `n_hunts`.

        Returns
        -------
        video_hunt_keys : list of int or None, (default=None)
        """
        return self._video_hunt_keys

    @video_hunt_keys.setter
    def video_hunt_keys(self, val: List[int]):
        val = validators.iterable(val, allow_empty=True)
        self._video_hunt_keys = val if val is None else [
            validators.numeric(i) for i in val
        ]

    @property
    def video_start_at(self):
        """Starting time in seconds of processing for video source files

        If `video_real_time = False`, the video file will be analyzed starting
        from `video_start_at` seconds. Note that if the duration of the video
        file is lower than `video_start_at` or `video_start_at` is higher than
        `video_stop_at`, `video_start_at` will be ignored.

        Returns
        -------
        video_start_at : float (default=0)
        """
        return self._video_start_at

    @video_start_at.setter
    def video_start_at(self, val: (float, int)):
        self._video_start_at = validators.numeric(val)

    @property
    def video_stop_at(self):
        """Stopping time in seconds of processing for video source files

        If `video_real_time = False`, the video file will analysis will be
        terminated at `video_stop_at` seconds. Note that if the duration of the
        video file is lower than `video_stop_at` or `video_start_at` is higher
        than `video_stop_at`, `video_stop_at` will be ignored. Set it to a
        negative value to run the analysis until the end of the video file.

        Returns
        -------
        video_stop_at : float (default=-1)
        """
        return self._video_stop_at

    @video_stop_at.setter
    def video_stop_at(self, val: (float, int)):
        self._video_stop_at = validators.numeric(val)

    @property
    def video_roi_adapt(self):
        """Use adaptive ROI during video analysis.

        If set to `True`, adaptive Region-Of-Interest (ROI) will be used during
        video analysis. This can help to save considerable processing time if
        the video comes from a static camera, as in these case the faces in the
        video frames are likely to be located inside a rectangle that changes
        little (in size and  location) over time. So, instead analyzing the
        entire frame, only a sub-region of the frame is analyzed. This
        rectangle is estimated at run time an used once it converges.

        Returns
        -------
        video_roi_adapt : bool (default=False)

        See also
        --------
        engine.AdaptiveRoi
        """
        return self._video_roi_adapt

    @video_roi_adapt.setter
    def video_roi_adapt(self, val: bool):
        self._video_roi_adapt = bool(validators.numeric(val))

    @property
    def store_face_frames(self):
        """Whether to store frames in faces or not

        During video analysis, when a face is detected a Face object is created.
        Optionally, the frame where the face was detected can be stored in the
        Face object if `store_face_frames` is set to `True`.

        Returns
        -------
        store_face_frames : bool (default=False)
        """
        return self._store_face_frames

    @store_face_frames.setter
    def store_face_frames(self, val: bool):
        self._store_face_frames = bool(validators.numeric(val))

    ############################################################################
    # LOGGING SETTINGS
    ############################################################################

    @property
    def logging_level(self):
        """Logging level

        Set the global logging level. Supported levels are "debug", "info",
        "warning", "error" or "critical".

        Returns
        -------
        logging_level : str (default="info")

        See also
        --------
        logging
        """
        return self._logging_level

    @logging_level.setter
    def logging_level(self, val: str):
        val = validators.string(val).lower()
        if val not in LOGGING_LEVELS:
            raise ValueError(f'"{val}" is not one of {LOGGING_LEVELS}.')
        self._logging_level = val

    @property
    def logging_file(self):
        """Path to logging file

        Path to a file to write logs. Set it to empty string to do not output
        logs to a file.

        Returns
        -------
        logging_file : str (default="")

        See also
        --------
        logging
        """
        return self._logging_file

    @logging_file.setter
    def logging_file(self, val: str):
        self._logging_file = validators.string(val)

    @property
    def log_to_console(self):
        """Whether to write logs to console or not

        If set to `True` all logs are written to the standard output.

        Returns
        -------
        logging_level : bool (default=False)

        See also
        --------
        logging
        """
        return self._log_to_console

    @log_to_console.setter
    def log_to_console(self, val: bool):
        self._log_to_console = bool(validators.numeric(val))

    ############################################################################
    # METHODS
    ############################################################################

    def serialize(self):
        data = {}
        for field in self._fields:
            data[field] = getattr(self, field)
        return data

    def dict_load(self, data: dict):
        data_fields = data.keys()
        for field in self._fields:
            if field in data_fields:
                setattr(self, field, data[field])

    def json_load(self, json_text):
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as err:
            raise InvalidJsonSettings(f'Can not decode json settings. {str(err)}.')

        if not isinstance(data, dict):
            raise InvalidJsonSettings(f'Decoded json settings is not a dictionary object.')

        self.dict_load(data)
