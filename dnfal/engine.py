from time import time, sleep
from typing import Dict
from typing import List, Tuple

import cv2 as cv
import cvtlib
import numpy as np
import torch
from cvtlib.video import VideoCapture

from .mtypes import Face, Frame, Subject
from .alignment import FaceAligner, FaceMarker
from .detection import FaceDetector
from .encoding import FaceEncoder
from .loggers import logger

EMBEDDINGS_LENGTH = 512


def distance_to_similarity(dists: (np.ndarray, float)):
    a = 5
    b = 1.3
    return 1 / (1 + np.exp(a * (dists - b)))


def similarity_to_distance(sim: (np.ndarray, float)):
    a = 5
    b = 1.3
    sim = np.clip(sim, a_max=(1 - 1e-4), a_min=1e-4)
    return np.clip((1/a) * np.log(1/sim - 1) + b, a_min=0, a_max=None)


class AdaptiveRoi:
    """Adaptive Region-of-Interest (ROI) generator.

    Iteratively computes a ROI by finding a rectangle that encloses a set of
    other rectangles successively added at each iteration. Initially the ROI
    is empty. ROI is updated by adding a new rectangles. When a new
    rectangle is added, the ROI is updated to a new rectangle that contains
    both, the actual ROI rectangle an the new rectangle added. At each step
    the moving average deviation of the ROI is computed. The smoothness of
    this average is controlled via a smoothing factor `alpha`. The ROI is
    considered valid only when the moving average deviation is below a
    threshold.

    An adaptive ROI can help to save considerable processing time in video
    object detection tasks, when the video comes from a static camera. In
    these cases the location of the detected objects in the video frames are
    likely to be located inside a rectangle that changes little (in size and
    location) over time. So, instead analyzing the entire frame, only a
    sub-region of the frame is analyzed.

    Parameters
    ----------
    thresh : float, optional, (default=1)
        Convergence threshold for the moving average of the ROI deviation.

    alpha : float, optional, (default=0.1)
        Smoothing factor of the exponential moving average of the ROI
        deviation.
    """
    def __init__(
        self,
        thresh: float = 1,
        alpha: float = 0.1
    ):
        self.alpha: float = alpha
        self.thresh: float = thresh

        self._cur_roi: [tuple, None] = None
        self._avg_roi: [tuple, None] = None
        self._roi_dev: float = float('inf')

    @property
    def roi(self):
        """Current ROI.

        The current ROI is a tuple of four float numbers representing a
        rectangle that encloses all the previously added rectangles, if the
        exponential moving average of its deviation have converged. Else it
        is `None`. The first two values in the tuple rectangle represents
        the top-left coordinates of the rectangle, while the latest two
        values represents the bottom-left coordinates.

        Returns
        -------
        curr_roi : tuple of length=4 or None
        """
        return self._cur_roi if self._roi_dev <= self.thresh else None

    def add_box(self, box: tuple):
        """Adds a new rectangle.

        Parameters
        -------
        box : tuple of length=4
            The first two values in the tuple rectangle represents the top-left
            coordinates of the rectangle, while the latest two values
            represents the bottom-left coordinates.
        """
        cur_roi = self._cur_roi
        avg_roi = self._avg_roi
        alpha = self.alpha

        if cur_roi is None:
            cur_roi = tuple(box)
        else:
            cur_roi = (
                min(cur_roi[0], box[0]),
                min(cur_roi[1], box[1]),
                max(cur_roi[2], box[2]),
                max(cur_roi[3], box[3]),
            )

        self._cur_roi = cur_roi

        if avg_roi is None:
            self._avg_roi = tuple(cur_roi)
            self._roi_dev = sum(cur_roi) / 4
        else:
            self._avg_roi = tuple([
                alpha * cur_roi[i] + (1 - alpha) * avg_roi[i]
                for i in range(4)
            ])

            roi_dev = sum((
                max(avg_roi[0] - cur_roi[0], 0),
                max(avg_roi[1] - cur_roi[1], 0),
                max(cur_roi[2] - avg_roi[2], 0),
                max(cur_roi[3] - avg_roi[3], 0)
            )) / 4

            self._roi_dev = alpha * roi_dev + (1 - alpha) * self._roi_dev


class FaceMatcher:
    """Face matcher.

    Parameters
    ----------
    similarity_thresh : float, optional, (default=0.5)
        Similarity threshold for face matching. Two faces are considered to
        belong to the same person if the similarity between them is higher
        than `similarity_thresh`. A similarity of `1` means that the two
        faces are exactly the same, while a similarity of `0` means that the
        two faces are very different. Similarity between two faces if
        computed on the basis of the distance between their embeddings
        vectors.
    """
    def __init__(
        self,
        similarity_thresh: float = 0.5
    ):
        self._matcher = cv.BFMatcher(cv.NORM_L2)
        self._distance_thresh: float
        self.similarity_thresh = similarity_thresh

    def match(self, x_test, x_train, y_train):
        """Match two sets of face embeddings vectors

        Parameters
        ----------
        x_test : array-like of shape = [n_tests, n_features]
            The test set of face embeddings vectors. Each entry represents a
            embeddings vector of a face.

        x_train : array-like of shape = [n_trains, n_features]
            The train set of face embeddings vectors. Each entry represents a
            embeddings vector of a face.

        y_train : array-like of shape = [n_trains]
            The train set of face labels. Each entry represents the label of
            the corresponding entry in `x_train`.

        Returns
        -------
        y_test : list of length = n_trains
            The matched face labels. Each entry represents the matched label
            of the corresponding entry in `x_test`.

        similarity : list of length = n_trains
            The similarity scores of matched face labels. Each entry
            represents the similarity score of the the corresponding entry
            in `x_test`.
        """

        matches = self._matcher.radiusMatch(
            x_test, x_train, self._distance_thresh
        )

        y_test = []
        similarity = []
        for i, matches_i in enumerate(matches):
            y = [y_train[m.trainIdx] for m in matches_i]
            d = np.float32([m.distance for m in matches_i])

            y, ind, count = np.unique(y, return_index=True, return_counts=True)
            d = d[ind] / count

            if len(y):
                ind = np.argsort(d)
                y = y[ind]
                d = d[ind]

            y_test.append(y)
            similarity.append(distance_to_similarity(d))

        return y_test, similarity

    @property
    def similarity_thresh(self):
        return distance_to_similarity(self._distance_thresh)

    @similarity_thresh.setter
    def similarity_thresh(self, value: float):
        self._distance_thresh = similarity_to_distance(value)


class FrameAnalyzer:
    """Frame analyzer.

    Parameters
    ----------
    detector : FaceDetector
        A FaceDetector object.

    marker : FaceMarker or None
        A FaceMarker object.

    aligner: FaceAligner or None
        A FaceAligner object.

    encoder: FaceEncoder or None
        FaceEncoder object.

    detection_only: bool, optional, (default=False)
        Face analysis includes three steps: detection, alignment and
        encoding. If `detection_only = True`, only the detection step is
        done and `marker`, `aligner` and `encoder` can be set all to None.

    max_frame_size: int, optional, (default=0)
        If `store_frame = True` then the input image will be stored in each
        Face object created from a detected face. If the input image is
        stored, it will be resized so that is maximum dimension is lowest
        than `max_frame_size`. Set it to a value lower than or equal to zero
        to do not resize.

    store_frames: bool, optional, (default=False)
        If `store_frame = True` then the input image will be stored in each
        Face object created from a detected face.

    max_deviation: tuple of length 2 or None, optional, (default=None)
        Face analysis includes three steps: detection, alignment and
        encoding. At the alignment step, the nose deviation is computed. The
        nose deviation is a tuple where the first and second elements
        represents the horizontal and vertical deviation of nose landmark,
        respectively. If this deviation is higher than `align_max_deviation`
        (element-wise comparision), the face is discarded. For a deviation
        lower than ( 0.4, 0.3), the face is approximately frontal in both
        planes. Set it to `None` to allow any deviation.

    marking_min_score: float, optional, (default=0.9)
        Face analysis includes three steps: detection, alignment and
        encoding. At the alignment step, face landmarks are computed and a
        marking score between 0 and 1 is assigned to each face. Lower scores
        mean lower face alignment confidence. Faces with marking score lower
        than the `marking_min_score` will be discarded.

    face_padding: float, optional, (default=0.2)
        After a face is detected, the corresponding face image is extracted
        from the original image and sent to a face alignment stage. Padding the
        detected face box can help to obtain better alignment results. Padding
        value is a number between 0 and 1, relative to the face box height.
    """

    def __init__(
        self,
        detector: FaceDetector,
        marker: FaceMarker,
        aligner: FaceAligner,
        encoder: FaceEncoder,
        detection_only: bool = False,
        max_frame_size: int = 0,
        store_frames: bool = False,
        max_deviation: Tuple[float, float] = None,
        marking_min_score: float = 0.9,
        face_padding: float = 0.2
    ):
        self.face_detector: FaceDetector = detector
        self.face_marker: FaceMarker = marker
        self.face_aligner: FaceAligner = aligner
        self.face_encoder: FaceEncoder = encoder
        self.detection_only: bool = detection_only
        self.max_frame_size: int = max_frame_size
        self.store_frames: bool = store_frames
        self.max_deviation: Tuple[float, float] = max_deviation
        self.marking_min_score: float = marking_min_score
        self.face_padding: float = face_padding

    def find_faces(
        self,
        image: np.ndarray,
        timestamp: float = 0
    ):
        """Detects, align and encode faces in a image.

        Parameters
        ----------
        image : array-like of `dtype=np.uint8`
            Input image to be analyzed.

        timestamp : float, optional, (default=0)
            A timestamp to be stored in each of the created Face objects.

        Returns
        -------
        faces : list of mtypes.Face objects of length=n_faces
            A list of a Face objects, each of them containing information
            about a detected face in the input image.
        embeddings : array_like of shape = [n_faces, 512]
            Embeddings vectors of detected faces. Each entry is the  embedding
            vector of the corresponding face in `faces`.
        """

        h, w = image.shape[0:2]
        faces = []
        embeddings = []

        boxes, detect_scores = self.face_detector.detect(image)
        n_boxes = len(boxes)

        if n_boxes == 0:
            return faces, embeddings

        face_images = [image[b[1]:b[3], b[0]:b[2]] for b in boxes]

        padded_face_images = face_images
        if self.face_padding != 0:
            pad = self.face_padding
            paddings = [int(pad * (b[2] - b[0])) for b in boxes]
            boxes = [(
                max(0, b[0] - p),
                max(0, b[1] - p),
                min(b[2] + p, w - 1),
                min(b[3] + p, h - 1),
            ) for b, p in zip(boxes, paddings)]

            padded_face_images = [image[b[1]:b[3], b[0]:b[2]] for b in boxes]
        else:
            paddings = [0] * n_boxes

        boxes = [(b[0] / w, b[1] / h, b[2] / w, b[3] / h) for b in boxes]

        frame = None
        if self.store_frames and n_boxes:
            frame_image = image
            if max(image.shape[0:2]) > self.max_frame_size > 0:
                frame_image, _ = cvtlib.image.resize(
                    image, self.max_frame_size
                )
            frame = Frame(image=frame_image)

        if self.detection_only:
            for i in range(n_boxes):
                faces.append(Face(
                    image=padded_face_images[i],
                    box=boxes[i],
                    frame=frame,
                    detect_score=detect_scores[i],
                    timestamp=timestamp,
                    padding=paddings[i]
                ))
        else:
            face_marks, mark_scores = self.face_marker.mark(face_images)
            aligned_face_images = []
            max_deviation = self.max_deviation
            marking_min_score = self.marking_min_score
            for i in range(n_boxes):
                face_image, nose_deviation = self.face_aligner.align(
                    padded_face_images[i], face_marks[i]
                )
                if (mark_scores[i] > marking_min_score) and (
                    max_deviation is None or (
                        nose_deviation[0] <= max_deviation[0]) and (
                        nose_deviation[1] <= max_deviation[1]
                    )
                ):
                    aligned_face_images.append(face_image)
                    faces.append(Face(
                        image=padded_face_images[i],
                        box=boxes[i],
                        frame=frame,
                        landmarks=face_marks[i] + paddings[i],
                        nose_deviation=nose_deviation,
                        detect_score=detect_scores[i],
                        mark_score=mark_scores[i],
                        timestamp=timestamp,
                        padding=paddings[i]
                    ))

            embeddings = []
            if len(faces):
                embeddings = self.face_encoder.encode(aligned_face_images)
                for i, face in enumerate(faces):
                    face.embeddings = embeddings[i]

        return faces, embeddings


VIDEO_PAUSE_SLEEP = 3
VIDEO_DEFAULT_FRAME_RATE = 24


class VideoAnalyzer:
    """Video analyzer.

    Parameters
    ----------

    frame_analyzer: FrameAnalyzer
        A FrameAnalyzer object.

    video_capture: cvtlib.VideoCapture
        A VideoCapture object.

    real_time: bool, optional, (default=True)
        Whether the video capture is from a camera or not. Video analysis can
        be done from a real time source, like a camera, or from a non-real time
        source, like a video file. Although the video analysis for both types
        of sources is similar,some differences exists. For example, for a real
        time source when the algorithm finish analyzing one frame, it waits
        (sleeps) a time equal to `detect_interval` before processing a new
        frame. For a video file, the algorithms jumps directly into the next
        target frame after analyzing the current frame.

    detect_interval: float, optional (default=1)
        Interval in seconds to analyze frames in video processing. In order to
        save processing time, not all frames are analyzed during video
        processing. For a real time source, when the algorithm finish analyzing
        one frame, it waits (sleeps) a time equal to `detect_interval` before
        processing a new frame. Similarly, for a video file, the algorithms
        jumps directly into the next target frame located
        `video_detect_interval` seconds beyond. Setting it to a reasonable
        value can help to save considerable processing time, taking into
        account the (typical) low movement speed of people.

    faces_memory: float, optional (default=30)
        Time memory in seconds of faces during video analysis. During video
        analysis, the algorithm try to match detected faces and group matched
        faces into Subject objects. A face is added to a Subject object only if
        no face have been added to that subject in the previous `face_memory`
        seconds. This setting helps to control the growth without limits of the
        number of faces added to a subject. Also, as face images belonging to
        the same person in successive nearby frames may be very similar, it
        also avoid storing redundant information.

    similarity_thresh: float, optional (default=0.5)
        Similarity threshold for face matching. During face matching, two faces
        are considered to belong to the same person if the similarity between
        them is higher than `similarity_thresh`. A similarity of `1` means that
        the two faces are exactly the same, while a similarity of `0` means
        that the two faces are very different. Similarity between two faces if
        computed on the basis of the distance between their embeddings vectors.

    mode: str, optional, (default='all')
        Video analysis mode. Video analysis can be executed in two different
        modes: "all" or "hunt". In the "all" mode, the algorithm tries to
        register all the detected faces, grouping them in Subject objects. In
        the "hunt" mode the algorithm tries to match the detected faces with a
        predefined list of face embeddings and to register only matched faces.

    hunt_embeddings: array_like of shape = [n_hunts, 512] or None
        Embeddings vectors to hunt for. If `mode = "hunt"` the algorithm tries
        to match the detected faces against a set of embeddings vectors in
        `hunt_embeddings` that represents the hunted faces, and to register
        only matched faces. Each embedding vector is of length 512, as returned
        by encoding.FaceEncoder.

    hunt_keys: array_like of shape = [n_hunts] or None
        List of face keys associated to the hunt embeddings. If `video_mode =
        "hunt"` the algorithm tries to match the detected faces against a set
        of embeddings vectors in `hunt_embeddings` that represents the hunted
        faces, and to register only matched faces. Each embedding vector of a
        hunted face must be associated to face key in `hunt_keys`, which
        is a list of integers numbers of length `n_hunts`.

    start_at: float, optional, (default=0)
        Starting time in seconds of processing for video source files. If
        `real_time = False`, the video file will be analyzed starting from
        `start_at` seconds. Note that if the duration of the video file is
        lower than `start_at` or `start_at` is higher than `stop_at`,
        `video_start_at` will be ignored.

    stop_at: float, optional, (default=-1)
        Stopping time in seconds of processing for video source files. If
        `real_time = False`, the video file will analysis will be terminated at
        `stop_at` seconds. Note that if the duration of the video file is lower
        than `stop_at` or `start_at` is higher than `stop_at`, `stop_at` will
        be ignored. Set it to a negative value to run the analysis until the
        end of the video file.

    roi_adapt: bool, optional, (default=False)
        Use adaptive ROI during video analysis. If set to `True`, adaptive
        Region-Of-Interest (ROI) will be used during video analysis. This can
        help to save considerable processing time if the video comes from a
        static camera, as in these case the faces in the video frames are
        likely to be located inside a rectangle that changes little (in size
        and  location) over time. So, instead analyzing the entire frame, only
        a sub-region of the frame is analyzed. This rectangle is estimated at
        run time an used once it converges.
    """

    LOGGING_FRAME_INTERVAL = 100
    MAX_NULL_FRAMES = 10
    MAX_STORED_SUBJECTS = 20
    STORED_SUBJECTS_SHRINK = 0.3

    MODE_HUNT = 'hunt'
    MODE_ALL = 'all'

    def __init__(
        self,
        frame_analyzer: FrameAnalyzer,
        video_capture: VideoCapture,
        real_time: bool = True,
        detect_interval: float = 1,
        faces_memory: float = 30,
        similarity_thresh: float = 0.5,
        mode: str = MODE_ALL,
        hunt_embeddings: np.ndarray = None,
        hunt_keys: list = None,
        start_at: float = 0,
        stop_at: float = -1,
        roi_adapt: bool = False
    ):

        self.frame_analyzer: FrameAnalyzer = frame_analyzer
        self.video_capture: VideoCapture = video_capture
        self.video_capture.open()
        self.real_time: bool = real_time
        self.faces_memory: float = faces_memory
        self.similarity_thresh: float = similarity_thresh
        self.detect_interval: float = detect_interval
        self.mode: str = mode
        self.start_at: float = start_at
        self.stop_at: float = stop_at
        self.roi_adapt: bool = roi_adapt

        self.hunt_keys = hunt_keys

        if self.mode == self.MODE_HUNT:
            error = ValueError(
                f'Invalid "hunt_embeddings" parameter. It must be a list of lists of numbers, '
                f'each one having {EMBEDDINGS_LENGTH} elements.'
            )
            try:
                self.hunt_embeddings: np.ndarray = np.array(hunt_embeddings, np.float32)
            except ValueError:
                raise error

            if len(self.hunt_embeddings) == 0:
                logger.warning('Mode is set to "hunt" and "hunt_embeddings" parameter is zero length.')

            if len(self.hunt_embeddings) and (
                len(self.hunt_embeddings.shape) != 2 or
                self.hunt_embeddings.shape[1] != EMBEDDINGS_LENGTH
            ):
                raise error

            if self.hunt_keys is not None:
                if len(self.hunt_keys) != len(self.hunt_embeddings):
                    raise ValueError('Parameters "hunt_keys" and "hunt_embeddings" have different lengths.')

        self._run: bool = False
        self._paused: bool = False
        self.timestamp: float = 0
        self.frame: np.ndarray = np.array([])
        self.stated_at: float = -1

        self.subjects: Dict[int, Subject] = {}
        self.embeddings: List[np.ndarray] = []
        self.keys: List[int] = []
        self.key_count: int = 0

        if not self.real_time:
            frame_rate = VIDEO_DEFAULT_FRAME_RATE
            try:
                frame_rate = self.video_capture.frame_rate
            except AttributeError:
                logger.warning(
                    f'Video frame rate could not be determined. '
                    f'Assuming {VIDEO_DEFAULT_FRAME_RATE} fps by default.'
                )
            self.frame_interval = int(frame_rate * self.detect_interval)

            self._check_time_control()

        self.frame_rate: float = 0
        self.frames_count: int = 0
        self.faces_count: int = 0
        self.processing_time: float = 0

        self.matcher: cv.BFMatcher = cv.BFMatcher(cv.NORM_L2)
        self.dist_thresh = similarity_to_distance(self.similarity_thresh)

        self.adaptive_roi = AdaptiveRoi(thresh=0.025)

    def stop(self):
        """Stop video analysis"""

        self._run = False
        self._paused = False

    def pause(self):
        """Pause video analysis"""

        self._paused = True

    def resume(self):
        """Resume video analysis"""

        self._paused = False

    def run(
        self,
        frame_callback: callable = None,
        update_subject_callback: callable = None
    ):
        """Run video analysis.

        Parameters
        ----------
        frame_callback : callable or None, optional, (default=None)
            A function called every time a video frame is analyzed. It

        update_subject_callback : callable or None, optional, (default=None)
            A function called every time a video frame is analyzed. It
        """

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._run = True
        self._paused = False
        self.key_count = 0

        last_detect_at = 0
        self.frames_count = 0
        self.processing_time = 0

        cap = self.video_capture

        if self.start_at > 0 and not self.real_time:
            try:
                cap.goto_time(self.start_at)
            except AttributeError:
                self.start_at = -1
                logger.warning(
                    'Video can be advanced, ignoring parameter "start_at".'
                )
            try:
                _ = cap.timestamp
            except AttributeError:
                self.stop_at = -1
                logger.warning(
                    'Video can be advanced, ignoring parameter "stop_at".'
                )

        logger.info('Starting video analysis.')

        if self.mode == self.MODE_HUNT:
            for index, embedding in enumerate(self.hunt_embeddings):
                self.subjects[index] = Subject(
                    faces=[],
                    embeddings=embedding
                )
                if self.hunt_keys is not None:
                    self.subjects[index].data['hunt_key'] = self.hunt_keys[index]

        cap_fail_counter = 0
        cap_success = False
        self.start_at = time()

        while self._run:

            if self.real_time:
                wait = last_detect_at + self.detect_interval - time()
                if wait > 0:
                    sleep(wait)

            while self._paused:
                sleep(VIDEO_PAUSE_SLEEP)

            self.frame = None

            if self.real_time:
                self.timestamp = time()
            else:
                try:
                    self.timestamp = cap.timestamp
                except AttributeError:
                    frame_rate = VIDEO_DEFAULT_FRAME_RATE
                    try:
                        frame_rate = cap.frame_rate
                    except AttributeError:
                        pass
                    self.timestamp = cap.frame_number / frame_rate

            if not self.real_time and (0 < self.stop_at < self.timestamp):
                logger.info('Stopping analysis, "stop_at" time reached.')
                break

            if self.real_time:
                do_detection = (
                    self.timestamp - last_detect_at >= self.detect_interval
                )
                if do_detection:
                    self.frame, cap_success = cap.next_frame()
            else:
                do_detection = cap.frame_number % self.frame_interval == 0
                if do_detection:
                    self.frame, cap_success = cap.next_frame()
                else:
                    cap_success = cap.grab_next()

            if do_detection and self.frame is not None:
                cap_fail_counter = 0
                last_detect_at = self.timestamp

                frame_h, frame_w, = self.frame.shape[0:2]

                roi_box = None
                if self.roi_adapt:
                    roi_box = self.adaptive_roi.roi

                roi_frame = self.frame
                if roi_box is not None:
                    roi_box = (
                        int(frame_w * roi_box[0]),
                        int(frame_h * roi_box[1]),
                        int(frame_w * roi_box[2]),
                        int(frame_h * roi_box[3]),
                    )
                    roi_frame = self.frame[
                        roi_box[1]:roi_box[3],
                        roi_box[0]:roi_box[2]
                    ]

                faces, embeddings = self.frame_analyzer.find_faces(
                    roi_frame, timestamp=self.timestamp
                )

                if self.roi_adapt:
                    roi_h, roi_w = roi_frame.shape[0:2]
                    for face in faces:
                        if roi_box is not None:
                            face.box = (
                                (roi_box[0] + int(roi_w * face.box[0]))/frame_w,
                                (roi_box[1] + int(roi_h * face.box[1]))/frame_h,
                                (roi_box[0] + int(roi_w * face.box[2]))/frame_w,
                                (roi_box[1] + int(roi_h * face.box[3]))/frame_h,
                            )
                        self.adaptive_roi.add_box(face.box)

                if len(faces) > 0:
                    self.faces_count += len(faces)

                    if self.frame_analyzer.detection_only:
                        self._create_subjects(faces, update_subject_callback)
                    else:
                        if self.mode == self.MODE_ALL:
                            self._step_mode_all(
                                faces, embeddings, update_subject_callback
                            )
                        elif self.mode == self.MODE_HUNT:
                            self._step_mode_hunt(
                                faces, embeddings, update_subject_callback
                            )

                self.processing_time = (time() - self.start_at)
                self.frames_count += 1
                self.frame_rate = self.frames_count / self.processing_time
                logger.debug(f'Current frame rate: {self.frame_rate:.1f} FPS.')

            if not cap_success:
                if self.real_time:
                    sleep(1)
                cap_fail_counter += 1
                if cap_fail_counter > self.MAX_NULL_FRAMES:
                    break

            if self.frame is not None and frame_callback is not None:
                frame_callback()
        # while self._run:

        logger.info('Video analysis finished.')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _step_mode_all(
        self,
        faces: List[Face],
        embeddings: np.ndarray,
        update_subject_callback: callable
    ):
        if len(self.subjects) == 0:
            self._create_subjects(faces, update_subject_callback)
            self.embeddings.extend(list(embeddings))
            self.keys.extend(self.subjects.keys())
        else:
            matches = self.matcher.knnMatch(
                embeddings,
                np.array(self.embeddings),
                1
            )
            update_faces = []
            update_keys = []
            create_faces = []
            index_create = []

            for match in matches:
                if len(match):
                    if match[0].distance < self.dist_thresh:
                        update_faces.append(faces[match[0].queryIdx])
                        update_keys.append(self.keys[match[0].trainIdx])
                    else:
                        create_faces.append(faces[match[0].queryIdx])
                        index_create.append(match[0].queryIdx)

            if len(update_faces):
                self._update_subjects(
                    update_keys,
                    update_faces,
                    update_subject_callback
                )

            if len(create_faces):
                created_keys = self._create_subjects(
                    create_faces,
                    update_subject_callback
                )
                self.embeddings.extend(list(embeddings[index_create]))
                self.keys.extend(created_keys)
        self._check_subjects_size()

    def _step_mode_hunt(
        self,
        faces: List[Face],
        embeddings: np.ndarray,
        update_subject_callback: callable
    ):
        matches = self.matcher.knnMatch(embeddings, self.hunt_embeddings, 1)
        update_faces = []
        update_keys = []
        for i, match in enumerate(matches):
            if match[0].distance < self.dist_thresh:
                update_faces.append(faces[match[0].queryIdx])
                update_keys.append(match[0].trainIdx)

        self._update_subjects(
            update_keys,
            update_faces,
            update_subject_callback
        )

    def _create_subjects(self, faces, update_subject_callback=None):
        created_keys = []
        store = not self.frame_analyzer.detection_only
        for face in faces:
            self.key_count += 1
            key = self.key_count
            subject = Subject(
                faces=[face],
                embeddings=face.embeddings,
                key=key
            )
            face.subject = subject
            if update_subject_callback is not None:
                update_subject_callback(face)
            if store:
                self.subjects[key] = subject
                created_keys.append(key)

        return created_keys

    def _update_subjects(self, keys, faces, update_subject_callback=None):
        for key, face in zip(keys, faces):
            subject = self.subjects[key]
            if (self.timestamp - subject.last_updated) > self.faces_memory:
                self.subjects[key].append_face(face)
                face.subject = self.subjects[key]
                if update_subject_callback is not None:
                    update_subject_callback(face)

    def _check_subjects_size(self):
        total = len(self.subjects)
        del_count = int(self.STORED_SUBJECTS_SHRINK * total)
        if del_count and total > self.MAX_STORED_SUBJECTS:
            delete_keys = self.keys[0:del_count]
            for key in delete_keys:
                del self.subjects[key]

            del self.keys[0:del_count]
            del self.embeddings[0:del_count]

    def _check_time_control(self):
        if self.start_at > 0 or self.stop_at > 0:
            if (
                self.start_at > 0 and self.stop_at > 0
            ) and self.stop_at < self.start_at:
                logger.warning(
                    'Parameter "stop_at" is lower than "start_at", ignoring.'
                )
                self.start_at = 0
                self.stop_at = -1
            else:
                try:
                    video_duration = self.video_capture.duration_seconds
                    if self.start_at > 0 and self.start_at > video_duration:
                        logger.warning(
                            'Parameter "start_at" is higher than video '
                            'duration, ignoring it.'
                        )
                        self.start_at = 0
                    if self.stop_at > 0 and self.stop_at > video_duration:
                        logger.warning(
                            'Parameter "stop_at" is higher than video '
                            'duration, ignoring it.'
                        )
                        self.stop_at = -1
                except AttributeError:
                    logger.warning(
                        'Video duration could not be determined. Ignoring '
                        'parameters "start_at" and "stop_at"'
                    )
                    self.start_at = 0
                    self.stop_at = -1
