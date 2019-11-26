from base64 import b64encode
from typing import List, Tuple

import cv2 as cv
import numpy as np


class Frame:
    def __init__(
        self,
        image: np.ndarray,
        key: int = None
    ):
        self.key: int = key
        self.data: dict = {}
        self.image: np.ndarray = image

    def serialize(self):
        return {
            'key': self.key,
            'image': self.image_bytes
        }

    @property
    def image_bytes(self):
        return cv.imencode('.jpg', self.image)[1]


class Face:
    def __init__(
        self,
        box: tuple,
        image: np.ndarray,
        key: int = None,
        frame: Frame = None,
        subject: 'Subject' = None,
        landmarks: np.ndarray = None,
        embeddings: np.ndarray = None,
        detect_score: float = 0.0,
        mark_score: float = 0.0,
        nose_deviation: Tuple[float, float] = (0.0, 0.0),
        timestamp: float = 0,
        padding: float = 0
    ):
        self.key: int = key
        self.data: dict = {}
        self.box: tuple = box
        self.image: np.ndarray = image
        self.frame: Frame = frame
        self.subject: 'Subject' = subject
        self.embeddings: np.ndarray = embeddings
        self.landmarks: np.ndarray = landmarks
        self.detect_score: float = detect_score
        self.mark_score: float = mark_score
        self.nose_deviation: Tuple[float, float] = nose_deviation
        self.timestamp: float = timestamp
        self.padding: float = padding

    def serialize(self):

        frame = None
        if self.frame is not None:
            frame = self.frame.serialize()

        return {
            'box': self.box,
            'image': self.image_bytes,
            'frame': frame,
            'embeddings': self.embeddings.tolist(),
            'landmarks': self.landmarks.tolist(),
            'timestamp': self.timestamp
        }

    @property
    def image_bytes(self):
        return cv.imencode('.jpg', self.image)[1]

    @staticmethod
    def image_to_base64(image: np.ndarray) -> str:
        return str(b64encode(cv.imencode('.jpg', image)[1]), 'utf-8')

    def __str__(self):
        w, h = self.box[2] - self.box[0], self.box[3] - self.box[1]
        ret = f'score: {self.detect_score:.2f}, size: {w}x{h}, nose deviation: ' \
              f'({(self.nose_deviation[0] * 100):.2f} %, {(self.nose_deviation[1] * 100):.2f} %)'
        return ret

    def __repr__(self):
        return self.__str__()


class Subject:
    def __init__(
        self,
        faces: List[Face],
        embeddings: np.ndarray,
        key: int = None
    ):
        self.key: int = key
        self.data: dict = {}
        self.faces: List[Face] = faces
        self.embeddings: np.ndarray = embeddings

    def append_face(self, face: Face):
        self.faces.append(face)

    def serialize(self):
        return {
            'faces': [face.serialize() for face in self.faces],
            'embeddings': self.embeddings.tolist()
        }

    @property
    def last_updated(self) -> float:
        if len(self.faces):
            return max([face.timestamp for face in self.faces])
        return 0
