from itertools import product as product
from math import ceil
from typing import Tuple
from os import path

import numpy as np
import torch

from ._model import FaceBoxes
from fnms import nms

MIN_SIZES = [[32, 64, 128], [256], [512]]
STEPS = [32, 64, 128]
CLIP = False
VARIANCE = [0.1, 0.2]
KEEP_TOP_K = 5000
TRANSFORM_OFFSET = np.float32((-104, -117, -123))


class FaceDetector:
    """Face detector.

    Parameters
    ----------
    weights_path : str
        Absolute path to file containing pretrained model weights.

    force_cpu : bool, optional, (default=False)
        Whether to force model run on CPU even if CUDA is available.

    min_score : float, optional, (default=0.9)
        Minimum confidence score of detected faces. Faces with scores lower
        than the `min_score` will be discarded.

    nms_thresh : float, optional, (default=0.5)
        Non-maximum suppression threshold of detected faces. From the set of
        face boxes which overlaps more than the corresponding NMS threshold the
        one with the highest score will be selected and the others discarded.

    min_height : int, optional, (default=24)
        Minimum height in pixels of face to be detected. Faces with height
        lower than `min_height` will be discarded.
    """

    def __init__(
        self,
        weights_path: str,
        force_cpu: bool = False,
        min_score: float = 0.9,
        nms_thresh: float = 0.5,
        min_height: int = 24
    ):
        self.model: FaceBoxes = FaceBoxes(
            phase='test', size=None, num_classes=2
        )
        # noinspection PyTypeChecker
        self.gpu: torch.device = None
        if torch.cuda.is_available() and not force_cpu:
            self.gpu = torch.device('cuda', 0)

        if self.gpu is None:
            cpu = torch.device('cpu')
            pretrained_dict = torch.load(weights_path, map_location=cpu)
        else:
            pretrained_dict = torch.load(weights_path)

        self.model.load_state_dict(pretrained_dict, strict=False)
        if self.gpu is not None:
            self.model = self.model.to(self.gpu)

        self.model.eval()
        # noinspection PyTypeChecker
        self.image_size: Tuple[int, int] = None
        # noinspection PyTypeChecker
        self.priors: torch.Tensor = None
        # noinspection PyTypeChecker
        self.scale: torch.Tensor = None

        self.min_score: float = min_score
        self.nms_thresh: float = nms_thresh
        self.min_height: int = min_height

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate face landmark locations.

        Parameters
        ----------
        image : array_like image
            Input image to be analyzed.

        Returns
        -------
        boxes : array_like of shape = [n_faces, 4] and `dtype=np.int32`
            Detected face boxes. Each entry is face box. The first two values
            in a face box represent the top-left corner coordinates, while the
            latest two values represents the bottom-left corner coordinates.
        scores : array_like of shape = [n_faces] and `dtype=np.float`
            The detection score for each detected face. The score is a number
            between 0 and 1 that represents the confidence of the detection.
        """

        h, w = image.shape[0:2]
        face_boxes = np.array([])
        face_scores = np.array([])

        if self.image_size != (w, h):
            self._size_updated((w, h))
            self.image_size = (w, h)

        image = self._image_transform(image)

        locations, confidence = self.model(image)
        boxes = _decode(locations.data.squeeze(0), self.priors.data, VARIANCE)
        boxes *= self.scale
        boxes = boxes.cpu().numpy()
        scores = confidence.squeeze(0).data.cpu().numpy()[:, 1]

        if len(boxes) == 0:
            return face_boxes, face_scores

        # ignore low scores
        inds = np.where(scores > self.min_score)[0]

        if len(inds) == 0:
            return face_boxes, face_scores

        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:KEEP_TOP_K]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack(
            (boxes, scores[:, np.newaxis])
        ).astype(np.float32, copy=False)
        keep = nms(dets, self.nms_thresh, force_cpu=(self.gpu is None))
        dets = dets[keep, :]

        # keep only boxes with height higher than min_height
        heights = dets[:, 3] - dets[:, 1]
        keep = np.where(heights >= self.min_height)[0]
        dets = dets[keep, :]

        boxes = dets[:, 0:4].astype(np.int32)
        np.clip(boxes, a_min=(0, 0, 0, 0), a_max=(w, h, w, h), out=boxes)
        scores = dets[:, 4]

        return boxes, scores

    def _image_transform(self, image: np.ndarray):
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image += TRANSFORM_OFFSET
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0)
        if self.gpu is not None:
            image = image.to(self.gpu)
        return image

    def _size_updated(self, size):
        priors = _prior_boxes((size[1], size[0]))
        scale = torch.tensor([size[0], size[1], size[0], size[1]])
        if self.gpu is not None:
            priors = priors.to(self.gpu)
            scale = scale.to(self.gpu)
        self.priors = priors
        self.scale = scale


def _prior_boxes(shape: Tuple[int, int]) -> torch.Tensor:
    feature_maps = [
        [ceil(shape[0] / step), ceil(shape[1] / step)]
        for step in STEPS
    ]
    anchors = []
    for k, f in enumerate(feature_maps):
        min_sizes = MIN_SIZES[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes:
                s_kx = min_size / shape[1]
                s_ky = min_size / shape[0]
                if min_size == 32:
                    dense_cx = [
                        x * STEPS[k] / shape[1]
                        for x in [j + 0, j + 0.25, j + 0.5, j + 0.75]
                    ]
                    dense_cy = [
                        y * STEPS[k] / shape[0]
                        for y in [i + 0, i + 0.25, i + 0.5, i + 0.75]
                    ]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
                elif min_size == 64:
                    dense_cx = [
                        x * STEPS[k] / shape[1]
                        for x in [j + 0, j + 0.5]
                    ]
                    dense_cy = [
                        y * STEPS[k] / shape[0]
                        for y in [i + 0, i + 0.5]
                    ]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
                else:
                    cx = (j + 0.5) * STEPS[k] / shape[1]
                    cy = (i + 0.5) * STEPS[k] / shape[0]
                    anchors += [cx, cy, s_kx, s_ky]

    output = torch.tensor(anchors).view(-1, 4)
    if CLIP:
        output.clamp_(max=1, min=0)
    return output


def _decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
