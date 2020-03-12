import numpy as np
import torch
import cv2 as cv

from fnms import nms

from ._model import CSPNet

SCALE = np.float32([0.01712475, 0.017507, 0.01742919])
OFFSET = np.float32([2.11790393, 2.03571429, 1.80444444])


class PersonDetector:
    """Full-body person detector.

    Parameters
    ----------
    weights_path : str
        Absolute path to file containing pretrained model weights.
    """

    def __init__(
        self,
        weights_path: str,
        score_thresh: float = 0.2,
        nms_thresh: float = 0.3,
        resize_height: int = 256,
        force_cpu: bool = False
    ):
        self.score_thresh: float = score_thresh
        self.nms_thresh: float = nms_thresh
        self.resize_height: int = resize_height
        self.model: CSPNet = CSPNet()

        # noinspection PyTypeChecker
        self.gpu: torch.device = None
        if torch.cuda.is_available() and not force_cpu:
            self.gpu = torch.device('cuda', 0)

        if self.gpu is None:
            cpu = torch.device('cpu')
            pretrained_dict = torch.load(weights_path, map_location=cpu)
        else:
            pretrained_dict = torch.load(weights_path)

        self.model.load_state_dict(pretrained_dict)

        if self.gpu is not None:
            self.model = self.model.to(self.gpu)

        self.model.eval()

    def detect(self, image: np.ndarray):
        """Predict gender and age of a list of face images.

        Parameters
        ----------
        image : array_like image
            A list of face images to be encoded.

        Returns
        -------
        genders : list of length = n_faces
            The predicted genders. Each entry represents the predicted gender
            of the corresponding entry in `images`.

        ages : list of length = n_faces
            The predicted ages. Each entry represents the predicted age
            of the corresponding entry in `images`.
        """

        image_h, image_w = image.shape[0:2]
        image, scale = _image_transform(image, self.resize_height, self.gpu)

        with torch.no_grad():
            # Eval model
            scores, heights, offsets = self.model(image)

        torch.cuda.synchronize(self.gpu)

        boxes, scores = _get_boxes(
            scores.cpu().numpy(),
            heights.cpu().numpy(),
            offsets.cpu().numpy(),
            (image_w, image_h),
            score_thresh=self.score_thresh,
            down=4,
            nms_thresh=self.nms_thresh
        )

        if scale != 1.0 and len(boxes):
            boxes = boxes / scale

        return np.int32(boxes), scores


def _image_transform(image: np.ndarray, resize_height: int, gpu=None):
    image_h, image_w = image.shape[0:2]
    scale = 1.0

    if image_h != resize_height:
        scale = resize_height / image_h
        image = cv.resize(image, (int(scale * image_w), resize_height))
        image_h, image_w = image.shape[0:2]

    if image_w % 32 != 0:
        pad_w = 32 - image_w % 32
        image = cv.copyMakeBorder(
            image, 0, 0, 0, pad_w,
            borderType=cv.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

    image = image[:, :, ::-1]
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    image = SCALE * image - OFFSET
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0)
    if gpu is not None:
        image = image.to(gpu)
    return image, scale


def _get_boxes(
    scores,
    heights,
    offsets,
    image_size,
    score_thresh=0.1,
    down=4,
    nms_thresh=0.3
):

    scores = np.squeeze(scores)
    heights = np.squeeze(heights)
    offset_y = offsets[0, 0, :, :]
    offset_x = offsets[0, 1, :, :]
    y_c, x_c = np.where(scores > score_thresh)
    boxes = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(heights[y_c[i], x_c[i]]) * down
            w = 0.41 * h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            score = scores[y_c[i], x_c[i]]
            x1 = int(max(0, (x_c[i] + o_x + 0.5) * down - w / 2))
            y1 = int(max(0, (y_c[i] + o_y + 0.5) * down - h / 2))
            x2 = min(int(x1 + w), image_size[0] - 1)
            y2 = min(int(y1 + h), image_size[1] - 1)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2, score))

        if len(boxes):
            boxes = np.float32(boxes)
            keep = nms(boxes, nms_thresh, force_cpu=True)
            boxes = boxes[keep, :]

    if len(boxes):
        return boxes[:, 0:4], boxes[:, 4]

    return [], []

