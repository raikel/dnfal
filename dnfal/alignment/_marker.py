from typing import List

import torch
import numpy as np
import cv2 as cv

from ._model import ONet

TRANSFORM_SIZE = (48, 48)


class FaceMarker:
    """Face marker.

    Parameters
    ----------
    weights_path : str
        Absolute path to file containing pretrained model weights.
    """

    def __init__(self, weights_path: str, force_cpu: bool = False):
        self.onet: ONet = ONet()

        weights: dict = np.load(weights_path, allow_pickle=True)[()]
        for n, p in self.onet.named_parameters():
            p.data = torch.tensor(weights[n], dtype=torch.float)

        # noinspection PyTypeChecker
        self.gpu: torch.device = None
        if torch.cuda.is_available() and not force_cpu:
            self.gpu = torch.device('cuda', 0)
            self.onet = self.onet.to(self.gpu)

        self.onet.eval()

    def mark(self, images: List[np.ndarray]):
        """Estimate face landmark locations.

        Parameters
        ----------
        images : list of array_like images of length = n_faces
            A list of face images to be marked.

        Returns
        -------
        marks : array_like of shape = [n_faces, 5, 2]
            The estimated face landmarks. Each entry is the landmark set for
            the corresponding input face. Each entry in a landmark set contains
            the estimated (x, y) locations of the face landmarks in the
            following order: (0) left eye center, (1) right eye center,
            (2) nose center, (3) mouth left corner and (4) mouth right corner.
        scores : array_like of shape = [n_faces]
            The marking score for each estimated landmark set. The score is a
            number between 0 and 1 that represents the confidence of the
            landmarks estimation.
        """

        n_images = len(images)
        stack_shape = (n_images, 3) + TRANSFORM_SIZE
        image_stack = np.zeros(stack_shape, dtype=np.float32)
        image_sizes = np.zeros((n_images, 2), dtype=np.float32)

        for i in range(n_images):
            image_stack[i, :, :, :] = _image_transform(images[i])
            image_sizes[i, :] = images[i].shape[-2::-1]

        image_stack = torch.from_numpy(image_stack)
        if self.gpu is not None:
            image_stack = image_stack.to(self.gpu)

        with torch.no_grad():
            marks, offsets, scores = self.onet(image_stack)

        if self.gpu is not None:
            marks = marks.cpu().data.numpy()
            scores = scores.cpu().data.numpy()
        else:
            marks = marks.data.numpy()
            scores = scores.data.numpy()

        marks[:, 0:5] = np.expand_dims(image_sizes[:, 0], 1) * marks[:, 0:5]
        marks[:, 5:10] = np.expand_dims(image_sizes[:, 1], 1) * marks[:, 5:10]

        scores = scores[:, 1].reshape((-1,))
        marks = marks.reshape(-1, 2, 5).transpose(0, 2, 1)

        return marks, scores


def _image_transform(image: np.ndarray):
    image = cv.resize(image, TRANSFORM_SIZE, cv.INTER_LINEAR)
    image = image[:, :, ::-1]
    image = np.float32(image)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = (image - 127.5) * 0.0078125
    return image
