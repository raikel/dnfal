from typing import List

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

from ._model import DenseNet

DENSE_NET_CLS_NUM = 751

SCALE = np.float32([0.01712475, 0.017507, 0.01742919])
OFFSET = np.float32([2.11790393, 2.03571429, 1.80444444])
INPUT_WIDTH = 256
INPUT_HEIGHT = 128


class PersonEncoder:
    """Full-body person encoder for re-identification.

    Parameters
    ----------
    weights_path : str
        Absolute path to file containing pretrained model weights.
    """

    def __init__(
        self,
        weights_path: str,
        force_cpu: bool = False
    ):
        self.model: DenseNet = DenseNet(class_num=DENSE_NET_CLS_NUM)

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

        self.model.classifier.classifier = nn.Sequential()

        if self.gpu is not None:
            self.model = self.model.to(self.gpu)

        self.model.eval()

    def encode(self, images: List[np.ndarray]):
        """Encode face images.

        Parameters
        ----------
        images : list of array_like images of length = n_persons
            A list of face images to be encoded.

        Returns
        -------
        embeddings : array_like of shape = [n_persons, 512]
            Embeddings vectors. Each entry is the embedding vector of the
            corresponding input person image.
        """

        batch_shape = (2 * len(images), 3, INPUT_HEIGHT, INPUT_WIDTH)
        batch = np.zeros(batch_shape, dtype=np.float32)

        for index, image in enumerate(images):
            batch[2 * index, :, :, :] = _image_transform(image)
            batch[2 * index + 1, :, :, :] = _image_transform(cv.flip(image, 1))

        batch_tensor = torch.from_numpy(batch)
        if self.gpu is not None:
            batch_tensor = batch_tensor.to(self.gpu)

        with torch.no_grad():
            embeddings = self.model(batch_tensor)
            embeddings = embeddings[0::2] + embeddings[1::2]
            embeddings_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
            embeddings = embeddings.div(embeddings_norm.expand_as(embeddings))

        if self.gpu is not None:
            embeddings = embeddings.cpu()

        return embeddings.numpy()


def _image_transform(image: np.ndarray):

    # BGR to RGB
    image = cv.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = image[:, :, ::-1]
    if image.dtype != np.float32:
        image = np.float32(image)
    image = SCALE * image - OFFSET
    image = image.transpose((2, 0, 1))
    image = np.reshape(image, (1, 3, INPUT_HEIGHT, INPUT_WIDTH))

    return image


