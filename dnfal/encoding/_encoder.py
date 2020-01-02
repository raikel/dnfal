from typing import List

import cv2
import numpy as np
import torch

from ._model import IR_50

INPUT_SIZE = 112
USE_TTA = True


class FaceEncoder:
    """Face encoder.

    Parameters
    ----------
    weights_path : str
        Absolute path to file containing pretrained model weights.
    """

    def __init__(self, weights_path: str, force_cpu: bool = False):
        self.model: IR_50 = IR_50((INPUT_SIZE, INPUT_SIZE))

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

    def encode(self, images: List[np.ndarray]):
        """Encode face images.

        Parameters
        ----------
        images : list of array_like images of length = n_faces
            A list of face images to be encoded.

        Returns
        -------
        embeddings : array_like of shape = [n_faces, 512]
            Embeddings vectors. Each entry is the embedding vector of the
            corresponding input face image.
        """

        if USE_TTA:
            shape = (2*len(images), 3, INPUT_SIZE, INPUT_SIZE)
            batch = np.zeros(shape, dtype=np.float32)
        else:
            shape = (len(images), 3, INPUT_SIZE, INPUT_SIZE)
            batch = np.zeros(shape, dtype=np.float32)

        for index, image in enumerate(images):
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

            image_flipped = None
            if USE_TTA:
                image_flipped = _image_transform(cv2.flip(image, 1))

            image = _image_transform(image)

            if USE_TTA:
                batch[2 * index, :, :, :] = image
                batch[2 * index + 1, :, :, :] = image_flipped
            else:
                batch[index, :, :, :] = image

        batch_tensor = torch.from_numpy(batch)
        if self.gpu is not None:
            batch_tensor = batch_tensor.to(self.gpu)

        with torch.no_grad():
            emb_batch = self.model(batch_tensor)
            if USE_TTA:
                emb_batch = emb_batch[0::2] + emb_batch[1::2]

            embeddings = _l2_norm(emb_batch)

        if self.gpu is not None:
            embeddings = embeddings.cpu()

        return embeddings.numpy()


def _image_transform(image: np.ndarray):
    # BGR to RGB
    image = image[:, :, ::-1]
    # load numpy to tensor
    image = image.transpose((2, 0, 1))
    image = np.reshape(image, [1, 3, INPUT_SIZE, INPUT_SIZE])
    if image.dtype != np.float32:
        image = np.float32(image)
    image = (image - 127.5) * 0.0078125
    return image


def _l2_norm(embeddings, axis=1):
    return torch.div(embeddings, torch.norm(embeddings, 2, axis, True))
