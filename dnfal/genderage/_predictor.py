from typing import List

import cv2
import numpy as np
import torch

from ._model import GenderAgeModel, AGE_CLS_COUNT

RESIZE = 256
CROP = 224
SCALE = np.float32([0.01712475, 0.017507, 0.01742919])
OFFSET = np.float32([2.11790393, 2.03571429, 1.80444444])

RESIZE_SHAPE = (RESIZE, RESIZE)

CROP_RECT = (
    int((RESIZE - CROP)/2),
    int((RESIZE - CROP)/2),
    int((RESIZE + CROP)/2),
    int((RESIZE + CROP)/2),
)

AGE_CLS_WEIGHTS = np.linspace(1, AGE_CLS_COUNT, AGE_CLS_COUNT, dtype=np.float32)


class GenderAgePredictor:
    """Gender and age predictor.

    Parameters
    ----------
    weights_path : str
        Absolute path to file containing pretrained model weights.
    """

    GENDER_WOMAN = 0
    GENDER_MAN = 1

    def __init__(
        self,
        weights_path: str,
        force_cpu: bool = False
    ):
        self.model: GenderAgeModel = GenderAgeModel()

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

    def predict(self, images: List[np.ndarray]):
        """Predict gender and age of a list of face images.

        Parameters
        ----------
        images : list of array_like images of length = n_faces
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

        shape = (len(images), 3, CROP, CROP)
        batch = np.zeros(shape, dtype=np.float32)

        for index, image in enumerate(images):
            image = _image_transform(image)
            batch[index, :, :, :] = image

        batch_tensor = torch.from_numpy(batch)
        if self.gpu is not None:
            batch_tensor = batch_tensor.to(self.gpu)

        with torch.no_grad():
            # Eval model
            genders_out, ages_probs = self.model(batch_tensor)

            genders_out = torch.softmax(genders_out, dim=1)
            genders_probs, genders_preds = torch.max(genders_out, dim=1)
            genders_probs = genders_probs.cpu().data.numpy()
            genders_preds = genders_preds.cpu().data.numpy()

            ages_probs = ages_probs.cpu().data.numpy()
            ages_probs = ages_probs.reshape((-1, AGE_CLS_COUNT))
            ages_preds = np.sum(ages_probs * AGE_CLS_WEIGHTS, axis=1)

            diff = AGE_CLS_WEIGHTS - ages_preds.reshape((-1, 1))
            age_vars = np.sqrt(np.mean(ages_probs * diff * diff, axis=1))

        return genders_preds, genders_probs, ages_preds, age_vars


def _image_transform(image: np.ndarray):

    # Resize
    if image.shape[0:2] != RESIZE_SHAPE:
        image = cv2.resize(image, (RESIZE, RESIZE))
    # Crop
    image = image[CROP_RECT[1]:CROP_RECT[3], CROP_RECT[0]:CROP_RECT[2]]
    # BGR to RGB
    image = image[:, :, ::-1]
    # Convert to float
    if image.dtype != np.float32:
        image = np.float32(image)
    # Normalize
    image = SCALE * image - OFFSET # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # Set tensor shape
    image = image.transpose((2, 0, 1))
    image = np.reshape(image, (1, 3, CROP, CROP))

    return image

