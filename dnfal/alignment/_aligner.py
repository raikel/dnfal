import cv2 as cv
import numpy as np

# Default reference facial points for crop size of 112x112
REFERENCE_POINTS = np.float32([
    [38.29459953, 51.69630051],
    [73.53179932, 51.50139999],
    [56.02519989, 71.73660278],
    [41.54930115, 92.3655014],
    [70.72990036, 92.20410156]
])

REFERENCE_SIZE = 112


class FaceAligner:
    """Face aligner.

    Parameters
    ----------

    out_size : int, optional, (default=112)
        Size of output aligned image.
    """

    def __init__(self, out_size: int = 112):
        scale = out_size / REFERENCE_SIZE
        self._reference: np.ndarray = scale * REFERENCE_POINTS
        self.out_size: int = out_size

    def align(self, image: np.ndarray, marks: np.ndarray):
        """Aligns a face image.

        Parameters
        ----------
        image : array_like
            The image face to be aligned.

        marks : array_like of shape = [5, 2]
            The landmark set of the input image face. Each entry in `marks`
            must contains the (x, y) locations of the face landmarks in the
            following order: (0) left eye center, (1) right eye center, (2)
            nose center, (3) mouth left corner and (4) mouth right corner.

        Returns
        -------
        aligned_image : array_like of shape = [out_size, out_size, 3]
            The resulting aligned face image. Alignment is made by computing
            the optimal limited affine transformation with 4 degrees of freedom
            between the set of input face landmarks and a set of reference
            points that represents a reference front-aligned face.
        nose_deviation : tuple of length = 2
            The nose deviation of the input face image. The nose deviation is a
            tuple where the first and second elements represents the horizontal
            and vertical deviation of nose landmark, respectively For a
            deviation lower than (0.4, 0.3), the face is approximately frontal
            in both planes.
        """

        if marks.dtype != np.float32:
            marks = np.float32(marks)

        tfm, _ = cv.estimateAffinePartial2D(
            marks,
            self._reference.copy(),
            method=cv.LMEDS
        )
        aligned_image = cv.warpAffine(
            image, tfm, (self.out_size, self.out_size)
        )
        aligned_marks = (
            np.matmul(tfm[:, 0:2], marks.T) +
            tfm[:, 2].reshape((-1, 1))
        ).T

        face_center = np.mean(aligned_marks[[0, 1, 3, 4]], axis=0)
        eye_dist = abs(aligned_marks[0][0] - aligned_marks[1][0])
        nose_x, nose_y = aligned_marks[2]
        nose_deviation = (
            abs(face_center[0] - nose_x) / eye_dist,
            abs(face_center[1] - nose_y) / eye_dist,
        )

        return aligned_image, nose_deviation
