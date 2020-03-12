import argparse
import sys
from os import path
from time import time

import cv2 as cv
from cvtlib.files import list_files

from dnfal.alignment import FaceMarker, FaceAligner
from dnfal.loggers import logger, config_logger

config_logger(level='DEBUG', to_console=True)

IMAGE_EXT = ('.jpeg', '.jpg', '.png')


def run(image_path: str, weights_path: str):

    face_marker = FaceMarker(weights_path=weights_path)
    face_aligner = FaceAligner()

    if path.isdir(image_path):
        image_paths = list_files(image_path, IMAGE_EXT, recursive=True)
        if len(image_paths) == 0:
            raise ValueError(f'No images found in {image_path}.')
    elif path.isfile(image_path):
        image_paths = [image_path]
    else:
        raise ValueError(f'Input "{image_path}" is not file nor a directory.')

    logger.info('Starting analysis...')
    logger.info('Press "space" key to display next result. Press "q" to quit.')

    images = []

    for image_path in image_paths:
        image_name = path.basename(image_path)
        image = cv.imread(image_path)
        if image is None:
            logger.warn(f'Unable to open image file {image_path}')
            continue
        h, w, = image.shape[0:2]
        logger.info(f'{image_name} loaded. Image size is {w}x{h} pixels.')
        images.append(image)

    n_images = len(images)

    tic = time()
    marks, scores = face_marker.mark(images)
    toc = time()
    logger.info(f'Number of images loaded: {n_images}')
    logger.info(f'Total processing time: {(toc - tic):.4f} s.')

    for i in range(n_images):

        image = images[i]
        mark = marks[i]
        score = scores[i]

        aligned_image, nose_dev = face_aligner.align(image, mark)

        for point in mark:
            cv.circle(image, (int(point[0]), int(point[1])), 2, (0,255,0), -1)

        cv.imshow(f'Face {i} (score {score:.3f})', image)
        title = f'Aligned Face {i} (dev: {nose_dev[0]:.3f}, {nose_dev[1]:.3f})'
        cv.imshow(title, aligned_image)

        ret = cv.waitKey()
        if ret == ord(' '):
            cv.destroyAllWindows()
        elif ret == ord('q'):
            cv.destroyAllWindows()
            break


if __name__ == '__main__':

    curr_dir = path.dirname(path.abspath(__file__))
    parent_dir, _ = path.split(curr_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=False,
        default=path.join(curr_dir, 'data/images/faces-aligned/'),
        help='Path to input image file or directory containing image files.'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=False,
        default=path.join(parent_dir, 'models/weights_face_marker.npy'),
        help='Path to file containing the model weights of face marker.'
    )
    args = parser.parse_args(sys.argv[1:])

    run(args.input, args.weights)


