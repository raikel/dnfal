import argparse
import sys
from os import path
from time import time

import cv2 as cv
from cvtlib.drawing import Drawer
from cvtlib.files import list_files
from cvtlib.image import resize

from dnfal.detection import FaceDetector
from dnfal.loggers import logger, config_logger

IMAGE_EXT = ('.jpeg', '.jpg', '.png')


def run(image_path: str, weights_path: str):

    config_logger(level='DEBUG', to_console=True)

    faces_detector = FaceDetector(
        weights_path=weights_path,  min_score=0.9, nms_thresh=0.7
    )

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

    max_image_size = 640

    drawer = Drawer()
    drawer.font_scale = 0.5
    drawer.font_linewidth = 2

    for image_path in image_paths:
        image_name = path.basename(image_path)

        logger.info(f'Analyzing image {image_name}...')

        image = cv.imread(image_path)

        if image is None:
            logger.warn(f'Unable to open image file {image_path}')
            continue

        h, w, = image.shape[0:2]

        logger.info(f'Image loaded. Image size is {w}x{h} pixels.')

        if max(w, h) > max_image_size:
            image, scale = resize(image, max_image_size)
            h, w, = image.shape[0:2]
            logger.info(f'Image resized to {w}x{h} pixels.')

        tic = time()
        boxes, scores = faces_detector.detect(image)
        toc = time()
        logger.info(f'Found {len(boxes)} faces in {int(1000*(toc - tic))} ms.')

        for ind, box in enumerate(boxes):
            drawer.draw_labeled_box(image, f'{scores[ind]:.3f}', box)

        cv.imshow(f'Faces in {image_name}', image)

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
        default=path.join(curr_dir, 'data/images/crowd/'),
        help='Path to input image file or directory containing image files.'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=False,
        default=path.join(parent_dir, 'models/weights_detector.pth'),
        help='Path to file containing the model weights of face detector.'
    )
    args = parser.parse_args(sys.argv[1:])

    run(args.input, args.weights)


