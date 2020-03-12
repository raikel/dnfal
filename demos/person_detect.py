import argparse
import sys
from os import path
from time import time

import cv2 as cv
from cvtlib.drawing import Drawer
from cvtlib.image import resize

from utils import list_images, DEMOS_DIR, MODELS_DIR
from dnfal.persons import PersonDetector
from dnfal.loggers import logger, config_logger


def run(image_path: str, weights_path: str):

    config_logger(level='DEBUG', to_console=True)

    person_detector = PersonDetector(
        weights_path=weights_path,
        resize_height=192
    )

    images_paths = list_images(image_path)

    logger.info('Starting analysis...')
    logger.info('Press "space" key to display next result. Press "q" to quit.')

    max_image_size = 1920

    drawer = Drawer()
    drawer.font_scale = 0.5
    drawer.font_linewidth = 1

    for image_path in images_paths:
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
        boxes, scores = person_detector.detect(image)
        toc = time()
        logger.info(f'Found {len(boxes)} persons in {(toc - tic):.3f} s.')

        for ind, box in enumerate(boxes):
            drawer.draw_labeled_box(image, f'{int(100*scores[ind])}%', box)

        cv.imshow(f'Faces in {image_name}', image)

        ret = cv.waitKey()
        if ret == ord(' '):
            cv.destroyAllWindows()
        elif ret == ord('q'):
            cv.destroyAllWindows()
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        required=False,
        default=path.join(DEMOS_DIR, 'data/images/persons'),
        help='Path to input image file or directory containing image files.'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=False,
        default=path.join(MODELS_DIR, 'weights_person_detector.pth'),
        help='Path to file containing the model weights of person detector.'
    )
    args = parser.parse_args(sys.argv[1:])

    run(args.input, args.weights)


