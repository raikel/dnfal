import argparse
import sys
from os import path
from time import time

import cv2 as cv
from cvtlib.drawing import Drawer
import numpy as np

from dnfal.loggers import logger, config_logger
from dnfal.persons import PersonEncoder
from utils import list_images, DEMOS_DIR, MODELS_DIR

SHOW_WIDTH = 32
SHOW_HEIGHT = 96
SHOW_MARGIN = [10, 18]


def resize_grid_item(image: np.ndarray):
    h, w = image.shape[0:2]
    image = cv.resize(image, (int(SHOW_HEIGHT * w / h), SHOW_HEIGHT))
    h, w = image.shape[0:2]
    if w > SHOW_WIDTH:
        image = image[:, 0:SHOW_WIDTH]
    return image


def run(image_path: str, output_path: str, weights_path: str):

    config_logger(level='DEBUG', to_console=True)

    person_encoder = PersonEncoder(weights_path=weights_path)
    images_paths = list_images(image_path)

    logger.info('Starting analysis...')
    logger.info('Press "space" key to display next result. Press "q" to quit.')

    drawer = Drawer()
    drawer.font_scale = 0.5
    drawer.font_linewidth = 1

    images = []

    for image_path in images_paths:
        image_name = path.basename(image_path)
        logger.info(f'Reading image {image_name}...')
        image = cv.imread(image_path)

        if image is None:
            logger.warn(f'Unable to open image file {image_path}')
            continue

        h, w, = image.shape[0:2]
        logger.info(f'Image loaded. Image size is {w}x{h} pixels.')
        images.append(image)

    logger.info(f'Starting encoding images...')
    tic = time()
    embeddings = person_encoder.encode(images)
    toc = time()
    logger.info(f'Encoding {len(images)} images took {(toc - tic):.3f} s.')

    score_matrix = np.matmul(embeddings, embeddings.T)

    logger.info(f'Building matching results image...')

    n_images = len(images)
    grid_width = (n_images + 1) * (SHOW_WIDTH + SHOW_MARGIN[0]) + SHOW_MARGIN[0]
    grid_height = (n_images + 1) * (SHOW_HEIGHT + SHOW_MARGIN[1]) + SHOW_MARGIN[1]
    show_grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    y = SHOW_MARGIN[1]
    for i in range(n_images):
        x = SHOW_MARGIN[0]
        image_row = resize_grid_item(images[i])
        h, w = image_row.shape[0:2]
        show_grid[y:(y + h), x:(x + w)] = image_row
        cv.rectangle(
            img=show_grid,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=(0, 0, 255),
            thickness=1,
            lineType=cv.LINE_AA
        )
        x += SHOW_WIDTH + SHOW_MARGIN[0]

        sort_score_ind = np.argsort(score_matrix[i, :])[::-1]
        row_score = score_matrix[i, sort_score_ind]
        row_images = [images[i] for i in sort_score_ind]

        for j in range(n_images):
            image_col = resize_grid_item(row_images[j])
            h, w = image_col.shape[0:2]
            show_grid[y:(y + h), x:(x + w)] = image_col

            score = row_score[j]
            cv.putText(
                img=show_grid,
                text=f'{int(100 * score)}%',
                org=(x, y + SHOW_HEIGHT + 13),
                color=(255, 255, 255),
                fontFace=cv.FONT_HERSHEY_PLAIN,
                fontScale=0.8,
                lineType=cv.LINE_AA
            )

            x += SHOW_WIDTH + SHOW_MARGIN[0]

        y += SHOW_HEIGHT + SHOW_MARGIN[1]

    logger.info(f'Saving matching results image...')
    cv.imwrite(output_path, show_grid)

    logger.info(f'Showing matching results image...')
    window_name = 'Person matching scoring matrix'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1800, 1000)
    cv.imshow(window_name, show_grid)

    if cv.waitKey() == ord('q'):
        cv.destroyAllWindows()

    logger.info(f'Done !!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        required=False,
        default=path.join(DEMOS_DIR, 'data/images/persons/boxes'),
        help='Path to input image file or directory containing image files.'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default=path.join(DEMOS_DIR, 'output/persons/encoder/match.jpg'),
        help='Path to output results image.'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=False,
        default=path.join(MODELS_DIR, 'weights_person_encoder.pth'),
        help='Path to file containing the model weights of person encoder.'
    )
    args = parser.parse_args(sys.argv[1:])

    run(args.input, args.output, args.weights)


