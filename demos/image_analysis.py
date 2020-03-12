import argparse
import sys
from os import path
from time import time

import cv2 as cv
from cvtlib.drawing import Drawer
from cvtlib.files import list_files
from cvtlib.image import resize

IMAGE_EXT = ('.jpeg', '.jpg', '.png')


def run(
    image_path: str,
    detector_weights_path: str,
    marker_weights_path: str,
    encoder_weights_path: str,
    show_scores: bool = False
):

    from dnfal.vision import FacesVision
    from dnfal.settings import Settings
    from dnfal.loggers import logger

    settings = Settings()
    settings.detector_weights_path = detector_weights_path
    settings.marker_weights_path = marker_weights_path
    settings.encoder_weights_path = encoder_weights_path
    settings.log_to_console = True
    settings.detection_min_height = 24
    settings.detection_min_score = 0.8
    settings.marking_min_score = 0.6

    faces_vision = FacesVision(settings)

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

    max_image_size = 1920

    drawer = Drawer()
    drawer.font_scale = 0.7
    drawer.font_linewidth = 2
    drawer.font_color = (0, 255, 0)

    for image_path in image_paths:
        image_name = path.basename(image_path)
        image = cv.imread(image_path)
        if image is None:
            logger.warn(f'Unable to open image file {image_path}')
            continue
        h, w, = image.shape[0:2]
        logger.info(f'{image_name} loaded. Image size is {w}x{h} pixels.')

        if max(w, h) > max_image_size:
            image, scale = resize(image, max_image_size)
            h, w, = image.shape[0:2]
            logger.info(f'Image resized to {w}x{h} pixels.')

        tic = time()
        faces, _ = faces_vision.frame_analyzer.find_faces(image)
        toc = time()
        logger.info(f'Found {len(faces)} faces in {(toc - tic):.3f} seconds.')
        for ind, face in enumerate(faces):
            face_image = face.image.copy()
            box = (
                int(w * face.box[0]),
                int(h * face.box[1]),
                int(w * face.box[2]),
                int(h * face.box[3]),
            )
            if show_scores:
                s = (int(100 * face.detect_score), int(100 * face.mark_score))
                label = f'Face {ind}, ({s[0]} %, {s[1]} %)'
            else:
                label = f'Face {ind}'

            drawer.draw_labeled_box(image, label, box)

            for m in face.landmarks:
                cv.circle(face_image, (m[0], m[1]), 2, (0, 255, 0), -1)

            nv = face.nose_deviation
            print(
                f'Detected face [{ind}]: {{ score: {face.detect_score}, '
                f'nose deviation: [{nv[0]:.3f}, {nv[1]:.3f}] }}'
            )

            cv.imshow(f'Face "{ind}"', face_image)

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
        default=path.join(curr_dir, 'data/images/pose/'),
        help='Path to input image file or directory containing image files.'
    )
    parser.add_argument(
        '--show_scores',
        type=bool,
        required=False,
        default=False,
        help='Show detection and marking score of each face.'
    )
    parser.add_argument(
        '--weights_detector',
        type=str,
        required=False,
        default=path.join(parent_dir, 'models/weights_face_detector.pth'),
        help='Path to file containing the model weights of face detector.'
    )
    parser.add_argument(
        '--weights_marker',
        type=str,
        required=False,
        default=path.join(parent_dir, 'models/weights_face_marker.npy'),
        help='Path to file containing the model weights of face marker.'
    )
    parser.add_argument(
        '--weights_encoder',
        type=str,
        required=False,
        default=path.join(parent_dir, 'models/weights_face_encoder.pth'),
        help='Path to file containing the model weights of face enocoder.'
    )
    args = parser.parse_args(sys.argv[1:])

    run(
        args.input,
        detector_weights_path=args.weights_detector,
        marker_weights_path=args.weights_marker,
        encoder_weights_path=args.weights_encoder,
        show_scores=args.show_scores
    )


