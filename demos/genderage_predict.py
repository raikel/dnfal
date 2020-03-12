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
    genderage_weights_path: str
):

    from dnfal.genderage import GenderAgePredictor
    from dnfal.detection import FaceDetector
    from dnfal.alignment import FaceMarker, FaceAligner
    from dnfal.engine import FrameAnalyzer
    from dnfal.loggers import logger, config_logger

    config_logger(level='info')

    genderage_predictor = GenderAgePredictor(genderage_weights_path)

    face_detector = FaceDetector(
        weights_path=detector_weights_path,
        min_score=0.9,
        nms_thresh=0.7
    )

    face_marker = FaceMarker(weights_path=marker_weights_path)
    face_aligner = FaceAligner(out_size=256)

    frame_analyzer = FrameAnalyzer(
        detector=face_detector,
        marker=face_marker,
        aligner=face_aligner,
        encoder=None,
        face_padding=0.4,
        store_aligned=True
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

    max_image_size = 1920

    drawer = Drawer()
    drawer.font_scale = 0.5
    drawer.font_linewidth = 1
    drawer.text_margins = (2, 3, 12, 3)

    GENDER_LABELS = {
        GenderAgePredictor.GENDER_MAN: 'Man',
        GenderAgePredictor.GENDER_WOMAN: 'Woman',
    }

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

        image = cv.copyMakeBorder(
            image,
            left=0,
            top=100,
            right=0,
            bottom=0,
            borderType=cv.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        h, w, = image.shape[0:2]

        tic = time()
        faces, _ = frame_analyzer.find_faces(image)
        n_faces = len(faces)
        toc = time()
        logger.info(f'Found {n_faces} faces in {(toc - tic):.3f} seconds.')

        if n_faces:
            tic = time()
            genders_ages = genderage_predictor.predict([
                face.aligned_image for face in faces
            ])
            delay = time() - tic
            logger.info(f'Genders and ages predicted in {delay:.3f} seconds.')

            genders_preds, genders_probs, ages_preds, ages_stds = genders_ages

            for ind, face in enumerate(faces):

                gender_pred = genders_preds[ind]
                gender_prob = genders_probs[ind]
                age_pred = ages_preds[ind]
                age_std = ages_stds[ind]

                gender_label = '{} ({})%'.format(
                    GENDER_LABELS[gender_pred], int(100 * gender_prob)
                )

                age_label = f'{int(age_pred)} years'

                label = f'{gender_label} \n {age_label}'

                box = (
                    int(w * face.box[0]),
                    int(h * face.box[1]),
                    int(w * face.box[2]),
                    int(h * face.box[3]),
                )

                drawer.draw_labeled_box(image, label, box)

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
        default=path.join(curr_dir, 'data/images/'),
        help='Path to input image file or directory containing image files.'
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
        '--weights_genderage',
        type=str,
        required=False,
        default=path.join(parent_dir, 'models/weights_genderage_predictor.pth'),
        help='Path to file containing the model weights.'
    )

    args = parser.parse_args(sys.argv[1:])

    run(
        args.input,
        detector_weights_path=args.weights_detector,
        marker_weights_path=args.weights_marker,
        genderage_weights_path=args.weights_genderage
    )


