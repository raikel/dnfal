import argparse
import sys
from datetime import timedelta
from os import path
from time import time
from warnings import warn

import cv2 as cv
from cvtlib.drawing import Drawer


def run(
    video_path: str,
    output_path: str,
    faces_paths: list,
    detector_weights_path: str,
    marker_weights_path: str,
    encoder_weights_path: str
):

    from dnfal.settings import Settings
    from dnfal.vision import FacesVision
    from dnfal.engine import VideoAnalyzer
    from dnfal.mtypes import Face

    settings = Settings()
    settings.detector_weights_path = detector_weights_path
    settings.marker_weights_path = marker_weights_path
    settings.encoder_weights_path = encoder_weights_path
    settings.log_to_console = True
    settings.similarity_thresh = 0.75
    settings.detection_min_score = 0.9
    settings.marking_min_score = 0.9
    settings.video_capture_source = video_path
    settings.video_mode = VideoAnalyzer.MODE_HUNT
    settings.video_detect_interval = 0.5
    settings.video_real_time = False
    settings.store_face_frames = True
    settings.detection_min_height = 24

    faces_vision = FacesVision(settings)

    drawer = Drawer(font_scale=0.8, text_margins=(2, 2, 8, 2))

    hunt_embeddings = []
    hunt_keys = []
    for image_path in faces_paths:
        image = cv.imread(image_path)
        if image is None:
            warn(f'Unable to read image "{image_path}"')
            continue
        faces, embeddings = faces_vision.frame_analyzer.find_faces(image)
        if len(faces):
            image_name, _ = path.splitext(path.basename(image_path))
            hunt_embeddings.extend(embeddings)
            if len(faces) == 1:
                hunt_keys.append(image_name)
            else:
                hunt_keys.extend([
                    f'{image_name} [{i}]' for i in range(len(faces))
                ])
        else:
            warn(f'No face detected in image {image_path}')

    settings.video_hunt_embeddings = hunt_embeddings
    settings.video_hunt_keys = hunt_keys

    capture = faces_vision.video_analyzer.video_capture
    n_frames = capture.duration_frames

    if len(hunt_embeddings):

        def on_frame():
            progress = capture.frame_number / n_frames
            print(f'Progress {(100 * progress):.1f}%')

        def on_subject_updated(face: Face):
            filename = 'frame_' + str(timedelta(seconds=face.timestamp))
            h, w, = face.frame.image.shape[0:2]
            box = (
                int(w * face.box[0]),
                int(h * face.box[1]),
                int(w * face.box[2]),
                int(h * face.box[3]),
            )
            key = face.subject.data['hunt_key']
            drawer.draw_labeled_box(face.frame.image, key, box)
            cv.imwrite(
                path.join(output_path, filename + '.jpg'),
                face.frame.image
            )

        print('Starting video analysis...')
        tic = time()
        faces_vision.video_analyzer.run(
            update_subject_callback=on_subject_updated,
            frame_callback=on_frame
        )
        toc = time()
        print('Done!')

        frames_count = faces_vision.video_analyzer.frames_count
        frame_rate = faces_vision.video_analyzer.frame_rate
        processing_time = faces_vision.video_analyzer.processing_time

        print('\nProcessing stats:')
        print('============================')
        print(f'Total time:         {(toc - tic):.2f} s')
        print(f'Processing time:    {processing_time:.2f} s')
        print(f'Frames count:       {frames_count}')
        print(f'Frame rate:         {frame_rate:.2f} fps')
    else:
        warn(f'Could no get any face embedding from images.')


if __name__ == '__main__':

    curr_dir = path.dirname(path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        type=str,
        help='Path to input video file.'
    )
    parser.add_argument(
        'find',
        type=str,
        help='Path to image file with faces to find.'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default=path.join(curr_dir, 'output/face_hunt'),
        help='Path to output results folder.'
    )
    parser.add_argument(
        '--weights_detector',
        type=str,
        required=False,
        default=path.join(curr_dir, 'models/weights_face_detector.pth'),
        help='Path to file containing the model weights of face detector.'
    )
    parser.add_argument(
        '--weights_marker',
        type=str,
        required=False,
        default=path.join(curr_dir, 'models/weights_face_marker.npy'),
        help='Path to file containing the model weights of face marker.'
    )
    parser.add_argument(
        '--weights_encoder',
        type=str,
        required=False,
        default=path.join(curr_dir, 'models/weights_face_encoder.pth'),
        help='Path to file containing the model weights of face enocoder.'
    )
    args = parser.parse_args(sys.argv[1:])

    run(
        video_path=args.input,
        output_path=args.output,
        faces_paths=[args.find],
        detector_weights_path=args.weights_detector,
        marker_weights_path=args.weights_marker,
        encoder_weights_path=args.weights_encoder,
    )


