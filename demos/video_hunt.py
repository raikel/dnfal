import sys
import argparse
from os import path
from datetime import datetime

import cv2 as cv
import numpy as np
from cvtlib.drawing import Drawer


def run(video_path: str, output_path: str, faces_paths: list):

    from dnfal.settings import Settings
    from dnfal.vision import FacesVision
    from dnfal.engine import VideoAnalyzer
    from dnfal.mtypes import Face

    drawer = Drawer(font_scale=1)

    settings = Settings()
    settings.video_capture_source = video_path
    settings.log_to_console = True
    settings.video_mode = VideoAnalyzer.MODE_HUNT
    settings.video_real_time = False
    settings.store_face_frames = True

    faces_vision = FacesVision(settings)

    hunt_embeddings = []
    for face_path in faces_paths:
        image = cv.imread(face_path)
        faces, embeddings = faces_vision.frame_analyzer.find_faces(image)
        hunt_embeddings.extend(embeddings)

    faces_vision.video_analyzer.hunt_embeddings = np.array(hunt_embeddings, np.float32)
    settings.video_hunt_embeddings = np.array(hunt_embeddings, np.float32)

    def on_subject_updated(face: Face):
        filename = str(datetime.fromtimestamp(face.timestamp))
        h, w, = face.frame.image.shape[0:2]
        box = (
            int(w * face.box[0]),
            int(h * face.box[1]),
            int(w * face.box[2]),
            int(h * face.box[3]),
        )
        drawer.draw_labeled_box(face.frame.image, 'Found', box)
        cv.imwrite(
            path.join(output_path, filename + '.jpg'),
            face.frame
        )

    faces_vision.video_analyzer.run(
        update_subject_callback=on_subject_updated
    )


if __name__ == '__main__':

    curr_dir = path.dirname(path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=False,
        default=path.join(curr_dir, 'data/video/people-walking-1.mp4'),
        help='Path to input video file.'
    )
    parser.add_argument(
        '--find',
        type=str,
        required=False,
        default=path.join(curr_dir, 'data/images/subject-1.jpg'),
        help='Path to image file with faces to find.'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default=path.join(curr_dir, 'output'),
        help='Path to output results folder.'
    )
    args = parser.parse_args(sys.argv[1:])

    run(video_path=args.input, output_path=args.output, faces_paths=[args.find])


