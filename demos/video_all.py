import argparse
import sys
from os import path

import cv2 as cv
from cvtlib.drawing import Drawer

from dnfal.engine import VideoAnalyzer
from dnfal.mtypes import Subject, Face
from dnfal.settings import Settings
from dnfal.vision import FacesVision


def run(video_path: str, output_path: str, show: bool = True):

    drawer = Drawer()
    drawer.font_scale = 1

    settings = Settings()
    settings.align_max_deviation = (0.4, 0.3)
    settings.detection_min_score = 0.95
    settings.detection_only = False
    settings.detection_min_height = 64
    settings.detection_face_padding = 0
    settings.video_capture_source = video_path
    settings.logging_level = 'debug'
    settings.log_to_console = True
    settings.video_mode = VideoAnalyzer.MODE_ALL
    settings.video_real_time = False
    settings.store_face_frames = True

    faces_vision = FacesVision(settings)

    def on_subject_updated(face: Face):
        subject = face.subject
        filename = f'{subject.key}_{len(subject.faces)}.jpg'
        cv.imwrite(
            path.join(output_path, filename),
            face.image
        )

    def on_frame():
        frame = faces_vision.video_analyzer.frame
        h, w, = frame.shape[0:2]
        avg_roi = faces_vision.video_analyzer.adaptive_roi._avg_roi
        roi_dev = faces_vision.video_analyzer.adaptive_roi._roi_dev
        roi = faces_vision.video_analyzer.adaptive_roi.roi

        if roi is not None:
            box = (
                int(w * roi[0]),
                int(h * roi[1]),
                int(w * roi[2]),
                int(h * roi[3]),
            )
            drawer.box_color = (0, 0, 255)
            drawer.draw_labeled_box(frame, f'Roi, dev = {roi_dev:.3f}', box)
        elif avg_roi is not None:
            box = (
                int(w * avg_roi[0]),
                int(h * avg_roi[1]),
                int(w * avg_roi[2]),
                int(h * avg_roi[3]),
            )
            drawer.box_color = (255, 0, 0)
            drawer.draw_labeled_box(frame, f'Avg Roi, dev = {roi_dev:.3f}', box)

        cv.imshow(f'Video {video_path}', frame)
        ret = cv.waitKey(1)
        if ret == ord('q'):
            cv.destroyAllWindows()
            faces_vision.video_analyzer.stop()

    faces_vision.video_analyzer.run(
        update_subject_callback=on_subject_updated,
        frame_callback=on_frame if show else None
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
        '--output',
        type=str,
        required=False,
        default=path.join(curr_dir, 'output/video_all'),
        help='Path to output results folder.'
    )
    parser.add_argument(
        '--show_video',
        type=bool,
        required=False,
        default=False,
        help='Show video capture.'
    )
    args = parser.parse_args(sys.argv[1:])

    run(video_path=args.input, output_path=args.output, show=args.show_video)


