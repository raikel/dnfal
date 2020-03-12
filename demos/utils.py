from os import path
from cvtlib.files import list_files

DEMOS_DIR = path.dirname(path.abspath(__file__))
_parent_dir, _ = path.split(DEMOS_DIR)
MODELS_DIR = path.join(_parent_dir, 'models')

IMAGE_EXT = ('.jpeg', '.jpg', '.png')


def list_images(root_path):

    if path.isdir(root_path):
        image_paths = list_files(root_path, IMAGE_EXT, recursive=True)
        if len(image_paths) == 0:
            raise ValueError(f'No images found in {root_path}.')
    elif path.isfile(root_path):
        image_paths = [root_path]
    else:
        raise ValueError(f'Input "{root_path}" is not file nor a directory.')

    return image_paths
