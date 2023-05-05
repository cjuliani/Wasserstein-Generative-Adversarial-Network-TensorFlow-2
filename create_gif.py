import os
import glob
import imageio
import argparse

from distutils.util import strtobool


bool_fn = lambda x: bool(strtobool(str(x)))  # callable parser type to convert string argument to boolean
parser = argparse.ArgumentParser()
parser.add_argument("--file_pattern", type=str, default='image*.png',
                    help='Pattern of image name considered for creating the gif file.')
parser.add_argument("--directory", type=str, default="results/",
                    help='Directory path to the images considered for the gif creation.')
ARGS, unknown = parser.parse_known_args()


if __name__ == '__main__':
    # Create a .gif image file from a number of images
    # generated in given result folder.
    anim_file = 'dcgan.gif'  # file name
    out = os.path.join(r"results/", anim_file)  # out path

    # Define the pattern of image names, whose corresponding files
    # will make up the gif image.
    image_patterns = os.path.join(ARGS.directory, ARGS.file_pattern)

    # Process.
    with imageio.get_writer(out, mode='I') as writer:
        filenames = glob.glob(image_patterns)
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
