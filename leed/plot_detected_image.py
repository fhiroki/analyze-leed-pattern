import argparse

from leed.detector import detect


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument('--input-image-path', help='input image path', required=True)
    parser.add_argument('--output-image-path', help='output image path')


def main(args):
    detect(args.input_image_path, isplot=True, output_image_path=args.output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_argument_parser(parser)
    args = parser.parse_args()
    main(args)
