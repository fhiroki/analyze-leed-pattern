import argparse

from leed.detector import detect


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument('--input-image', help='input image path', required=True)
    parser.add_argument('--output-image', help='output image path')


def main(args):
    detect(args.input_image, isplot=True, output_image=args.output_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_argument_parser(parser)
    args = parser.parse_args()
    main(args)
