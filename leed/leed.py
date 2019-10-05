import argparse
import ast
import os
import re
import sys

# package name
PACKAGE_NAME = 'leed'
with open(os.path.join(os.path.dirname(__file__), '__init__.py')) as f:
    match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))

# adding current dir to lib path
mydir = os.path.dirname(__file__)
sys.path.insert(0, mydir)


def register_plot_detected_image(parser):
    from leed.plot_detected_image import setup_argument_parser, main

    def command(args):
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command)


def register_detect_base_blob(parser):
    from leed.detect_base_blob import setup_argument_parser, main

    def command(args):
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command)


def main():
    # top-level command line parser
    parser = argparse.ArgumentParser(prog=PACKAGE_NAME, description='analyze leed pattern')
    parser.add_argument('--version', action='version', version='%(prog)s ' + version)
    subparsers = parser.add_subparsers()

    epilog_plot_detected_image = '''
    ex) plot-detected-image --input-image data/Coronene_Ag111/image/Coronene/L16501.tif --output-image output/images
    '''
    parser_plot_detected_image = subparsers.add_parser(
        'plot-detected-image', help='see `-h`', epilog=epilog_plot_detected_image)
    register_plot_detected_image(parser_plot_detected_image)

    epilog_detect_base_blob = '''
    ex) detect-base-blob --kind Ag --surface 111
     --voltages 80.6,94.7,109.2,122.9,136.0,150.9,159.1,179.2,193.7,215.3,230.3,252.7,264.9
     --input-dir data/Coronene_Ag111/image/Ag111/ --isplot --output-dir output/r
    '''
    parser_detect_base_blob = subparsers.add_parser('detect-base-blob', help='see `-h`', epilog=epilog_detect_base_blob)
    register_detect_base_blob(parser_detect_base_blob)

    # to parse command line arguments, and execute processing
    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        # if unknwon subcommand was given, then showing help
        parser.print_help()


if __name__ == "__main__":
    main()
