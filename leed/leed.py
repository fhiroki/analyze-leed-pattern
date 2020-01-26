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


def register_detect_spots(parser):
    from leed.detect_spots import setup_argument_parser, main

    def command(args):
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command)


def register_calc_distortion(parser):
    from leed.calc_distortion import setup_argument_parser, main

    def command(args):
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command)


def register_plot_spots(parser):
    from leed.plot_spots import setup_argument_parser, main

    def command(args):
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command)


def main():
    # top-level command line parser
    parser = argparse.ArgumentParser(prog=PACKAGE_NAME, description='analyze leed pattern')
    parser.add_argument('--version', action='version', version='%(prog)s ' + version)
    subparsers = parser.add_subparsers()

    epilog_detect_spots = '''
    ex) detect-spots --input-image-path images/L16501.tif
        --output-image-path output/images/L16501_detected.tif
    '''
    parser_detect_spots = subparsers.add_parser(
        'detect-spots', help='see `-h`', epilog=epilog_detect_spots)
    register_detect_spots(parser_detect_spots)

    epilog_calc_distortion = '''
    ex) calc-distortion --kind Ag --surface 111 --input-images-dir image/Ag111/
        --input-voltages-path voltages.csv --isplot --output-image_path output/distortion.png
    '''
    parser_calc_distortion = subparsers.add_parser('calc-distortion', help='see `-h`', epilog=epilog_calc_distortion)
    register_calc_distortion(parser_calc_distortion)

    epilog_plot_spots = '''
    ex) plot-spots --input-images-dir images/Coronene/
        --input-voltages-path voltages.csv --output-image-path output/spots.png
    '''
    parser_plot_dinvese = subparsers.add_parser('plot-spots', help='see `-h`', epilog=epilog_plot_spots)
    register_plot_spots(parser_plot_dinvese)

    # to parse command line arguments, and execute processing
    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        # if unknwon subcommand was given, then showing help
        parser.print_help()


if __name__ == "__main__":
    main()
