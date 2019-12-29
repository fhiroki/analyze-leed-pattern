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


def register_plot_detected_spot(parser):
    from leed.plot_detected_spot import setup_argument_parser, main

    def command(args):
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command)


def register_calc_rprime(parser):
    from leed.calc_rprime import setup_argument_parser, main

    def command(args):
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command)


def register_plot_dinverse(parser):
    from leed.plot_dinverse import setup_argument_parser, main

    def command(args):
        main(args)

    setup_argument_parser(parser)
    parser.set_defaults(handler=command)


def main():
    # top-level command line parser
    parser = argparse.ArgumentParser(prog=PACKAGE_NAME, description='analyze leed pattern')
    parser.add_argument('--version', action='version', version='%(prog)s ' + version)
    subparsers = parser.add_subparsers()

    epilog_plot_detected_spot = '''
    ex) plot-detected-spot --input-image-path images/L16501.tif
        --output-image-path output/images/L16501_detected.tif
    '''
    parser_plot_detected_spot = subparsers.add_parser(
        'plot-detected-spot', help='see `-h`', epilog=epilog_plot_detected_spot)
    register_plot_detected_spot(parser_plot_detected_spot)

    epilog_calc_rprime = '''
    ex) calc-rprime --kind Ag --surface 111 --input-images-dir image/Ag111/
        --input-voltages-path voltages.csv --isplot --output-image_path output/rprime.png
    '''
    parser_calc_rprime = subparsers.add_parser('calc-rprime', help='see `-h`', epilog=epilog_calc_rprime)
    register_calc_rprime(parser_calc_rprime)

    epilog_plot_dinverse = '''
    ex) plot-dinverse --input-images-dir images/Coronene/
        --input-voltages-path voltages.csv --output-image-path output/dinverse.png
    '''
    parser_plot_dinvese = subparsers.add_parser('plot-dinverse', help='see `-h`', epilog=epilog_plot_dinverse)
    register_plot_dinverse(parser_plot_dinvese)

    # to parse command line arguments, and execute processing
    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        # if unknwon subcommand was given, then showing help
        parser.print_help()


if __name__ == "__main__":
    main()
