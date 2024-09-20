# Run me from console!
import asyncio
import os
import sys
import argparse

from Huehistogram import Huehistogram

SIG_INVALID_ARGS = 1

if __name__ == "__main__":

    def show_help():
        print("Usage:")
        print("  To show color histogram:")
        print("    python huehistogram_cli.py [path/to/image.jpg]")
        print("  To save color histogram:")
        print(
            "    python huehistogram_cli.py -i [path/to/image.jpg] [image_2.jpg] -o [path/to/histograms/]"
        )
        return

    if len(sys.argv) <= 1:
        show_help()
        exit(SIG_INVALID_ARGS)

    parser = argparse.ArgumentParser()
    parser.add_argument("args", nargs="*")
    parser.add_argument("-i", "--input", help="path(s) to input image files", nargs="+")
    parser.add_argument(
        "-o", "--output", help="path(s) to the directory to output histograms"
    )
    parser.add_argument("-d", "--dpi", help="the dpi to output histograms")
    args = parser.parse_args()

    input_files = []
    output_dir = ""
    output_dpi = args.dpi

    if args.input is not None and args.output is not None:
        if args.args:
            print(
                "Extra arguments are supplied! If you intend to use --input and --output options, please remove extra arguments other than -i and -o options."
            )
            print("Use:")
            print(
                "  python huehistogram_cli.py -i [path/to/image.jpg] [image_2.jpg] -o [path/to/histograms/]"
            )
            exit(SIG_INVALID_ARGS)
        input_files = args.input
        output_dir = args.output
    else:
        if args.input is None and args.output is None:
            input_files = sys.argv[
                1:-1
            ]  # 実行するプログラム名と最後を除いて input だと思い込む
            output_dir = sys.argv[-1]  # 最後は output だと思い込む
        else:
            print("Please specify paths to input files and output directory!")
            print("Use:")
            print(
                "  python huehistogram_cli.py -i [path/to/image.jpg] [image_2.jpg] -o [path/to/histograms/]"
            )
            exit(SIG_INVALID_ARGS)

    if not (input_files is not None and output_dir is not None):
        print("Please specify paths to input files and output directory!")
        exit(SIG_INVALID_ARGS)

    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist!")
        exit(SIG_INVALID_ARGS)

    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist!")
            exit(SIG_INVALID_ARGS)

    asyncio.run(
        Huehistogram.generate_hue_histograms(input_files, output_dir, output_dpi)
    )
    exit(0)
