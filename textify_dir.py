#!/usr/bin/env python3
import argparse
import logging
import os
import glob
import shutil

from tqdm import tqdm
from tempfile import TemporaryDirectory
from pyunpack import Archive, PatoolError


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)

logger.addHandler(consoleHandler)

formatter = logging.Formatter('%(asctime)s  %(levelname)s: %(message)s')
consoleHandler.setFormatter(formatter)

zip_exts = {".7z", ".ace", ".alz", ".a", ".arc", ".arj", ".bz2", ".cab", ".Z", ".cpio", ".deb",
    ".dms", ".gz", ".lrz", ".lha", ".lzh", ".lz", ".lzma", ".lzo", ".rpm", ".rar", ".rz", ".tar",
    ".xz", ".zip", ".jar", ".zoo"}


def get_normalized_ext(fn):
    return os.path.splitext(fn)[1].lower()


def is_zipfile(fn):
    return get_normalized_ext(fn) in zip_exts


def process_dir(input_dir, file_op_lambda, temp_dir=None, prefix="", do_not_unzip=False):
    all_files = glob.glob(os.path.join(input_dir, "**"), recursive=True)

    zip_files = [fn for fn in all_files if is_zipfile(fn)]
    failed_fns = set()
    if len(zip_files) > 0:
        logger.info(f"Processing archives in {input_dir} with prefix \"{prefix}\"...")
        for zip_fn in tqdm(zip_files):
            with TemporaryDirectory(dir=temp_dir) as temp_dir_name:
                try:
                    Archive(zip_fn).extractall(os.path.join(temp_dir_name))
                except PatoolError:
                    logger.warning(f"Failed to extract {zip_fn} with prefix \"{prefix}\"")
                    failed_fns.add("Failed: " + prefix + zip_fn)
                except RuntimeError:
                    logger.warning(f"Encrypted: {zip_fn} with prefix \"{prefix}\"")
                    failed_fns.add("Encrypted: " + prefix + zip_fn)
                failed_fns = failed_fns.union(
                    process_dir(
                        temp_dir_name,
                        file_op_lambda,
                        prefix=prefix + os.path.basename(zip_fn) + "_",
                        temp_dir=temp_dir,
                        do_not_unzip=do_not_unzip))
    logger.info(f"Processing files in {input_dir} with prefix \"{prefix}\"...")
    for non_zip_fn in tqdm([fn for fn in all_files if not is_zipfile(fn) and os.path.isfile(fn)]):
        file_op_lambda(non_zip_fn, prefix)
    
    return failed_fns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory to operate on",
    )
    parser.add_argument(
        "--op",
        choices=["list_extensions", "unzip", "textify", "unzip_and_textify"],
        type=str,
        default="unzip_and_textify"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to output",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Directory to use for creating temporary files in during flattening",
    )
    parser.add_argument(
        "--failed_fn_name",
        type=str,
        default="./textify_extract_failures",
        help="Filename to store failed to extract archive names to",
    )
    parser.add_argument(
        "--do_not_unzip",
        action="store_true",
        help="Do not unzip any zip files in the input_dir",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise Exception("Not a directory: " + args.input_dir)

    with TemporaryDirectory(dir=args.temp_dir) as temp_dir:
        if args.op == "list_extensions":
            all_exts = set()
            failed_fns = process_dir(
                args.input_dir,
                lambda fn, _: all_exts.add(get_normalized_ext(fn)),
                do_not_unzip=args.do_not_unzip)
            print("")
            print("Found extensions:")
            print("=================")
            for ext in sorted(list(all_exts)):
                print(ext)
        elif args.op == "unzip":
            failed_fns = process_dir(
                args.input_dir,
                lambda fn, prefix:
                    shutil.copy(fn, os.path.join(args.output_dir, prefix + os.path.basename(fn))),
                do_not_unzip=args.do_not_unzip)

        with open(args.failed_fn_name, "w") as failed_f:
            failed_f.writelines([fn + "\n" for fn in sorted(list(failed_fns))])


if __name__ == "__main__":
    main()