#!/usr/bin/env python3
import argparse
from cgitb import text
import logging
from multiprocessing.sharedctypes import Value
import os
import glob
import shutil
import textract
import pytesseract

from tqdm import tqdm
from multiprocessing import Pool, TimeoutError
from tempfile import TemporaryDirectory
from pyunpack import Archive, PatoolError
from email.parser import Parser as EmailParser
from email.policy import default
from openpyxl import load_workbook
from xlrd.compdoc import CompDocError
from pytesseract.pytesseract import TesseractError

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate

from deskew import determine_skew


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

img_exts = {".gif", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}     # ".pdf"

parseable_exts = {".csv", ".doc", ".docx", ".eml", ".epub", ".gif", ".jpg", ".jpeg", ".json",
    ".html", ".htm", ".mp3", ".msg", ".odt", ".ogg", ".pdf", ".png", ".pptx", ".ps", ".rtf",
    ".tiff", ".tif", ".txt", ".wav", ".xlsx", ".xls"}


def get_normalized_ext(fn):
    return os.path.splitext(fn)[1].lower()


def is_zipfile(fn):
    return get_normalized_ext(fn) in zip_exts


def is_image(fn):
    return get_normalized_ext(fn) in img_exts


def is_parseable(fn):
    return get_normalized_ext(fn) in parseable_exts


def _prefixed_fn(fn, prefix, output_dir):
    return os.path.join(output_dir, prefix + os.path.basename(fn))


def process_dir(
    input_dir,
    output_dir,
    file_op_lambda,
    num_pool_procs=20,
    pool_timeout=60,
    temp_dir=None,
    prefix="",
    do_not_unzip=False
):
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
                        output_dir,
                        file_op_lambda,
                        num_pool_procs=num_pool_procs,
                        pool_timeout=pool_timeout,
                        prefix=prefix + os.path.basename(zip_fn) + "_",
                        temp_dir=temp_dir,
                        do_not_unzip=do_not_unzip))
    logger.info(f"Processing files in {input_dir} with prefix \"{prefix}\"...")
    with Pool(processes=num_pool_procs) as pool:
        jobs = [
            (pool.apply_async(file_op_lambda, (fn, prefix, output_dir)), fn)
            for fn in all_files if not is_zipfile(fn) and os.path.isfile(fn)]
        for job, non_zip_fn in tqdm(jobs):
            try:
                res = job.get(timeout=pool_timeout)
            except TimeoutError:
                res = "Timeout"
            if res is not None:
                logger.warning(f"{res} during processing {non_zip_fn} with prefix \"{prefix}\"")
                failed_fns.add(f"{res}: {prefix}{non_zip_fn}")
    
    return failed_fns


def deskew(fn_in, fn_out):
    osd = pytesseract.image_to_osd(fn_in, output_type=pytesseract.Output.DICT)
    angle = osd['orientation']
    image = io.imread(fn_in)
    rotated = rotate(image, angle, resize=True) * 255
    io.imsave(fn_out, rotated.astype(np.uint8))


def extract_text(fn):
    if get_normalized_ext(fn) == ".doc":
        try:
            return textract.process(fn, language="rus")
        except textract.exceptions.ShellError:
            fn_new = os.path.splitext(fn)[0] + ".rtf"
            shutil.copy(fn, fn_new)
            return textract.process(fn_new, language="rus")

    if get_normalized_ext(fn) == ".eml":
        with open(fn) as stream:
            parser = EmailParser(policy=default)
            message = parser.parse(stream)
        text_content = []
        for part in message.walk():
            if part.get_content_type().startswith('text/plain'):
                text_content.append(part.get_content())
        return '\n\n'.join(text_content)

    if get_normalized_ext(fn) == ".xlsx":
        cells_text = []
        wb = load_workbook(filename=fn)
        for sheet_name in wb.sheetnames:
            worksheet = wb[sheet_name]
            cells_text += [str(cell_val) for row_vals in worksheet.values for cell_val in row_vals if cell_val is not None]
        wb.close()
        return '\n'.join(cells_text)

    return textract.process(fn, language="rus")


def textify_fn(fn, prefix, output_dir):
    if not is_parseable(fn):
        return None
    try:
        if is_image(fn):
            fn_in_base, fn_in_ext = os.path.splitext(fn)
            fn_out = fn_in_base + "_deskewed" + fn_in_ext
            deskew(fn, fn_out)
            fn = fn_out
        text = extract_text(fn)
        if type(text) is bytes:
            text = text.decode(encoding="utf-8")
        text = text.replace("\n", " ")
        with open(_prefixed_fn(fn, prefix, output_dir), 'w') as f:
            f.write(text)
        return None
    except (AttributeError, UnicodeDecodeError, KeyError, CompDocError, TesseractError):
        return "Bad file"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory to operate on",
    )
    parser.add_argument(
        "--op",
        choices=["list_extensions", "flatten", "flatten_and_textify"],
        type=str,
        default="flatten_and_textify"
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
    parser.add_argument(
        "--num_processes",
        type=int,
        default=20,
        help="Number of parallel processes to use for extraction",
    )
    parser.add_argument(
        "--extraction_timeout",
        type=int,
        default=60,
        help="Number of seconds for extraction timeout",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise Exception("Not a directory: " + args.input_dir)

    if args.op == "list_extensions":
        all_exts = set()
        failed_fns = process_dir(
            args.input_dir,
            args.output_dir,
            lambda fn, _1, _2: all_exts.add(get_normalized_ext(fn)),
            num_pool_procs=args.num_processes,
            pool_timeout=args.extraction_timeout,
            do_not_unzip=args.do_not_unzip)
        print("")
        print("Found extensions:")
        print("=================")
        for ext in sorted(list(all_exts)):
            print(ext)
    elif args.op == "flatten":
        failed_fns = process_dir(
            args.input_dir,
            args.output_dir,
            lambda fn, prefix, output_dir:
                shutil.copy(fn, _prefixed_fn(fn, prefix, output_dir)),
            num_pool_procs=args.num_processes,
            pool_timeout=args.extraction_timeout,
            do_not_unzip=args.do_not_unzip)
    elif args.op == "flatten_and_textify":
        failed_fns = process_dir(
            args.input_dir,
            args.output_dir,
            textify_fn,
            num_pool_procs=args.num_processes,
            pool_timeout=args.extraction_timeout,
            do_not_unzip=args.do_not_unzip)
    else:
        raise ValueError("Unknown operation: " + args.op)

    failed_out_dir = args.output_dir if args.output_dir is not None else '.'
    with open(os.path.join(failed_out_dir, args.failed_fn_name), "w") as failed_f:
        failed_f.writelines([fn + "\n" for fn in sorted(list(failed_fns))])


if __name__ == "__main__":
    main()