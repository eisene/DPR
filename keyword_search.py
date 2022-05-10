import argparse
import logging
import os
import pickle

from tqdm import tqdm
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)

logger.addHandler(consoleHandler)

formatter = logging.Formatter('%(asctime)s  %(levelname)s: %(message)s')
consoleHandler.setFormatter(formatter)


def build_keyword_index(args):
    logger.info("Loading flattened directory...")
    corpus, corpus_fns = [], []
    for fn in tqdm(os.listdir(args.flattened_dir)):
        if fn == args.file_status_name:
            continue
        with open(os.path.join(args.flattened_dir, fn), 'r') as cur_f:
            cur_doc = cur_f.read()
            corpus.append(cur_doc.split(" "))
            corpus_fns.append(fn)

    logger.info("Building BM25-Okapi index...")
    bm25 = BM25Okapi(corpus)

    if args.output_dir is None:
        args.output_dir = args.flattened_dir

    logger.info("Saving index to disk...")
    index_fn = os.path.join(args.output_dir, "bm25okapi.index")
    with open(index_fn, "wb") as f_out:
        pickle.dump(bm25, f_out)
    logger.info(f"Index is saved to {index_fn}")

    logger.info("Saving corpus index to disk...")
    corpus_fn = os.path.join(args.output_dir, "corpus.index")
    with open(corpus_fn, "wb") as f_out:
        pickle.dump(corpus_fns, f_out)
    logger.info(f"Corpus is saved to {corpus_fn}")


def _do_keyword_search(search_text, bm25, corpus_fns, num_results, print_res=True):
    tokenized_query = search_text.split(" ")
    res_fns = bm25.get_top_n(tokenized_query, corpus_fns, n=num_results)

    if print_res:
        print("Results:")
        print("========")
        for fn in res_fns:
            print("    ", fn)

    return res_fns


def keyword_search(args):
    logger.info("Loading BM25-Okapi index...")
    with open(os.path.join(args.index_dir, "bm25okapi.index"), "rb") as f_in:
        bm25 = pickle.load(f_in)

    logger.info("Loading corpus index...")
    with open(os.path.join(args.index_dir, "corpus.index"), "rb") as f_in:
        corpus_fns = pickle.load(f_in)

    logger.info("Searching...")
    if args.interactive:
        while True:
            search_text = input("Enter search text:")
            _do_keyword_search(
                search_text,
                bm25,
                corpus_fns,
                args.num_results,
                print_res=True
            )
    else:
        _do_keyword_search(
            args.search_text,
            bm25,
            corpus_fns,
            args.num_results,
            print_res=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    subparsers = parser.add_subparsers()

    parser_keyword_index = subparsers.add_parser('keyword_index', help='Build keyword search index')
    parser_keyword_index.add_argument(
        "--flattened_dir",
        type=str,
        help="Flattened textified directory to search in",
    )
    parser_keyword_index.add_argument(
        "--file_status_name",
        type=str,
        default="textify_file_status.csv",
        help="Filename of CSV to ignore with status of extracted files",
    )
    parser_keyword_index.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save index",
    )
    parser_keyword_index.set_defaults(func=build_keyword_index)

    parser_keyword_search = subparsers.add_parser('keyword_search', help='Search for text via keywords')
    parser_keyword_search.add_argument(
        "--index_dir",
        type=str,
        default=None,
        help="Directory with saved index files",
    )
    parser_keyword_search.add_argument(
        "--search_text",
        type=str,
        help="Text to search for",
    )
    parser_keyword_search.add_argument(
        "--num_results",
        type=int,
        default=10,
        help="Number of top results to return",
    )
    parser_keyword_search.set_defaults(func=keyword_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
