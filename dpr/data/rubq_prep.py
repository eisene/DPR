#!/usr/bin/env python3
"""
 CLI for prepping the RuBQ 2.0 dataset
"""
import argparse
import logging
import json
import os
import random
from tqdm import tqdm
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)

logger.addHandler(consoleHandler)

formatter = logging.Formatter('%(asctime)s  %(levelname)s: %(message)s')
consoleHandler.setFormatter(formatter)


num_gold_examples = 10
gold_example_set = {}


def load_RuBQ(input_dir: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[int, str]]:
    logger.info("Checking structure of RuBQ 2.0 files...")

    # Dev and test appear to be switched in the original dataset
    dev_file = os.path.join(input_dir, "RuBQ_2.0_test.json")
    test_file = os.path.join(input_dir, "RuBQ_2.0_dev.json")
    paragraph_file = os.path.join(input_dir, "RuBQ_2.0_paragraphs.json")

    if not os.path.isfile(dev_file):
        logger.error("Missing development JSON!")
        return None
    if not os.path.isfile(test_file):
        logger.error("Missing test JSON!")
        return None
    if not os.path.isfile(paragraph_file):
        logger.error("Missing paragraphs JSON!")
        return None
    logger.info("Structure of RuBQ 2.0 files looks good")

    logger.info("Loading paragraphs...")
    with open(paragraph_file, 'r') as p_f:
        paragraphs = json.load(p_f)
    logger.info("Loading training set...")
    with open(dev_file, 'r') as p_f:
        train_rubq = json.load(p_f)
    logger.info("Loading test set...")
    with open(test_file, 'r') as p_f:
        test_rubq = json.load(p_f)
    logger.info("All files loaded")

    logger.info("Pivoting paragraphs by uid...")
    paragraphs = {
        paragraph['uid']: paragraph['text']
        for paragraph in paragraphs
    }

    logger.info("RuBQ 2.0 loaded")
    return train_rubq, test_rubq, paragraphs


def prep_gold_examples(train_rubq: List[Dict[str, object]]):
    global gold_example_set
    logger.info("Preparing gold examples...")
    gold_example_set = set([answer_uid
        for example in tqdm(train_rubq) for answer_uid in example['paragraphs_uids']['with_answer']])


def _parse_RuBQ_file(inputs: List[Dict[str, object]], paragraphs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    res = []
    for example_rubq in tqdm(inputs):
        negative_set = gold_example_set - set(example_rubq['paragraphs_uids']['with_answer'])
        res.append({
            "dataset": "rubq_2.0",
            "question": example_rubq['question_text'],
            "answers": [answer['label'] for answer in example_rubq['answers']],
            "positive_ctxs": [paragraphs[p_uid] for p_uid in example_rubq['paragraphs_uids']['with_answer']],
            "negative_ctxs": [paragraphs[p_uid] for p_uid in random.sample(negative_set, num_gold_examples)],
            "hard_negative_ctxs": [
                paragraphs[p_uid] for p_uid in example_rubq['paragraphs_uids']['all_related']
                if not p_uid in set(example_rubq['paragraphs_uids']['with_answer'])]
        })
    return res


def parse_RuBQ(
    train_rubq: List[Dict[str, object]], test_rubq: List[Dict[str, object]], paragraphs: Dict[int, str]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    logger.info("Parsing training set...")
    train_dpr = _parse_RuBQ_file(train_rubq, paragraphs)
    logger.info("Parsing test set...")
    test_dpr = _parse_RuBQ_file(test_rubq, paragraphs)
    logger.info("Parsing finished")

    return train_dpr, test_dpr


def main():
    global num_gold_examples

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing the input dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to dump the prepared dataset to",
    )
    parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_gold_examples",
        default=10,
        type=int,
        help="Number of so-called gold examples to add to the dataset, see DPR paper",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    num_gold_examples = args.num_gold_examples

    rubq = load_RuBQ(args.input_dir)
    if rubq is None:
        logger.error("Something went wrong during loading of the dataset")
        return
    train_rubq, test_rubq, paragraphs = rubq

    prep_gold_examples(train_rubq)
    train_set, test_set = parse_RuBQ(train_rubq, test_rubq, paragraphs)

    logger.info("Saving training set...")
    with open(os.path.join(args.output_dir, "rubq2-train.json"), 'w', encoding='utf8') as train_f:
        json.dump(train_set, train_f, indent=4, ensure_ascii=False)
    logger.info("Saving test set...")
    with open(os.path.join(args.output_dir, "rubq2-test.json"), 'w', encoding='utf8') as test_f:
        json.dump(test_set, test_f, indent=4, ensure_ascii=False)
    logger.info("Done!")


if __name__ == "__main__":
    main()
