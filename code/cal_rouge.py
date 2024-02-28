#!/usr/bin/env python

import argparse
import datetime
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from utils import calculate_rouge


logger = getLogger(__name__)

def run_generate(verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")

  
    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    # Compute scores
    score_fn = calculate_rouge
    output_lns = [x.rstrip() for x in open(args.save_path, encoding='utf-8').readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path, encoding='utf-8').readlines()][: len(output_lns)]
    scores: dict = score_fn(output_lns, reference_lns)
    # scores.update(runtime_metrics)

    # if args.dump_args:
    #     scores.update(parsed_args)
    # if args.info:
    #     scores["info"] = args.info

    if verbose:
        print(scores)

    if args.score_path is not None:
        json.dump(scores, open(args.score_path, "w"))

    return scores


if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME $DATA_DIR/test.source $save_dir/test_translations.txt --reference_path $DATA_DIR/test.target --score_path $save_dir/test_bleu.json  --task translation $@
    run_generate(verbose=True)
