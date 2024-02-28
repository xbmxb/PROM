import re, json, jsonlines
from filelock import FileLock
from rouge_score import rouge_scorer, scoring
from typing import Callable, Dict, Iterable, List, Tuple, Union
import os, math
import argparse
from transformers import BartTokenizer
from tqdm import tqdm
from multiprocessing import Pool
import codecs, sys
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

CR = 3
# MAX_LEN_SENT = 16 
MAX_LEN_SENT = 8
MIN_LEN_SENT = 4
ROUGE_KEYS = ["rouge1", "rouge2", "rougeLsum"]

try:
    import nltk

    NLTK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLTK_AVAILABLE = False

if NLTK_AVAILABLE:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)

def save_json_line(eg, path):
    with jsonlines.open(path, 'a') as f:
        f.write(eg)

def create_an_example(eg):
    sentences = eg[0]
    args = eg[1]
    # tgt_len = int(math.floor(CR * len(sentences)))
    tgt_len = CR
    tgt = sentences[:tgt_len]
    src = sentences[tgt_len:]
    ex = {
        'src': src,
        # 'src_rm': src_rm,
        'tgt': tgt
    }
    # print(sentences)
    # print(src)
    # print(tgt)
    return ex



def create_examples(args, used_datas, tokenizer):
    process_arguments=[]
    data_dir = args.data_dir
    do_token = args.do_token
    # overlap_metric = args.metric
    examples = []
    for data in used_datas:
        fn = os.path.join(data_dir, data)
        print('processing data file {}'.format(fn))
        num = sum([1 for i in open(fn, "r", encoding='utf-8')])
        
        sentences = []
        
        with open(fn, 'r', encoding='utf-8') as f:
            # num = sum([1 for i in f])
            for idx, line_ in enumerate(tqdm(f,total=num)):
                if args.debugging and idx > 20:
                    break
                if line_ == '\n': # or len(sentences) >= MAX_LEN_SENT-1:
                    if len(sentences) < MIN_LEN_SENT:
                        sentences = []
                        continue
                    # if line_ != '\n':
                    #     line = ' '.join(tokenizer.tokenize(line_.strip())) if tokenizer else line_
                    #     sentences.append(line)
                    # process a sample
                    # # print(sentences)

                    process_arguments.append((sentences, args))

                    #done
                    sentences = []
                else:
                    line = ' '.join(tokenizer.tokenize(line_.strip())) if tokenizer else line_ # subword
                    sentences.append(line)
                # print(line)
                # print(line_)
                # break
        print("done with {}, paragraph number: {}".format(fn, len(process_arguments)))

    with Pool(args.nproc) as p:
        examples = list(tqdm(p.imap(create_an_example, process_arguments), total=len(process_arguments)))


    return examples


# data_dir = './clean/'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./clean/')
parser.add_argument('--do_token', action='store_true')
# parser.add_argument('--metric', type=str, default='efd')
parser.add_argument('--debugging', action ='store_true')
parser.add_argument('--datas_prefix', type=str, default='stories1')
parser.add_argument('--output_path', type=str, default='./output/from_stories1_nproc_8_efd.json')
parser.add_argument('--nproc', type=int, choices=range(2, 100), default=64, required=True)


args = parser.parse_args()

if os.path.exists(args.output_path):
    os.remove(args.output_path)

datas = os.listdir(args.data_dir)
used = [x  for x in datas if not x.startswith('book')]
used_eg = [ x for x in datas if x.startswith('wiki1')] # for debug
# used_datas = [ x for x in datas if x.startswith(args.datas_prefix)]
used_datas = ['wiki1.txt', 'stories1.txt', 'realnews10.txt', 'openweb1.txt']

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large') if args.do_token else None

examples = create_examples(args, used_datas, tokenizer)

# process_arguments=[]

save_json(examples, args.output_path)
