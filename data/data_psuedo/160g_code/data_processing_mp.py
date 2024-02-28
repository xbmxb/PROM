import re, json, jsonlines
from filelock import FileLock
from rouge_score import rouge_scorer, scoring
from typing import Callable, Dict, Iterable, List, Tuple, Union
import os, math
import argparse
from transformers import BartTokenizer
from tqdm import tqdm
from multiprocessing import Pool

CR = 0.25
MAX_LEN_SENT = 16 
# MAX_LEN_SENT = 8
MIN_LEN_SENT = 4
ROUGE_KEYS = ["rouge1", "rouge2", "rougeLsum"]
MAX_LEN_TOK = 1600
MIN_LEN_TOK = 256
MIN_TGT = 30

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
 
def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict

def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    assert NLTK_AVAILABLE, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))

def calculate_rouge(
    pred_lns: List[str],
    tgt_lns: List[str],
    use_stemmer=True,
    rouge_keys=ROUGE_KEYS,
    return_precision_and_recall=False,
    bootstrap_aggregation=False,
    newline_sep=True,
) -> Dict:
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lns, pred_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)

def compute_extractive_fragment(A, S):
    """
    :param A: word list of article
    :param S: word list of summary
    :return: F: a list of word list, each word list is an extractive fragment
    """
    F = []
    i = 0
    j = 0
    while i < len(S):
        f = []
        while j < len(A):
            if S[i] == A[j]:
                i_pie = i
                j_pie = j
                while S[i_pie] == A[j_pie]:
                    i_pie += 1
                    j_pie += 1
                    if i_pie >= len(S) or j_pie >= len(A):
                        break
                if len(f) <= i_pie - i:
                    f = S[i:i_pie]
                j = j_pie
            else:
                j += 1
        i = i + max(len(f), 1)
        j = 1
        if len(f) > 0:
            F.append(f)
    return F
def compute_extractive_fragment_density(A, S):
    # string -> list
    A = A.lower().split(" ")
    S = S.lower().split(" ")
    ext_fragment_list = compute_extractive_fragment(A, S)
    coverage = sum([len(f)**2 for f in ext_fragment_list]) / len(S)
    return coverage


def overlap(s1, s2, metric = 'efd', debug = False, tokenizer = None): # s1 short s2 long
    # print(metric)
    if metric == 'rouge1':
        ol = calculate_rouge([s1], [s2],rouge_keys=['rouge1'])['rouge1'][0].fmeasure
        # print(ol)
        # print(s2)
    elif metric == 'rouges':
        rgs = calculate_rouge([s1], [s2])
        # ol = 0.5 * rgs['rouge2'][0].fmeasure + 0.3 * rgs['rouge1'][0].fmeasure + 0.2 * rgs['rougeLsum'][0].fmeasure
        ol = 2 * rgs['rouge2'][0].fmeasure + rgs['rouge1'][0].fmeasure # from coconet 
    elif metric == 'efd':
        ol = compute_extractive_fragment_density(s2, s1)
    else:
        print('no availabel metric.')
        return 
    if debug:
        # display
        print("-----------------")
        print(metric)
        if tokenizer:
            print("s1 :", tokenizer.convert_tokens_to_string(s1.split(" ")) )
            print("s2 :", tokenizer.convert_tokens_to_string(s2.split(" ")) )
        else:
            print("s1 :", s1.encode() )
            print("s2 :", s2.encode() )
        print("overlap score: ", ol)

    return ol

def create_an_example(eg):
    sentences = eg[0]
    args = eg[1]
    tgt_len = int(math.floor(CR * len(sentences)))
    ols = []
    for isen, sen in enumerate(sentences):
        left_sen = sentences[0 : isen]
        left_sen.extend(sentences[isen+1 : -1])
        left_sen = ' '.join(left_sen)
        # cal the overlap(sen, left_sen)
        ol = overlap(sen, left_sen, debug=args.debugging, metric=args.metric, tokenizer = tokenizer if args.do_token else None)
        ols.append((isen, ol))
    
    ols = sorted(ols, key=lambda x : x[1], reverse=True)
    tgt = []
    # for i in range(tgt_len):
    tgt_tok = 0
    i=0
    while True:
        # print(len(sentences))
        # print(ols[i][0])
        tgt.append((ols[i][0], sentences[ols[i][0]]))
        tgt_tok += len(sentences[ols[i][0]].split())
        i +=1
        if i >= len(ols):
            break
        if tgt_tok > MIN_TGT:
            break
        # src.pop(ols[i][
    tgt_ = sorted(tgt, key = lambda x : x[0], reverse=False)
    tgt = [ x[1] for x in tgt_]
    for i in range(len(tgt)):
        sentences.remove(tgt[i])
    src = sentences
    # delete sentences without overlap
    # for sen in src:
    #     if overlap(sen, ' '.join(tgt), metric=overlap_metric) == 0:
    #         sentences.remove(sen)
    # src_rm = sentences
    ex = {
        'src': src,
        # 'src_rm': src_rm,
        'tgt': tgt
    }
    # path = tgt[:6]
    # save_json_line(ex, args.output_path)
    return ex



def create_examples(args, used_datas, tokenizer):
    process_arguments=[]
    data_dir = args.data_dir
    do_token = args.do_token
    overlap_metric = args.metric
    examples = []
    for data in used_datas:
        fn = os.path.join(data_dir, data)
        print('processing data file {}'.format(fn))
        num = sum([1 for i in open(fn, "r", encoding='utf-8')])
        
        sentences = []
        len_in_tok = 0
        
        with open(fn, 'r', encoding='utf-8') as f:
            # num = sum([1 for i in f])
            for idx, line_ in enumerate(tqdm(f,total=num)):
                if args.debugging and idx > 166:
                    break
                line_len = len(tokenizer.tokenize(line_.strip()))
                if line_ == '\n' or len_in_tok > MAX_LEN_TOK - line_len:
                    if len_in_tok < MIN_LEN_TOK:
                        sentences = []
                        continue
                    if line_ != '\n':
                        line = ' '.join(tokenizer.tokenize(line_.strip())) if do_token else line_.strip()
                        sentences.append(line)
                        len_in_tok+=line_len
                    # process a sample
                    # # print(sentences)

                    process_arguments.append((sentences, args))
                    # print(len_in_tok)
                    #done
                    sentences = []
                    len_in_tok = 0
                else:
                    if line_.strip():
                        line = ' '.join(tokenizer.tokenize(line_.strip())) if do_token else line_.strip() # subword
                        sentences.append(line)
                        len_in_tok += line_len
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
parser.add_argument('--metric', type=str, default='efd')
parser.add_argument('--debugging', action ='store_true')
parser.add_argument('--datas_prefix', type=str, default='stories1')
parser.add_argument('--output_path', type=str, default='./output/from_stories1_nproc_8_efd.json')
parser.add_argument('--nproc', type=int, choices=range(2, 65), default=64, required=True)


args = parser.parse_args()

if os.path.exists(args.output_path):
    os.remove(args.output_path)

datas = os.listdir(args.data_dir)
used = [x  for x in datas if not x.startswith('book')]
used_eg = [ x for x in datas if x.startswith('wiki1')] # for debug
used_datas = [ x for x in datas if x.startswith(args.datas_prefix)]

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large') #if args.do_token else None

examples = create_examples(args, used_datas, tokenizer)

# process_arguments=[]

save_json(examples, args.output_path)

