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

def efd_single(args):
    return (compute_extractive_fragment_density(args[0], args[1]), args[0], args[1])

THRES = 3
import jsonlines, json, os
from tqdm import tqdm
from multiprocessing import Pool
# datadir = './output_nat'
# datadir = './output_short'
datadir = './output_lead'
# mini = [ x for x in os.listdir(datadir) if x.startswith('wikiall')]
# mini = [
#     'wiki1_nproc_nat_efd.json',
#     'openweb1_nproc_nat_efd.json',
#     'realnews10_nproc_nat_efd.json',
#     'stories1_nproc_nat_efd.json'
# ]
# mini = [ x for x in os.listdir(datadir) if x.endswith('all_nproc_nat_efd.json')]
mini  = [ x for x in os.listdir(datadir) if x.endswith('4_8_efd.json')]
mini  = [ x for x in os.listdir(datadir) if x.endswith('mini_lead_3.json')]
print(mini)
# sd = './output_nat/wikiall_nat_efd2.json'  -->wikiall
# sd = './output_nat/mini_nat_efd3.json'
# sd = './output_nat/all_nat_efd3.json'
# sd = './output_short/all_short_efd3.json'
sd = './output_lead/mini_lead3_efd3.json'
exs = []
for d in mini:
    print('processing ', d)
    d = os.path.join(datadir, d)
    # sd = d + 'lines'
    # calculate efd
    with open(d, 'r') as f:
        defd = json.load(f)
    print('load ',d)
    mp_args = []
    for data in tqdm(defd):
        src = ' '.join(x.strip() for x in data['src'])
        tgt = ' '.join(x.strip() for x in data['tgt'])
        mp_args.append((src, tgt))

    # def efd_single(args):
    #     return (compute_extractive_fragment_density(args[0], args[1]), args[0], args[1])

    with Pool(8) as p:
            examples_rs = list(tqdm(p.imap(efd_single, mp_args), total=len(mp_args)))
        
    # marks = [10, 5.5, 4.1, 3]
    se = sorted(examples_rs, key= lambda x: x[0], reverse=True)
    idx = 0
    
    for i, it in enumerate(tqdm(se)):
        if it[0] >= THRES:
            ex = {
                'src' : it[1],
                'tgt' : it[2]
            }
            exs.append(ex)
            # with jsonlines.open(sd, 'a') as f:
            #     f.write(ex)
        else:
            print('stop at', i)
            break
print(len(exs))
with open(sd, "a") as f:
    json.dump(exs, f, indent=4)