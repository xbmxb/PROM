sp = 'test.source'
tp = 'test.target'
pp = ''

s = []
t = []
p = []

with open(sp, 'r', encoding='utf-8') as f:
    for line in f:
        s.append(line)
with open(tp, 'r', encoding='utf-8') as f:
    for line in f:
        t.append(line)
with open(pp, 'r', encoding='utf-8') as f:
    for line in f:
        p.append(line)
print(len(s))
print(len(t))
print(len(p))
s = s[:len(p)]
t = t[:len(p)]
from transformers import AutoTokenizer

def samengrams(s1, s2, n, tokenizer):
    
    s1_t = tokenizer.tokenize(s1)
    s2_t = tokenizer.tokenize(s2)
    # copy_ngram = [0] * len(s1_t) # which ngrams appeared in tgt, among of src, 0 for no copy, 1 for copy
    ngrams_ = []
    n_windows = []
    for i in range(len(s2_t) - n + 1):
        n_windows.append(s2_t[i:i + n])
    for j in range(len(s1_t) - n + 1):
        ngram = s1_t[j:j + n]
        if ngram in n_windows:
            # copy_ngram[j:j + n] = [1] * n
            if ngram not in ngrams_:
                ngrams_.append(ngram)
    return ngrams_

from tqdm import tqdm
prec_ = []
rec_ = []
f1_ = []
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
for i in tqdm(range(len(p))):
    # if i>0:
    #     break
    s_t = samengrams(s[i], t[i], 5, tokenizer)
    s_p = samengrams(s[i], p[i], 5, tokenizer)
    # print(s[i])
    # print(t[i])
    # print(p[i])
    # print(s_t)
    # print(s_p)
    same = [v for v in s_t if v in s_p]
    # print(same)
    if len(same) > 0:
        prec = len(same) / len(s_p)
        rec = len(same) /len(s_t)
        f1 = 2 * prec * rec / (prec + rec)
    else:
        prec = 0
        rec = 0
        f1 = 0
    prec_.append(prec)
    rec_.append(rec)
    f1_.append(f1)
    # print(f1)

print(sum(prec_)/len(prec_))
print(sum(rec_)/len(rec_))
print(sum(f1_)/len(f1_))