import sys
sys.path.append('../')

from bert import tokenization
from tqdm import tqdm

import pandas as pd

tokenizer = tokenization.FullTokenizer('../bert/uncased_L-12_H-768_A-12/vocab.txt')

with open('../dataset/train.txt','wt',encoding='utf-8') as train_set_after_filter:
  with open('../dataset/test.txt','wt',encoding='utf-8') as test_set_after_filter:
    train_f = pd.read_csv('../dataset/train.csv')
    test_f = pd.read_csv('../dataset/test.csv')

    for cid, tgt, cmt in zip(train_f['id'],train_f['target'],train_f['comment_text']):
        words = tokenizer.tokenize(cmt)
        words = ["[CLS]"]+words+["[SEP]"]
        ids = tokenizer.convert_tokens_to_ids(words)
        ids = ','.join(map(str,ids))
        tgt = int(float(tgt) >= 0.5 )
        new_line = '{},{},{}\n'.format(cid,tgt,ids)
        train_set_after_filter.write(new_line)
    for cid, cmt in zip(test_f['id'],test_f['comment_text']):
        words = tokenizer.tokenize(cmt)
        words = ["[CLS]"]+words+["[SEP]"]
        ids = tokenizer.convert_tokens_to_ids(words)
        ids = ','.join(map(str,ids))
        new_line = '{},{}\n'.format(cid,ids)
        test_set_after_filter.write(new_line)  

"""
    8
    104
    109
    114
    119
    124
    130
    136
    144
    152
    163
    174
    184
    194
    206
    219
    237
    259
    289
    322
"""