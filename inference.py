import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from bert import modeling, tokenization
from model_util import build_model

tf.app.flags.DEFINE_integer(
    'num_classes',
    2,
    '# of classes'
    )
tf.app.flags.DEFINE_integer(
    'batch_size',
    256,
    '# of batch_size'
    )
tf.app.flags.DEFINE_string(
    'model_path',
    './cls_model',
    'path to the saved model'
    )
tf.app.flags.DEFINE_integer(
    'max_sentence_len',
    200,
    'max sentence length'
    )
tf.app.flags.DEFINE_integer(
    'bert_layers4use',
    6,
    'number of bert layers used'
)
FLAGS = tf.app.flags.FLAGS

tokenizer = tokenization.FullTokenizer('./bert/uncased_L-24_H-1024_A-16/vocab.txt')

def batch_iter(test_csv, batch_size):
    num_iter = int(np.ceil(len(test_csv) / batch_size))
    ids = test_csv['id']
    txts = test_csv['comment_text']

    for i in range(num_iter):
        cid, cmt, msk = [], [], []

        for id_,cmt_ in zip(ids[i * batch_size : (i+1) * batch_size],txts[i * batch_size : (i+1) * batch_size]):
            tokens = tokenizer.tokenize(cmt_)
            tokens_id =  tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ["[SEP]"])
            mask = [1] * len(tokens_id)

            if len(tokens_id) < FLAGS.max_sentence_len:
                mask += [0] * (FLAGS.max_sentence_len - len(tokens_id))
                tokens_id += [0] * (FLAGS.max_sentence_len - len(tokens_id))
            mask = mask[:FLAGS.max_sentence_len]
            tokens_id = tokens_id[:FLAGS.max_sentence_len]

            cid.append(id_)
            cmt.append(tokens_id)
            msk.append(mask)
            
        yield np.array(cid),np.array(cmt),np.array(msk)



def main(_):

    input_tokens = tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.max_sentence_len])
    input_mask = tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.max_sentence_len])


    features = build_model.build_model(
        input_tokens, 
        bert_in_use=FLAGS.bert_layers4use,
        is_training=False,
        input_mask=input_mask,
        batch_size=FLAGS.batch_size)
    
    probs = tf.constant(np.ones((FLAGS.batch_size, FLAGS.max_sentence_len)))
    probs = tf.cast(probs,tf.uint8)
    loss, labels_pred = build_model.cal_loss(
        features,
        probs,
        is_training=False,
        num_classes=2,
        input_mask=input_mask
    )
    labels_pred = labels_pred[:,1]
    all_vars = tf.global_variables()

    saver_bert = tf.train.Saver(all_vars)
    ckpt = tf.train.latest_checkpoint(FLAGS.model_path)

    with tf.Session() as sess:
        saver_bert.restore(sess,ckpt)

        result_dict = {'id':[],'prediction':[]}

        test_file = pd.read_csv('./dataset/test.csv')
        batches = batch_iter(test_file,FLAGS.batch_size)
        for cid,cmt,msk in tqdm(batches,total=len(test_file)//FLAGS.batch_size + 1):
            #print(cmt.shape, msk.shape)
            result = sess.run([labels_pred],feed_dict = {
                input_tokens:cmt,
                input_mask:msk
            })[0]

            result_dict['id'] += list(cid)
            result_dict['prediction'] += list(result)
        result_df = pd.DataFrame(result_dict,columns=['id','prediction'])
        result_df.to_csv('submission.csv',index=False)
                
if __name__ == "__main__":
    tf.app.run()
