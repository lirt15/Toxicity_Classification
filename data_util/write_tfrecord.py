import tensorflow as tf
import sys
import os

from collections import Iterable
from tqdm import tqdm
start = sys.argv[1]
end = sys.argv[2]

tf.app.flags.DEFINE_string(
    'source_txt','../dataset/train.txt','path to the comments after filter'
)
tf.app.flags.DEFINE_string(
    'tfrecord_path','../dataset/tf_record','path to save tfrecord files'
)
tf.app.flags.DEFINE_integer(
    'max_sentence_len',200,'the max length of tokens for each comment'
)
FLAGS = tf.flags.FLAGS

if not os.path.exists(FLAGS.tfrecord_path):
    os.makedirs(FLAGS.tfrecord_path)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    if isinstance(value,Iterable):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))    
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if isinstance(value,Iterable):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def instance_to_example(line):
    line = line.strip().split(',')
    cid, target, tokens_id = int(line[0]), float(line[1]), list(map(int,line[2:]))
    mask = [1] * len(tokens_id)
    if len(tokens_id) < FLAGS.max_sentence_len:
        mask += [0] * (FLAGS.max_sentence_len - len(tokens_id))
        tokens_id += [0] * (FLAGS.max_sentence_len - len(tokens_id))
    mask = mask[:FLAGS.max_sentence_len]
    tokens_id = tokens_id[:FLAGS.max_sentence_len]
    feature = {
        'cid':_int64_feature(cid),
        'target':_float_feature(target),
        'tokens_id':_int64_feature(tokens_id),
        'mask':_int64_feature(mask)
    }
    return tf.train.Example(
        features = tf.train.Features(feature=feature)
    ).SerializeToString()

def main(_):
    with open(FLAGS.source_txt) as dataset_txt:
        txt_piece = dataset_txt.readlines()[int(start):int(end)]
        with tf.python_io.TFRecordWriter(
            os.path.join(
                FLAGS.tfrecord_path, 'train_dataset_{}_{}'.format(start,end)
        )) as writer:
            for line in tqdm(txt_piece,total=len(txt_piece)):
                example = instance_to_example(line)
                writer.write(example)


if __name__ == "__main__":
    tf.app.run()
