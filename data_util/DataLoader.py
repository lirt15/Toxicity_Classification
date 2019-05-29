import tensorflow as tf


def parser(single_record):
    feature_description = {
        'cid': tf.FixedLenFeature([1], tf.int64),
        'target': tf.FixedLenFeature([1], tf.float32),
        'tokens_id': tf.FixedLenFeature([200], tf.int64),
        'mask': tf.FixedLenFeature([200], tf.int64),
    }
    features = tf.parse_single_example(single_record, feature_description)
    return features['cid'],features['target'],features['tokens_id'],features['mask']

def get_dataset(
        dataset_list,
        batch,
        shuffle=True):
    dataset = tf.data.TFRecordDataset(dataset_list)
    dataset = dataset.map(parser)
    dataset = dataset.batch(batch,True)
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.make_initializable_iterator()

    return dataset
