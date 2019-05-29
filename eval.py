import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from data_util.DataLoader import get_dataset
from bert import modeling
from model_util import build_model
from tqdm import tqdm

import time
from sklearn import metrics
import numpy as np
import pandas as pd

tf.app.flags.DEFINE_integer(
    'num_samples',
    100000,
    '# of samples'
    )    
tf.app.flags.DEFINE_string(
    'record_path',
    './dataset/tf_record/train_dataset_1000000_1100000',
    'path to the saved tfrecord files'
    )
tf.app.flags.DEFINE_integer(
    'num_classes',
    2,
    '# of classes'
    )
tf.app.flags.DEFINE_integer(
    'batch_size',
    200,
    '# of classes'
    )
tf.app.flags.DEFINE_string(
    'save_path',
    './cls_model_high_lr',
    'path to save model'
    )
tf.app.flags.DEFINE_integer(
    'bert_layers4use',
    4,
    'how many transformer blocks to use'
)
FLAGS = tf.app.flags.FLAGS

if not os.path.exists(FLAGS.save_path):
    os.makedirs(FLAGS.save_path)


# From baseline kernel
TOXICITY_COLUMN = 'target'

def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]>0.5
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]>0.5]
    return compute_auc((subgroup_examples[label]>0.5), subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup]>0.5) & (df[label]<=0.5)]
    non_subgroup_positive_examples = df[(df[subgroup]<=0.5) & (df[label]>0.5)]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup]>0.5) & (df[label]>0.5)]
    non_subgroup_negative_examples = df[(df[subgroup]<=0.5) & (df[label]<=0.5)]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]>0.5])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


def main(_):
    #tfrecord_files = [ os.path.join(FLAGS.record_path,tp) for tp in os.listdir(FLAGS.record_path)]
    #tfrecord_files = tfrecord_files[-1:]
    #print(tfrecord_files)
    
    dataset = get_dataset(FLAGS.record_path,batch=FLAGS.batch_size,shuffle=False)

    _, probs, tokens_id,mask = dataset.get_next()
    
    probs = tf.reshape(probs,[FLAGS.batch_size])
    label_ids = tf.cast(probs>= 0.5, tf.uint8)

    #(total_loss, per_example_loss, logits, probabilities) = create_model(
    #    bert_config, False, tokens_id, mask, None, label_ids,
    #    2, False)

    features = build_model.build_model(
        tokens_id, 
        bert_in_use=FLAGS.bert_layers4use,
        is_training=False,
        input_mask=mask,
        batch_size=FLAGS.batch_size)
        
    loss, log_probs = build_model.cal_loss(
        features,
        label_ids,
        is_training=False,
        num_classes=2,
        input_mask=mask
    )
    #pred = tf.argmax(logits,1)
    toxic_prob = log_probs[:,1]

    all_vars = tf.global_variables()

    saver_bert = tf.train.Saver(all_vars)

    test_df = pd.read_csv('./dataset/train.csv')[1000000:1100000]
    print('loaded %d records' % len(test_df))    
    
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(FLAGS.save_path)
            print('latest:',ckpt)
            sess.run(dataset.initializer)
            sess.run(tf.variables_initializer(tf.local_variables()))
            saver_bert.restore(sess,ckpt)

            steps = FLAGS.num_samples // FLAGS.batch_size
            pred_total = []
            for i in tqdm(range(steps),total=steps):
                pred_ = sess.run([toxic_prob])[0]
                pred_total += list(pred_)
                #print(pred_[:,:10])

        # Make sure all comment_text values are strings

        # List all identities
        identity_columns = [
            'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
            'muslim', 'black', 'white', 'psychiatric_or_mental_illness']


        MODEL_NAME = 'bert'

        test_df[MODEL_NAME] = np.array(pred_total)
        bias_metrics_df = compute_bias_metrics_for_model(test_df, identity_columns, MODEL_NAME, 'target')
        result=get_final_metric(bias_metrics_df, calculate_overall_auc(test_df, MODEL_NAME))
        print('result: ', result)
        for _ in tqdm(range(20), total=20):
             time.sleep(60)

if __name__ == "__main__":
    tf.app.run()
