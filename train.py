import tensorflow as tf
import os
import numpy as np
from data_util.DataLoader import get_dataset
from bert import modeling,optimization
from model_util import build_model
from tqdm import tqdm

tf.app.flags.DEFINE_string(
    'record_path',
    './dataset/tf_record/train_dataset_0_1000000',
    'path to the saved tfrecord files'
    )
tf.app.flags.DEFINE_integer(
    'num_classes',
    2,
    '# of classes'
    )
tf.app.flags.DEFINE_integer(
    'steps',
    60000,
    '# of classes'
) 
tf.app.flags.DEFINE_integer(
    'batch_size',
    64,
    '# of classes'
    )
tf.app.flags.DEFINE_integer(
    'seq_len',
    200,
    'len(sentence)'
    )    
tf.app.flags.DEFINE_float(
    'learning_rate',
    5e-5,
    'lr'
    )
tf.app.flags.DEFINE_string(
    'pretrain_model',
    './bert/uncased_L-12_H-768_A-12/bert_model.ckpt',
    'path to the pretrain_model'
    )
tf.app.flags.DEFINE_string(
    'save_path',
    './cls_model_high_lr',
    'path to save model'
    )
tf.app.flags.DEFINE_integer(
    'bert_layers4use',
    4,
    'how many transformer blocks to use.'
)
FLAGS = tf.app.flags.FLAGS

if not os.path.exists(FLAGS.save_path):
    os.makedirs(FLAGS.save_path)

def main(_):
    #tfrecord_files = [ os.path.join(FLAGS.record_path,tp) for tp in os.listdir(FLAGS.record_path)]
    #tfrecord_files = tfrecord_files[:-2]

    dataset = get_dataset(FLAGS.record_path,batch=FLAGS.batch_size)
    
    cid, probs, tokens_id,mask = dataset.get_next()

    probs = tf.reshape(probs,[FLAGS.batch_size])
    probs = tf.cast(probs, tf.uint8)

    model = build_model.build_model(
        tokens_id,
        bert_in_use=FLAGS.bert_layers4use, 
        is_training=True,
        input_mask=mask,
        batch_size=FLAGS.batch_size)
        
    loss, log_probs = build_model.cal_loss(
        model,
        probs,
        is_training=True,
        num_classes=2,
        input_mask=mask
    )

    tf.summary.scalar('step_loss', loss)

    """
    bert/encoder/layer_k
    """

    with tf.variable_scope('training'):
        global_step = tf.Variable(0, name='global_step')

        #opt = optimization.create_optimizer(
        #    loss = loss, 
        #    init_lr = FLAGS.learning_rate, 
        #    num_train_steps=FLAGS.steps,
        #    num_warmup_steps=3000, 
        #    use_tpu=False,
        #)
        
        global_step = tf.Variable(0, name='global_step')
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate,
            global_step,
            decay_rate = 0.5,
            decay_steps= 10000,
            staircase=False
        )        
        opt = tf.train.RMSPropOptimizer(
            learning_rate,
        )
        glb_vars = tf.global_variables()
        cls_vars = [v for v in glb_vars if v.name.startswith('cls_branch')] 
        
        bert_last=[]
        for last in range(4):
            for v in glb_vars:
                if v.name.startswith('bert/encoder/layer_{}'.format(FLAGS.bert_layers4use-last)):
                    bert_last.append(v)

        trainable_vars = cls_vars + bert_last
        for tp in trainable_vars:
            print(tp)
            
        train_step = opt.minimize(loss, global_step=global_step, var_list=trainable_vars)
        merged_summary = tf.summary.merge_all()

    all_vars = tf.global_variables()
    bert_vars = [v for v in all_vars if v.name.startswith('bert')]
    not_bert_vars = [v for v in all_vars if not v.name.startswith('bert')]

    saver_bert = tf.train.Saver(bert_vars)
    saver_all = tf.train.Saver([v for v in all_vars if not v.name.startswith('training')])
    
    #ckpt = tf.train.latest_checkpoint(FLAGS.pretrain_model)
    
    step_loss= 0.
    cum_loss = 0.
    with tf.Session() as sess:
        sess.run(dataset.initializer)
        saver_bert.restore(sess,FLAGS.pretrain_model)
        sess.run(tf.variables_initializer(not_bert_vars))

        writer = tf.summary.FileWriter(
            FLAGS.save_path, tf.get_default_graph(), flush_secs=15)

        steps = int(FLAGS.steps)
        
        for i in tqdm(range(steps),total=steps):
            _,loss_, smry, step_now,= sess.run([train_step,loss,merged_summary,global_step])

            step_loss += loss_
            if i % 200 == 0 and i != 0:
                writer.add_summary(smry, step_now)
                step_loss /= 200.
                if i == 200: cum_loss = step_loss
                cum_loss = cum_loss * 0.9 + 0.1 * step_loss

                print("step {0} loss {1}\ncum  {0} loss {2}".format(i, step_loss , cum_loss))
                step_loss = 0.
            if i % 2000 == 0:
                tf.train.write_graph(sess.graph_def, FLAGS.save_path,
                                        'graph.pb')

                saver_all.save(
                    sess,
                    os.path.join(FLAGS.save_path, "cls_bert"),
                    global_step=step_now)


if __name__ == "__main__":
    tf.app.run()
