import tensorflow as tf
from bert import modeling
from tensorflow.python.ops import array_ops
import numpy as np

slim = tf.contrib.slim
def build_model(input_ids, is_training,bert_in_use, max_length=200,input_mask=None,token_type_ids=None,batch_size=32):
    """
    {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "type_vocab_size": 2,
    "vocab_size": 30522
    }
    """
    config = modeling.BertConfig(
        attention_probs_dropout_prob = 0.1,
        hidden_act = "gelu",
        hidden_dropout_prob = 0.1,
        hidden_size = 768,
        initializer_range = 0.02,
        intermediate_size = 3072,
        max_position_embeddings = 512,
        num_attention_heads = 12,
        num_hidden_layers = bert_in_use,
        type_vocab_size = 2,
        vocab_size = 30522
    )

    model = modeling.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids, 
            input_mask=input_mask)
    
    #output_layer = model.get_sequence_output()
    return model

def cal_loss(output_layer,gt_labels, is_training, num_classes,input_mask):

    #one_hot = tf.squeeze(one_hot)

    
    with tf.variable_scope("cls_branch"):
        hidden_size = 768
        output_layer=output_layer.get_pooled_output()
        output_weights = tf.get_variable(
            "output_weights", [num_classes, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_classes], initializer=tf.zeros_initializer())        
        if is_training:
        # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        
        #probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(gt_labels, depth=2, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
    """

    with tf.variable_scope("cls_branch"):
        output_layer = output_layer.get_sequence_output()
        lstm_c = tf.nn.rnn_cell.BasicLSTMCell(128,state_is_tuple=False)
        output, state = tf.nn.dynamic_rnn(lstm_c,
                    output_layer,
                    sequence_length=tf.reduce_sum(input_mask,1),
                    dtype=tf.float32)

        output_weights = tf.get_variable(
            "output_weights", [num_classes, 128 * 2],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        logits = tf.matmul(state, output_weights, transpose_b=True)

        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(gt_labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
    """
    return loss,tf.nn.softmax(logits)

def focal_loss(prediction_tensor, target_tensor, alpha=0.05, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, 
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, 
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    target_tensor = tf.one_hot(target_tensor, depth=2)

    softmax_p = tf.nn.softmax(prediction_tensor)
    zeros = array_ops.zeros_like(prediction_tensor, dtype=softmax_p.dtype)
    
    pos_p_sub = array_ops.where(target_tensor > 0.5, 1 - softmax_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > 0.5, zeros, 1 - softmax_p)
    
    
    per_entry_cross_ent = - (1 - alpha) * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(softmax_p, 1e-8, 1.0)) \
                          - alpha * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - softmax_p, 1e-8, 1.0))
    return tf.reduce_mean(per_entry_cross_ent)
