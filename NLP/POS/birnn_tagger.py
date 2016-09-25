from glob import glob
from reader import Reader

import numpy as np
import tensorflow as tf
import time

logging = tf.logging

# tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability.")
tf.flags.DEFINE_float("init_scale", 0.1, "Initialization scale.")
tf.flags.DEFINE_float("learning_rate", 1.0, "Initial LR.")
tf.flags.DEFINE_float("lr_decay", 0.5, "LR decay.")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Maximum gradient norm.")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size.")
tf.flags.DEFINE_integer("hidden_size", 128,
                        "Dimensionality of character embedding and lstm hidden size.")
tf.flags.DEFINE_integer("max_epoch", 4, "Max number of training epochs before LR decay.")
tf.flags.DEFINE_integer("max_max_epoch", 13, "Stop after max_max_epoch epochs.")
tf.flags.DEFINE_integer("num_layers", 3, "Number of stacked RNN layers.")

reader = Reader(split=0.9)
(X_train, Y_train, mask_train,
 X_test, Y_test, mask_test) = \
    reader.get_data(glob('/scratch/Experiments/Risk Neurons/POS/bi_tagger/neural/data/WSJ/*/*.POS'))
print('len(X_train)', len(X_train), 'len(X_test)', len(X_test))
print('reader.ignore_ids', reader.ignore_ids)
print('len(reader.word_to_id)', len(reader.word_to_id),
      'len(reader.tag_to_id)', len(reader.tag_to_id))

FLAGS = tf.flags.FLAGS
graph = tf.Graph()
with graph.as_default():
    batch_size = FLAGS.batch_size
    hidden_size = FLAGS.hidden_size
    num_layers = FLAGS.num_layers
    # dropout_keep_prob = FLAGS.dropout_keep_prob
    vocab_size = len(reader.word_to_id)
    tag_size = len(reader.tag_to_id)
    maxlen = reader.maxlen

    input_data = tf.placeholder(tf.int64, [None, maxlen])
    targets = tf.placeholder(tf.int64, [None, maxlen])
    mask = tf.placeholder(tf.bool, [None, maxlen])

    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
    # if is_training and dropout_keep_prob < 1:
    #     lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
    #         lstm_cell, output_keep_prob=dropout_keep_prob)

    cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    initial_state_fw = cell_fw.zero_state(tf.shape(targets)[0], tf.float32)
    initial_state_bw = cell_bw.zero_state(tf.shape(targets)[0], tf.float32)

    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size,
                                                  hidden_size])
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    inputs = [input_ for input_ in tf.unpack(tf.transpose(inputs, [1, 0, 2]))]
    # if is_training and dropout_keep_prob < 1:
    #     inputs = tf.nn.dropout(tf.pack(inputs), dropout_keep_prob)
    #     inputs = tf.unpack(inputs)
    outputs, _, _ = tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs,
                                            initial_state_fw=initial_state_fw,
                                            initial_state_bw=initial_state_bw)

    # output from forward and backward cells.
    output = tf.reshape(tf.concat(1, outputs), [-1, 2 * hidden_size])
    softmax_w = tf.get_variable("softmax_w", [2 * hidden_size, tag_size])
    softmax_b = tf.get_variable("softmax_b", [tag_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(targets, [-1])],
        [tf.reshape(tf.cast(mask, tf.float32), [-1])], tag_size)
    cost = tf.reduce_sum(loss) / batch_size

    equality = tf.equal(tf.argmax(logits, 1),
                        tf.cast(tf.reshape(targets, [-1]), tf.int64))
    masked = tf.boolean_mask(equality, tf.reshape(mask, [-1]))
    misclass = 1 - tf.reduce_mean(tf.cast(masked, tf.float32))

    lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      FLAGS.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

def assign_lr(session, lr_value):
    session.run(tf.assign(lr, lr_value))


def run_epoch(x_data, y_data, data_mask, eval_op, training=True, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(x_data) // batch_size) - 1)
    start_time = time.time()
    costs = 0.0
    iters = 0
    misclass_ = []
    for step, (x, y, data_mask) in enumerate(Reader.iterator(x_data, y_data, data_mask, batch_size)):
        if training is True:
            l, misclassifications, _ = session.run([cost, misclass, eval_op],
                                                   {input_data: x, targets: y, mask: data_mask})
        else:
            l, misclassifications = session.run([cost, misclass],
                                                {input_data: x, targets: y, mask: data_mask})
        costs += l
        iters += batch_size

        if verbose and step % (epoch_size // 10) == 0:
            print("[%s] %.3f perplexity: %.3f misclass:%.3f speed: %.0f wps" %
                  ('train' if training else 'test', step * 1.0 / epoch_size,
                   np.exp(costs / iters), misclassifications,
                   iters * batch_size / (time.time() - start_time)))
        misclass_.append(misclassifications)
    return np.exp(costs / iters), np.mean(misclass_)

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value))
print("")

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    saver = tf.train.Saver()
    best_misclass = 1.0

    for i in range(FLAGS.max_max_epoch):
        lr_decay = FLAGS.lr_decay ** max(i - FLAGS.max_epoch, 0.0)
        assign_lr(session, FLAGS.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(lr)))
        train_perplexity, _ = run_epoch(X_train, Y_train, mask_train,
                                        train_op, verbose=True)
        _, misclassifications = run_epoch(X_test, Y_test, mask_test,
                                tf.no_op(), training=False, verbose=True)
        if misclassifications < best_misclass:
            best_misclass = misclassifications
            saver.save(session, './data/bid3rnn_tagger.ckpt', global_step=i)
            print('Saving')
