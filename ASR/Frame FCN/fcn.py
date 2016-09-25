import tensorflow as tf
import numpy as np
import h5py as h5

# training parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 128

# architecture parameters
n_hidden = 1024
n_labels = 39   # 39 phones
n_coeffs = 26
n_context_frames = 11   # 5 + 1 + 5
p = 0.75        # keep rate

def enumerate_context(i, sentence, num_frames):
    r = range(i-num_frames, i+num_frames+1)
    r = [x if x>=0 else 0 for x in r]
    r = [x if x<len(sentence) else len(sentence)-1 for x in r]
    return sentence[r]

def add_context(sentence, num_frames=11):
    # [sentence_length, coefficients] -> [sentence_length, num_frames, coefficients]

    assert num_frames % 2 == 1, "Number of frames must be odd (since left + 1 + right, left = right)"

    if num_frames == 1:
        return sentence

    context_sent = []

    for i in range(0, len(sentence)):
        context_sent.append([context for context in enumerate_context(i, sentence, (num_frames-1)//2)])

    return np.array(context_sent).reshape((-1, num_frames*n_coeffs))

print('Making graph')
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=[None, n_coeffs*n_context_frames])
    y = tf.placeholder(dtype=tf.int64, shape=[None])
    risk_labels = tf.placeholder(dtype=tf.float32, shape=[None])
    is_training = tf.placeholder(tf.bool)

    # nonlinearity
    def gelu_fast(_x):
        return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))
    f = gelu_fast

    W = {}
    b = {}

    with tf.variable_scope("in_sample"):
        W['1'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_context_frames*n_coeffs, n_hidden]), 0)/tf.sqrt(1 + p*0.425))
        W['2'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)/tf.sqrt(0.425/p + p*0.425))
        W['3'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)/tf.sqrt(0.452/p + p*0.425))
        W['logits'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_labels]), 0)/tf.sqrt(0.425/p + 1))
        b['1'] = tf.Variable(tf.zeros([n_hidden]))
        b['2'] = tf.Variable(tf.zeros([n_hidden]))
        b['3'] = tf.Variable(tf.zeros([n_hidden]))
        b['logits'] = tf.Variable(tf.zeros([n_labels]))

        W['bottleneck'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden//2]), 0)/tf.sqrt(0.425/p + 0.425))
        W['decode1'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden//2, n_hidden]), 0)/tf.sqrt(0.425 + p*0.425))
        W['decode2'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)/tf.sqrt(0.425/p + 0.425*p))
        W['reconstruction'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_context_frames*n_coeffs]), 0)/tf.sqrt(0.425/p + 1))
        b['bottleneck'] = tf.Variable(tf.zeros([n_hidden//2]))
        b['decode1'] = tf.Variable(tf.zeros([n_hidden]))
        b['decode2'] = tf.Variable(tf.zeros([n_hidden]))
        b['reconstruction'] = tf.Variable(tf.zeros([n_context_frames*n_coeffs]))

    with tf.variable_scope("out_of_sample"):
        W['residual_to_risk1'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_context_frames*n_coeffs, n_hidden//2]), 0)/tf.sqrt(1 + 0.425))
        W['hidden_to_risk1'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden//2]), 0)/tf.sqrt(0.425/p + 0.425))
        W['logits_to_risk1'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_labels, n_hidden//2]), 0)/tf.sqrt(1 + 0.425))
        W['risk2'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden//2, 128]), 0)/tf.sqrt(0.425 + 0.425))
        W['risk'] = tf.Variable(tf.nn.l2_normalize(tf.random_normal([128, 1]), 0)/tf.sqrt(0.425 + 1))

        b['risk1'] = tf.Variable(tf.zeros([n_hidden//2]))
        b['risk2'] = tf.Variable(tf.zeros([128]))
        b['risk'] = tf.Variable(tf.zeros([1]))

    def feedforward(x):
        h1 = f(tf.matmul(x, W['1']) + b['1'])
        h1 = tf.cond(is_training, lambda: tf.nn.dropout(h1, p), lambda: h1)
        h2 = f(tf.matmul(h1, W['2']) + b['2'])
        h2 = tf.cond(is_training, lambda: tf.nn.dropout(h2, p), lambda: h2)
        h3 = f(tf.matmul(h2, W['3']) + b['3'])
        h3 = tf.cond(is_training, lambda: tf.nn.dropout(h3, p), lambda: h3)
        out = tf.matmul(h3, W['logits']) + b['logits']

        hidden_to_bottleneck = f(tf.matmul(h2, W['bottleneck']) + b['bottleneck'])
        d1 = f(tf.matmul(hidden_to_bottleneck, W['decode1']) + b['decode1'])
        d1 = tf.cond(is_training, lambda: tf.nn.dropout(d1, p), lambda: d1)
        d2 = f(tf.matmul(d1, W['decode2']) + b['decode2'])
        d2 = tf.cond(is_training, lambda: tf.nn.dropout(d2, p), lambda: d2)
        recreation = tf.matmul(d2, W['reconstruction']) + b['reconstruction']

        risk1 = f(tf.matmul(out, W['logits_to_risk1']) +
                  tf.matmul(x - recreation, W['residual_to_risk1']) +
                  tf.matmul(h2, W['hidden_to_risk1']) + b['risk1'])
        risk2 = f(tf.matmul(risk1, W['risk2']) + b['risk2'])
        risk_out = tf.matmul(risk2, W['risk'])

        return out, recreation, tf.squeeze(risk_out)

    logits, reconstruction, risk = feedforward(x)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)) +\
           0.5 * tf.reduce_mean(tf.square(x - reconstruction))# +\
           #1e-4*(tf.nn.l2_loss(W['3']) + tf.nn.l2_loss(W['2']) + tf.nn.l2_loss(W['1'])) 

    lr = tf.constant(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    compute_error = tf.reduce_mean(tf.to_float(tf.not_equal(tf.argmax(logits, 1), y)))

print('Loading Data')
data = h5.File("train.h5")
X_train = data['X'][()]
Y_train = data['y'][()]
train_idxs = data['start_idx'][()]

# get validation set
X_val = X_train[-500:]
Y_val = Y_train[-500:]
val_indxs = train_idxs[-500:]
X_train = X_train[:-500]
Y_train = Y_train[:-500]
train_idxs = train_idxs[:-500]

train_mean = np.mean(X_train, axis=(0,1))
train_std = np.std(X_train, axis=(0,1))
X_train -= train_mean
X_train /= (train_std + 1e-11)

data = h5.File("test.h5")
X_test = data['X'][()] - train_mean
Y_test = data['y'][()]
test_idxs = data['start_idx'][()]
X_test -= train_mean
X_test /= (train_std + 1e-11)
del data
print('Number of training examples', X_train.shape[0])
print('Number of validation examples', X_val.shape[0])
print('Number of testing examples', X_test.shape[0])


with tf.Session(graph=graph) as sess:
    print('Training classification and reconstruction components')
    in_sample_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "in_sample")
    out_of_sample_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "out_of_sample")
    sess.run(tf.initialize_variables(set(tf.all_variables()) - set(out_of_sample_vars)))

    num_batches = X_train.shape[0] // batch_size
    loss_ema = 3.66     # -log(1/39)
    err_ema = 1/39.
    for epoch in range(training_epochs):
        # shuffle data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]
        train_idxs = train_idxs[indices]

        for i in range(num_batches):
            # print(i, 'out of', num_batches)
            offset = i * batch_size
            _bx, mask_x, _by = X_train[offset:offset+batch_size], train_idxs[offset:offset+batch_size], Y_train[offset:offset+batch_size]

            bx, by = [], []
            for i in range(_bx.shape[0]):
                sentence_frames = add_context(_bx[i][mask_x[i]:])
                bx.append(sentence_frames)
                by.append(_by[i][mask_x[i]:])

            bx, by = np.concatenate(bx), np.concatenate(by)

            _, err, l = sess.run([optimizer, compute_error, loss], feed_dict={x: bx, y: by, is_training: True, lr: learning_rate})
            loss_ema = loss_ema * 0.9 + 0.1 * l
            err_ema = err_ema * 0.9 + 0.1 * err

        print('Epoch:', epoch, '|', 'ema of loss for epoch:', loss_ema, 'ema for error (%)', err_ema * 100)

        err_total = 0
        for i in range(X_test.shape[0]//batch_size//2):
            # print(i, 'out of', num_batches)
            offset = i * batch_size
            _bx, mask_x, _by = X_test[offset:offset+batch_size], test_idxs[offset:offset+batch_size], Y_test[offset:offset+batch_size]

            bx, by = [], []
            for i in range(_bx.shape[0]):
                sentence_frames = add_context(_bx[i][mask_x[i]:])
                bx.append(sentence_frames)
                by.append(_by[i][mask_x[i]:])

            bx, by = np.concatenate(bx), np.concatenate(by)

            err = sess.run(compute_error, feed_dict={x: bx, y: by, is_training: False})
            err_total += err

        print('Test error for epoch:', err_total/(X_test.shape[0]//batch_size//2))

    # done training

    # initialize other variables so we can save
    risk_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(risk, risk_labels))
    phase2_vars = list(set(tf.all_variables()) - set(in_sample_vars))
    risk_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(risk_loss, var_list=phase2_vars)
    sess.run(tf.initialize_variables(set(tf.all_variables()) - set(in_sample_vars)))

    saver = tf.train.Saver(max_to_keep=1)
    saver.save(sess, "./fcn.ckpt")

    print('Training risk neuron')
