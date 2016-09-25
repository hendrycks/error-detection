'''
Based on Jon Rein's code
'''

from tensorflow.python.ops import ctc_ops as ctc
# from tensorflow.contrib.ctc import ctc_ops as ctc   # deprecated in future
import tensorflow as tf
import numpy as np
from utils import load_batched_data, target_list_to_sparse_tensor
import pickle

####Learning Parameters
nEpochs = 60
batchSize = 100

####Network Parameters
nFeatures = 39      # MFCC coefficients, energy, delta, delta delta
nHidden = 256
nClasses = 40       # 40 because of 39 phones, plus the "blank" for CTC

####Load data
print('Loading data')
data = pickle.load(open("TIMIT_data_prepared_for_CTC_clean.pkl", 'rb'), encoding='latin1')

# 6300 x 776 x 39

# we will take 5000 examples from the 6300
data_list = []
for i in range(5000//batchSize):
    offset = batchSize * i
    target_list = []
    for j in range(batchSize):
        target_list.append(data['y_phones'][offset+j])
    data_list.append(
        (data['x'][offset:offset+batchSize,:,:],
         target_list_to_sparse_tensor(target_list),
         data['mask'][offset:offset+batchSize]))

del data

batchedData, maxTimeSteps, totalN = data_list, 776, 50

# def gelu_fast(x):
#     return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def clipped_gelu(x):
    return tf.minimum(0.5 * x * (1 + tf.tanh(x)), 6)

####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():

    ####NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow

    ####Graph input
    inputX = tf.placeholder(tf.float32, shape=(batchSize, maxTimeSteps, nFeatures)) + tf.random_normal(shape=(batchSize, maxTimeSteps, nFeatures), stddev=0.05)

    #Prep input data to fit requirements of rnn.bidirectional_rnn
    #  Reshape to 2-D tensor (nTimeSteps*batchSize, nfeatures)
    inputXrs = tf.reshape(tf.transpose(inputX, [1, 0, 2]), [-1, nFeatures])
    #  Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
    inputList = tf.split(0, maxTimeSteps, inputXrs)
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(batchSize))
    # print(inputX, targetIxs, targetVals, targetShape, seqLengths)

    ####Weights & biases
    weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    biasesOutH1 = tf.Variable(tf.zeros([nHidden]))
    weightsOutH2 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    biasesOutH2 = tf.Variable(tf.zeros([nHidden]))
    weightsClasses = tf.Variable(tf.truncated_normal([nHidden, nClasses],
                                                     stddev=np.sqrt(2.0 / nHidden)))
    biasesClasses = tf.Variable(tf.zeros([nClasses]))

    ####Network
    lstm_cell = tf.nn.rnn_cell.LSTMCell(nHidden, state_is_tuple=True, activation=clipped_gelu)

    cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)

    fbH1, _, _ = tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputList, dtype=tf.float32,
                                         scope='BDLSTM_H1')
    fbH1rs = [tf.reshape(t, [batchSize, 2, nHidden]) for t in fbH1]
    outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]

    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

    ####Optimizing
    logits3d = tf.pack(logits)
    loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))

    lr = tf.Variable(0.005, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
    opt = tf.train.RMSPropOptimizer(lr)
    optimizer = opt.apply_gradients(zip(grads, tvars))

    ####Evaluating
    logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seqLengths[0], 1])
    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
    errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / \
                tf.to_float(tf.size(targetY.values))

def assign_lr(session, lr_value):
    session.run(tf.assign(lr, lr_value))

####Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(max_to_keep=1)
    for epoch in range(nEpochs):
        print('Epoch', epoch+1, '...')
        batchErrors = np.zeros(len(batchedData))
        batchRandIxs = np.random.permutation(len(batchedData))      # randomize batch order
        for batch, batchOrigI in enumerate(batchRandIxs):
            batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
            batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals.tolist(),
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
            _, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
            print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            batchErrors[batch] = er*len(batchSeqLengths)
        epochErrorRate = batchErrors.sum() / totalN
        print('Epoch', epoch+1, 'error rate:', epochErrorRate)
        if epoch % 10 == 0 and epoch > 0:
            saver.save(session, "./bdlstm-timit-clean.ckpt")
            print('Saved')
        if epoch == 50:
            session.run(tf.assign(lr, lr * 0.2))

