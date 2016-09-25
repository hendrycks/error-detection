from tensorflow.python.ops import ctc_ops as ctc
# from tensorflow.contrib.ctc import ctc_ops as ctc   # depreciated in future
import tensorflow as tf
import numpy as np
from utils import load_batched_data, target_list_to_sparse_tensor
import pickle
import sklearn.metrics as sk
import gc

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

# we will the last 1300 examples from the 6300
data_list = []
for i in range(1300//batchSize):
    offset = 5000 + batchSize * i
    target_list = []
    for j in range(batchSize):
        target_list.append(data['y_phones'][offset+j])
    data_list.append(
        (data['x'][offset:offset+batchSize,:,:],
         target_list_to_sparse_tensor(target_list),
         data['mask'][offset:offset+batchSize]))

del data

batchedData, maxTimeSteps, totalN = data_list, 776, 13


def clipped_gelu(x):
    return tf.minimum(0.5 * x * (1 + tf.tanh(x)), 6)

####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():

    ####NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow

    ####Graph input
    inputX = tf.placeholder(tf.float32, shape=(batchSize, maxTimeSteps, nFeatures))

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
    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
    errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / \
                tf.to_float(tf.size(targetY.values))

session = tf.InteractiveSession(graph=graph)
tf.initialize_all_variables().run()
saver = tf.train.Saver(max_to_keep=1)
saver.restore(session, "./bdlstm-timit-clean.ckpt")
print('Model Restored')

kl_all = []
pred_all = []

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

batchErrors = np.zeros(len(batchedData))
batchRandIxs = np.random.permutation(len(batchedData))      # randomize batch order
for batch, batchOrigI in enumerate(batchRandIxs):
    batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
    batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
    feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals.tolist(),
                targetShape: batchTargetShape, seqLengths: batchSeqLengths}
    er, preds = session.run([errorRate, logits3d], feed_dict=feedDict)

    for i in range(preds.shape[1]):
        preds_cut_by_time = preds[:int(batchSeqLengths[i]), i, :]
        # remove example where blank is predicted
        s_pred_blanks_removed = softmax(preds_cut_by_time[:,:39])

        kl = np.mean(np.log(nFeatures-1) + np.sum(s_pred_blanks_removed * np.log(s_pred_blanks_removed + 1e-11), axis=1))

        kl_all.append(kl)
        pred_all.append(np.mean(np.max(s_pred_blanks_removed, axis=1)))

    batchErrors[batch] = er*len(batchSeqLengths)
epochErrorRate = batchErrors.sum() / len(batchedData)

print('Edit distance', epochErrorRate, 'Softmax Confidence (mean, std)', np.mean(pred_all), np.std(pred_all))

del data_list; del batchedData; del batch   # save memory

gc.collect()

for oos_name in ['airport', 'babble', 'car', 'exhibition', 'restaurant', 'street', 'subway', 'train']:
    print('Loading OOD data')
    data = pickle.load(open("TIMIT_data_prepared_for_CTC_" + oos_name + ".pkl", 'rb'), encoding='latin1')

    # 6300 x 776 x 39

    # we will the last 1300 examples from the 6300
    data_list = []
    for i in range(1300//batchSize):
        offset = 5000 + batchSize * i
        target_list = []
        for j in range(batchSize):
            target_list.append(data['y_phones'][offset+j])
        data_list.append(
            (data['x'][offset:offset+batchSize,:,:],
             target_list_to_sparse_tensor(target_list),
             data['mask'][offset:offset+batchSize]))

    del data

    batchedData, maxTimeSteps, totalN = data_list, 776, 13

    kl_ood = []
    pred_ood = []

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    batchErrors = np.zeros(len(batchedData))
    batchRandIxs = np.random.permutation(len(batchedData))      # randomize batch order
    for batch, batchOrigI in enumerate(batchRandIxs):
        batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
        batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
        feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals.tolist(),
                    targetShape: batchTargetShape, seqLengths: batchSeqLengths}
        er, preds = session.run([errorRate, logits3d], feed_dict=feedDict)

        for i in range(preds.shape[1]):
            preds_cut_by_time = preds[:int(batchSeqLengths[i]), i, :]
            # remove example where blank is predicted
            s_pred_blanks_removed = softmax(preds_cut_by_time[:,:39])

            kl = np.mean(np.log(nFeatures-1) + np.sum(s_pred_blanks_removed * np.log(s_pred_blanks_removed + 1e-11), axis=1))

            kl_ood.append(kl)
            pred_ood.append(np.mean(np.max(s_pred_blanks_removed, axis=1)))

        batchErrors[batch] = er*len(batchSeqLengths)
    epochErrorRate = batchErrors.sum() / len(batchedData)

    print(oos_name, 'edit distance', epochErrorRate, 'Softmax Confidence (mean, std)', np.mean(pred_ood), np.std(pred_ood))

    print('\n' + oos_name, 'KL[p||u]: In/out distribution distinction')
    in_sample, oos = kl_all, kl_ood
    labels = np.zeros((len(in_sample) + len(oos)), dtype=np.int32)
    labels[:len(in_sample)] += 1
    examples = np.squeeze(np.vstack((np.array(in_sample).reshape((-1,1)), np.array(oos).reshape((-1,1)))))
    print('AUPR', sk.average_precision_score(labels, examples))
    print('AUROC', sk.roc_auc_score(labels, examples))

    print('\n' + oos_name, 'Prediction Prob: In/out distribution distinction')
    in_sample, oos = pred_all, pred_ood
    labels = np.zeros((len(in_sample) + len(oos)), dtype=np.int32)
    labels[:len(in_sample)] += 1
    examples = np.squeeze(np.vstack((np.array(in_sample).reshape((-1,1)), np.array(oos).reshape((-1,1)))))
    print('AUPR', sk.average_precision_score(labels, examples))
    print('AUROC', sk.roc_auc_score(labels, examples))

    del data_list; del batchedData; del batch   # save memory; it's possible that this doesn't work at all
    gc.collect()
