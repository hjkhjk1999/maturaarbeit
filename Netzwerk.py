import tensorflow as tf
import numpy as np
import os
tf.logging.set_verbosity(tf.logging.INFO)

patients = [id.replace(".npy", "") for id in os.listdir("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/64-256-256_data/")]
patients.sort()

def load_array(path):
    return np.load(path)

def get_training_batches():
    much_data = []
    training_data = []
    training_labels = []
    onehot_training_labels = []
    index = np.random.randint(low=0,high=len(patients[:-100-batch_size]), size=1)
    index = np.asscalar(index)
    for ix, patient in enumerate(patients[index:index+batch_size]):
        sample = load_array("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/64-256-256_data/{}.npy".format(patient))
        much_data.append(sample)
    for data in much_data:
        training_data.append(data[0][0])
        training_labels.append(data[1])
    trainig_data = np.array(training_data, dtype = np.float32)
    training_labels = np.array(training_labels, dtype = np.float32)
    for label in training_labels:
        if label == 1:
            onehot_training_labels.append(np.array([1,0]))
        elif label == 0:
            onehot_training_labels.append(np.array([0,1]))
    onehot_training_labels = np.array(onehot_training_labels, dtype=np.float32)
    return training_data, onehot_training_labels

def get_test_batches():
    much_data = []
    training_data = []
    training_labels = []
    onehot_training_labels = []
    index = np.random.randint(low=len(patients[:-100]),high=len(patients[:-batch_size]),size = 1)
    index = np.asscalar(index)
    for ix, patient in enumerate(patients[index:index+batch_size]):
        sample = load_array("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/64-256-256_data/{}.npy".format(patient))
        much_data.append(sample)
    for data in much_data:
        training_data.append(data[0][0])
        training_labels.append(data[1])
    trainig_data = np.array(training_data, dtype = np.float32)
    training_labels = np.array(training_labels, dtype = np.float32)
    for label in training_labels:
        if label == 1:
            onehot_training_labels.append(np.array([1,0]))
        elif label == 0:
            onehot_training_labels.append(np.array([0,1]))
    onehot_training_labels = np.array(onehot_training_labels, dtype=np.float32)
    return training_data, onehot_training_labels
    
x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv3d(x,filters, kernel):
    return tf.layers.conv3d(x, filters, kernel, use_bias = True, strides = (2,2,2),
                            bias_initializer = tf.zeros_initializer(), activation = tf.nn.relu, padding = 'SAME')

def max_pool(x):
    return tf.nn.max_pool3d(x, ksize = [1,2,2,2,1], strides = [1,2,2,2,1], padding = 'SAME')

x_image = tf.reshape(x, [-1,64,256,256,1])

W_conv1 = (3,6,6)
filters_1 = 32
h_conv1 = conv3d(x_image, filters_1, W_conv1)
h_pool1 = max_pool(h_conv1)

W_conv2 = (3,4,4)
filters_2 = 64
h_conv2 = conv3d(h_pool1, filters_2, W_conv2)
h_pool2 = max_pool(h_conv2)

W_conv3 = (2,2,2)
filters_3 = 128
h_conv3 = conv3d(h_pool2, filters_3, W_conv3)
h_pool3 = max_pool(h_conv3)

W_fc1 = weight_variable([2048, 1024])
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool3, [-1, 2048])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

batch_size = 2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(y_,1), tf.argmax(y_conv,1))

saver = tf.train.Saver()
train_accuracy_list = []
train_i =[]
test_accuracy_list = []
test_confmatrix_list = []
test_i = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("initialized")
    for i in range(0,100000):
        training_data_batch, training_labels_batch = get_training_batches()
        eval_data_batch, eval_labels_batch = get_test_batches()
        if i % 100 ==0:
            train_accuracy = accuracy.eval(feed_dict = {x: training_data_batch, y_: training_labels_batch, keep_prob: 1.0})
            print("step %d, training accuracy %g" %(i, train_accuracy))
        if i % 100 ==0:
            test_accuracy = accuracy.eval(feed_dict = {x: eval_data_batch, y_: eval_labels_batch, keep_prob: 1.0})
            test_confmatrix = confusion_matrix.eval(feed_dict = {x: eval_data_batch, y_: eval_labels_batch, keep_prob: 1.0})
            print("step %d, test accuracy %g" %(i, test_accuracy))
            print("test confusion matrix: ",test_confmatrix)
        try:
            train_step.run(feed_dict = {x: training_data_batch, y_: training_labels_batch, keep_prob: 0.4})
        except Exception as e:
            print("PASSED PASSED PASSED")
            pass
    saver.save(sess, "D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/processed_64-256-256_data/tmp6/model.ckpt")