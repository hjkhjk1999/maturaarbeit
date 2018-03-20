import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
tf.logging.set_verbosity(tf.logging.INFO)

patients = [id.replace(".npy", "") for id in os.listdir("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/64-256-256_data/")]
patients.sort()

def load_array(path):
    return np.load(path)

def get_training_batches(index):
    much_data = []
    training_data = []
    training_labels = []
    onehot_training_labels = []
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

def get_test_batches(index):
    much_data = []
    training_data = []
    training_labels = []
    onehot_training_labels = []
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
print(h_conv1.shape)
print(h_pool1.shape)

W_conv2 = (3,4,4)
filters_2 = 64
h_conv2 = conv3d(h_pool1, filters_2, W_conv2)
h_pool2 = max_pool(h_conv2)
print(h_conv2.shape)
print(h_pool2.shape)

W_conv3 = (2,2,2)
filters_3 = 128
h_conv3 = conv3d(h_pool2, filters_3, W_conv3)
h_pool3 = max_pool(h_conv3)
print(h_conv3.shape)
print(h_pool3.shape)

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

def get_test_evaluation(i):
    much_data = []
    training_data = []
    training_labels = []
    onehot_training_labels = []
    if i == 0:
        for ix, patient in enumerate(patients[-100:-50]):
            sample = load_array("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/64-256-256_data/{}.npy".format(patient))
            much_data.append(sample)
    if i == 1:
        for ix, patient in enumerate(patients[-50:]):
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

def get_train_evaluation(i):
    much_data = []
    training_data = []
    training_labels = []
    onehot_training_labels = []
    for ix, patient in enumerate(patients[0+i:i+50]):
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
	
saver = tf.train.Saver()
	
with tf.Session() as sess:
    test_accuracy_list = []
    for epoch in range(0,100):
        saver.restore(sess, 
        "D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/processed_64-256-256_data/tmp6/model.ckpt")
        eval_data1, onehot_eval_labels1 = get_test_evaluation(0)
        test_accuracy1= accuracy.eval(feed_dict = {x: eval_data1, y_: onehot_eval_labels1, keep_prob: 1.0})
        eval_data2, onehot_eval_labels2 = get_test_evaluation(1)
        print(test_accuracy1)
        test_accuracy2= accuracy.eval(feed_dict = {x: eval_data2, y_: onehot_eval_labels2, keep_prob: 1.0})
        print(test_accuracy2)
        test_accuracy = (test_accuracy1+test_accuracy2)/2
        print(test_accuracy)
        test_accuracy_list.append(test_accuracy)
    np.save("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/processed_64-256-256_data/test_accuracy.npy",
            test_accuracy_list)
			
with tf.Session() as sess:
    train_accuracy_list = []
    train_list = []
    conf_list = []
    saver.restore(sess, 
        "D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/processed_64-256-256_data/tmp6/model.ckpt")
    for i in range(0, len(patients[:-150]),50):
        train_data, onehot_train_labels = get_train_evaluation(i)
        train_accuracy = accuracy.eval(feed_dict = {x: train_data, y_: onehot_train_labels, keep_prob: 1.0})
        print(train_accuracy)
        train_list.append(train_accuracy)
        conf_matrix = confusion_matrix.eval(feed_dict = 
                                                    {x: train_data, y_: onehot_train_labels, keep_prob: 1.0})
        print(conf_matrix)
        conf_list.append(conf_matrix)
    accuracy_1 = np.mean(train_list)
    print("accuracy",accuracy_1)
    np.save("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/processed_64-256-256_data/train_accuracy.npy",
            train_list)
    np.save("D:/Datasets/Lung Cancer/Data/Stage 1/Pictures/processed_64-256-256_data/train_confmatrix.npy",
            conf_list)