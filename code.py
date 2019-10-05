import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
import pandas as pd
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_dir = './gnr638-mls4rs-a1/train/'
test_dir = './gnr638-mls4rs-a1/test/'

def loadTrainingImages(path):
    aircrafts_images = sorted([os.path.join(path,'aircrafts', file) for file in os.listdir(path + 'aircrafts') if file.endswith('.png')])
    tag_aircraft_images = [[1,0,0] for file in os.listdir(path + 'aircrafts') if file.endswith('.png')]
    ships_images = sorted([os.path.join(path,'ships', file) for file in os.listdir(path + 'ships') if file.endswith('.png')])
    tag_ships_images = [[0,1,0] for file in os.listdir(path + 'ships') if file.endswith('.png')]
    none_images = sorted([os.path.join(path,'none', file) for file in os.listdir(path + 'none') if file.endswith('.png')])
    tag_none_images = [[0,0,1] for file in os.listdir(path + 'none') if file.endswith('.png')]
    return aircrafts_images,tag_aircraft_images,ships_images,tag_ships_images,none_images,tag_none_images

def preprocessing(image, training = False):
    img = tf.read_file(image)
    img = tf.image.decode_png(img,channels=3)
    img = tf.image.resize_images(img,[224,224],method=tf.image.ResizeMethod.BICUBIC)
    img /= 255.0
    return img

def split_dataset(images,labels,split_size = 0.8):
    num_images = len(images)
    train_idx = np.random.choice(num_images,size=int(split_size*num_images),replace=False)
    validation_idx = [idx for idx in range(num_images) if idx not in train_idx]
    x_train = [] 
    y_train = []
    x_validate = []
    y_validate = []
    for index in train_idx:
        x_train.append(images[index])
        y_train.append(labels[index])
    for index in validation_idx:
        x_validate.append(images[index])
        y_validate.append(labels[index])
    return x_train,y_train,x_validate,y_validate

(aircrafts_images,y_aircrafts,ships_images,y_ships,none_images,y_none) = loadTrainingImages(train_dir)
(x_train_aircrafts,y_train_aircrafts,x_validate_aircrafts,y_validate_aircrafts) = split_dataset(aircrafts_images,y_aircrafts)
(x_train_ships,y_train_ships,x_validate_ships,y_validate_ships) = split_dataset(ships_images,y_ships) 
(x_train_none,y_train_none,x_validate_none,y_validate_none) = split_dataset(none_images,y_none)
x_train = x_train_aircrafts + x_train_ships + x_train_none
y_train = y_train_aircrafts + y_train_ships + y_train_none
x_validate = x_validate_aircrafts + x_validate_ships + x_validate_none
y_validate = y_validate_aircrafts + y_validate_ships + y_validate_none

# import matplotlib.pyplot as plt
# plt.imshow(x_train[1])

x_path = tf.placeholder(tf.string)
x_proc = preprocessing(x_path)
x = tf.placeholder(tf.float32,shape=[None,224,224,3])
y_true = tf.placeholder(tf.float32,shape=[None,3])
y_true_cls = tf.argmax(y_true, dimension=1)

def conv_layer(input, num_input_channels, filter_size, num_filters, name):
    with tf.variable_scope(name) as scope:
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='VALID')
        layer += biases
        return layer, weights
    
def pool_layer(input, name):
    with tf.variable_scope(name) as scope:
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        return layer

def relu_layer(input, name):
    with tf.variable_scope(name) as scope:
        layer = tf.nn.relu(input)
        return layer
    
def fc_layer(input, num_inputs, num_outputs, name):
    with tf.variable_scope(name) as scope:
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        layer = tf.matmul(input, weights) + biases
        return layer

layer_conv1, weights_conv1 = conv_layer(input=x, num_input_channels=3, filter_size=3, num_filters=32, name ="conv1")
layer_pool1 = pool_layer(layer_conv1, name="pool1")
layer_relu1 = relu_layer(layer_pool1, name="relu1")

layer_conv2, weights_conv2 = conv_layer(input=layer_relu1, num_input_channels=16, filter_size=5, num_filters=64, name= "conv2")
layer_pool2 = pool_layer(layer_conv2, name="pool2")
layer_relu2 = relu_layer(layer_pool2, name="relu2")

layer_conv3, weights_conv3 = conv_layer(input=layer_relu2, num_input_channels=32, filter_size=3, num_filters=128, name= "conv3")
layer_pool3 = pool_layer(layer_conv3, name="pool3")
layer_relu3 = relu_layer(layer_pool3, name="relu3")

layer_conv4, weights_conv4 = conv_layer(input=layer_relu3, num_input_channels=64, filter_size=3, num_filters=128, name= "conv4")
layer_pool4 = pool_layer(layer_conv4, name="pool4")
layer_relu4 = relu_layer(layer_pool4, name="relu4")

num_features = layer_relu4.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_relu4, [-1, num_features])

layer_fc1 = fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")
layer_relu3 = relu_layer(layer_fc1, name="relu3")

layer_fc2 = fc_layer(input=layer_relu3, num_inputs=128, num_outputs=3, name="fc2")

with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
writer = tf.summary.FileWriter("Training_FileWriter/")
writer1 = tf.summary.FileWriter("Validation_FileWriter/")

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()

init = tf.global_variables_initializer()

batch_size = 100
with tf.Session() as sess:
    sess.run(init)
    writer.add_graph(sess.graph)
    for epoch in range(50):
        start_time = time.time()
        train_accuracy = 0
    
        x_processed = []
        for path in x_train:
            x_processed.append(sess.run(x_proc,feed_dict = {x_path:path}))
        
        p = np.random.permutation(len(x_train))
        x_processed = np.asarray(x_processed)
        x_processed = x_processed[p]
        y_new = np.asarray(y_train)
        y_new = y_new[p]     
#         feed_dict_train = {x: x_processed, y_true: y_train}
#         sess.run(optimizer, feed_dict=feed_dict_train)
#         train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
#         summ = sess.run(merged_summary, feed_dict=feed_dict_train)
#         writer.add_summary(summ, epoch)        
        
        for batch in range(0, int(len(x_train)/batch_size)):
            x_batch = x_processed[batch*batch_size:(batch+1)*batch_size:1]
            y_true_batch = y_new[batch*batch_size:(batch+1)*batch_size:1]
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            sess.run(optimizer, feed_dict=feed_dict_train)
            train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
            summ = sess.run(merged_summary, feed_dict=feed_dict_train)
            writer.add_summary(summ, epoch*int(len(x_train)/batch_size) + batch)
            
        train_accuracy /= int(len(x_train)/batch_size)
        
        x_processed = []
        for path in x_validate:
            x_processed.append(sess.run(x_proc,feed_dict = {x_path:path}))
        x_processed = np.asarray(x_processed)
        y_val_new = np.asarray(y_validate)
        summ, vali_accuracy = sess.run([merged_summary, accuracy], feed_dict={x:x_processed, y_true:y_val_new})
        writer1.add_summary(summ, epoch)

        end_time = time.time()
        print(epoch)
        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\tAccuracy:")
        print ("\t- Training Accuracy:\t{}".format(train_accuracy))
        print ("\t- Validation Accuracy:\t{}".format(vali_accuracy))


# x_test = sorted([os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith('.png')])

# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     x_processed = []
#     for path in x_test:
#         x_processed.append(sess.run(x_proc,feed_dict = {x_path:path}))
#     x_processed = np.asarray(x_processed)
#     y_p = sess.run(y_pred_cls,feed_dict = {x:x_processed})
    
    