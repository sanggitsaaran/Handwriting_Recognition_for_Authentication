import cv2
import os
import tensorflow.compat.v1 as tf

import sklearn.metrics as sk
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random


tf.compat.v1.disable_eager_execution()
current_dir = os.path.dirname(__file__)

author = '045'
training_folder = os.path.join(current_dir, 'data/training/', author)
test_folder = os.path.join(current_dir, 'data/test/', author)


import cv2
import numpy as np


def imageprep(input):
    # preprocessing the image input
    if len(input.shape) > 2 :
       input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    
    clean = cv2.fastNlMeansDenoising(input,31,7,21)
    ret, tresh = cv2.threshold(clean, 127, 1, cv2.THRESH_BINARY_INV)
    img = imgcrop(tresh)

    # 32x10 image as a flatten array
    flatten_img = cv2.resize(img, (32, 10), interpolation=cv2.INTER_AREA).flatten()

    # resize to 320x100
    resized = cv2.resize(img, (320, 100), interpolation=cv2.INTER_AREA)
    columns = np.sum(resized, axis=0)  # sum of all columns
    lines = np.sum(resized, axis=1)  # sum of all lines
    
    h, w = img.shape
    aspect = w / h
    
    return [*flatten_img, *columns, *lines, aspect]


def imgcrop(img):
    points = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(points)
    return img[y: y+h, x: x+w]

def main():
    print('OpenCV version {} '.format(cv2.__version__))

    current_dir = os.path.dirname(__file__)

    author = '045'
    training_folder = os.path.join(current_dir, 'data/training/', author)
    test_folder = os.path.join(current_dir, 'data/test/', author)
    authentication_folder= os.path.join(current_dir, 'data/authenticating',author)

    training_data = []
    training_labels = []
    for filename in os.listdir(training_folder):
        img = cv2.imread(os.path.join(training_folder, filename), 0)
        if img is not None:
            data = imageprep(img)
            
            training_data.append(data)
            training_labels.append([0, 1] if "genuine" in filename else [1, 0])
    
    
    authentication_files = []
    authentication_data = []
    authentication_labels = []
    y_true = []

    #all_files = os.listdir(authentication_folder)
    #random_filename = random.choice(all_files)
    filename='test_2.png'

    img = cv2.imread(os.path.join(authentication_folder,filename), 0)
    if img is not None:
        data = imageprep(img)
        
        authentication_files.append(filename)
        authentication_data.append(data)
        authentication_labels.append([0, 1])
        print("authenticating....")            
        authentication=runmodel(training_data,training_labels, authentication_data,authentication_labels)
        if authentication==True:
            print("authenticated")
        else:
            print("access denied")
        
    test_files = []
    test_data = []
    test_labels = []
    pred_gen = []
    y_true = []
    for filename in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder, filename), 0)
        if img is not None:
            data = imageprep(img)
            test_files.append(filename)
            test_data.append(data)
            test_labels.append([0, 1] if "genuine" in filename else [1, 0])
            y_true.append(True if "genuine" in filename else False )

    predictions = runmodel(training_data, training_labels, test_data, test_labels)
    for i in range(len(test_files)) :
         print(f"Image : {test_files[i]} is ",predictions[i],test_labels[i])
         if predictions[i] == True : 
              pred_gen.append(True if "genuine" in test_files[i] else False)
         else :
              pred_gen.append(False if "genuine" in test_files[i] else True)
    
    cm_display = sk.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_true,pred_gen), display_labels = [False, True])
    cm_display.plot()
    fpr , tpr , thresholds = sk.roc_curve(y_true,pred_gen)
    roc_auc = sk.auc(fpr,tpr)
    rocplt = sk.RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc)
    rocplt.plot()
    plt.show()
    print(f"Accuracy on test data: {roc_auc:.2%}")



# Softmax Regression Model
def smregression(x):
    W = tf.Variable(tf.zeros([741, 2]), name="W")
    b = tf.Variable(tf.zeros([2]), name="b")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y, [W, b]


def runmodel(training_data, training_labels, test_data, test_labels):
    # Model
    with tf.variable_scope("regression"):
        x = tf.placeholder(tf.float32, [None, 741])
        y, variables = smregression(x)

    # Training
    y_ = tf.placeholder("float", [None, 2])

    # Simple cross entropy function of -SUM (y'i log(yi))

    cross_entropy = -tf.reduce_sum(y_ * tf.math.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_step, feed_dict={x: training_data, y_: training_labels})
        predictions = sess.run(correct_prediction, feed_dict={x: test_data,y_:test_labels})
        accuracy_value = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})

    
   

    return predictions        

if __name__ == '__main__':
    main()