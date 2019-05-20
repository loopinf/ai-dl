# -*- coding: utf-8 -*-
"""
Created on Wed May  8 20:35:23 2019

@author: shkim
"""

import tensorflow as tf
import numpy as np

def softmax_regression():
    # x = [[1., 2., 4., 5., 8., 9.],    # 공부한 시간
    #      [2., 1., 5., 4., 9., 8.]]    # 출석한 일수
    # y = [[0., 0., 1., 1., 1., 1.]]    # 합격 여부
    x = [[1., 2.],  # C
         [2., 1.],
         [4., 5.],  # B
         [5., 4.],
         [8., 9.],  # A
         [9., 8.]]

    y = [[0., 0., 1.],
         [0., 0., 1.],
         [0., 1., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [1., 0., 0.]]


    w = tf.Variable(tf.random_normal([2, 3]))  # ?????
    b = tf.Variable(tf.random_normal([3]))

    ph_x = tf.placeholder(tf.float32)

    # ?????
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    print('-'* 50)
    preds = sess.run(hx, {ph_x: x})
    print(preds, end='\n\n')
    preds_arg = np.argmax(preds, axis=1)
    y_arg = np.argmax(y, axis=1)
    print(preds_arg)
    print(y_arg)
    
    equals = [preds_arg == y_arg]  
    print(equals)
    print('acc :', np.mean(equals)) 


    sess.close()


softmax_regression()

print('\n\n\n\n')

