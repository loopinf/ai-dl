# -*- coding: utf-8 -*-
import tensorflow as tf

#hello ='Hello, TensorFlow!!'
#print(hello)

hello = tf.constant('Hello, TensorFlow!!')
print(hello)

sess = tf.Session()

print(hello.eval(session=sess))
print(sess.run(hello))
sess.close()

