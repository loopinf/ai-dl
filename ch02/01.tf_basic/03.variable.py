# -*- coding: utf-8 -*-
import tensorflow as tf

a = tf.constant(5, name='a')
b = tf.constant(2, name='b')
#print(a,b)
#print(type(a), type(b))

va = tf.Variable(5, name='va')
vb = tf.Variable(3, name='vb')
vc = tf.Variable(tf.zeros(0, tf.int32), name='vc')
#print(vc.get_shape())

sess = tf.Session()

print(sess.run(a))    #ok
#print(sess.run(va))      #error

sess.run(tf.global_variables_initializer())
print(sess.run(va))

print('-----------------')
#Tensor.eval returns a numpy array with the same contents as the tensor.
print(va.eval(sess))
print(vb.eval(sess))
print(vc.eval(sess))

print('-----------------')

va = va + 10
vb = vb - 5
vc = va
vc = vc + vb
print(sess.run(va))
print(sess.run(vb))
print(sess.run(vc))
sess.close()






