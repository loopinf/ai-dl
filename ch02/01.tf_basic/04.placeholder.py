# -*- coding: utf-8 -*-
import tensorflow as tf

va = tf.Variable(5.0, name='va')
pa = tf.placeholder(tf.float32, name='pa')  #any shape ok
#print(pa)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(va.eval(sess))
#print(pa.eval(sess))    #error
t = pa + 1.0
print(t.eval(session=sess, feed_dict={pa:8.0}))    #ok
print(t.eval(session=sess, feed_dict={pa:[8.0,3.0]}))  #ok
print('-----------------')

#ta = tf.placeholder(tf.float32)    # any shape ok
ta = tf.placeholder(tf.float32, 3)  # shape(3,)
tb = tf.placeholder(tf.float32, 1)  # shape(1,)
tc = tf.multiply(ta, tb)
print(ta)
print(tb)
print(tc)
print(sess.run(tc, feed_dict={ta:[1.,2.,3.], tb:[2.]}))

sess.close()

