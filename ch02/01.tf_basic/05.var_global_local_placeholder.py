# https://www.tensorflow.org/api_docs/python/tf/

import tensorflow as tf

vl = tf.local_variables()
vl = [[1,10],20,30]
print(vl)
print(type(vl))

vg = tf.Variable(tf.zeros(3, dtype=tf.int32), name='vg')
#print(vg)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(vg))

td = tf.zeros((3,2))
print(sess.run(td))    # [[0. 0.][0. 0.][0. 0.]]

ta = tf.placeholder(tf.float32, (2,2))
tb = tf.placeholder(tf.float32, (1,2))
tc = tf.multiply(ta, tb)
print(sess.run(tc, feed_dict={ta:[[1.,2.],[3., 1.]], tb:[[2.,1.]]}))

ta = tf.placeholder(tf.float32, (2,2))
tb = tf.placeholder(tf.float32, (2,1))
tc = tf.multiply(ta, tb)
print(sess.run(tc, feed_dict={ta:[[1.,2.],[3., 1.]], tb:[[2.],[1.]]}))
sess.close()