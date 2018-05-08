import tensorflow as tf
import numpy as np

def add_layer(input,in_size,out_size,n_layer=None,activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(input,Weights)+biases
        if activation_function==None:
            output=Wx_plus_b
        else:
            output=activation_function(Wx_plus_b)
        with tf.name_scope('output'):
            return output

with tf.name_scope('inputs'):
    xp=tf.placeholder(tf.float32,x.shape,name='x_input')
    yp=tf.placeholder(tf.float32,y.shape,name='y_input')

x=np.linspace(-1,1,100)[:,np.newaxis]
noise=np.random.normal(0,0.05,x.shape)
y=np.square(x)+noise

l1=add_layer(xp,1,10,n_layer=1,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,n_layer=2)
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-y),
                                      reduction_indices=[1]))
with tf.name_scope('train'):
    train=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init=tf.global_variables_initializer()
writer = tf.summary.FileWriter("tensorflow/", sess.graph)

with tf.Session() as sess:
    sess.run(init)
    for step in range(100):
        sess.run(train,feed_dict={xp:x,yp:y})
 #       if step%10==0:
 #           print(sess.run(loss,feed_dict={xp:x,yp:y}))
            