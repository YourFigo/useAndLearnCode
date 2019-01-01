import tensorflow as tf

# placeholder 定义一个占位符，在会话运行时传入数据
input1 = tf.placeholder(tf.int32)
input2 = tf.placeholder(tf.int32)
print(input1)
print(input2)

output = tf.add(input1, input2)

sess = tf.Session()
print(sess.run(output, feed_dict={input1:[1], input2:[2]}))
