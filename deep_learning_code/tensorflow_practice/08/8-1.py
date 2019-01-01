import tensorflow as tf

# 定义常量
# 打印出来是一个int32张量，而不是一个具体数值
input1 = tf.constant(1)
print(input1)

# 定义变量
# 没有在会话中执行过，所以真正的发生真实值的改变
input2 = tf.Variable(2,tf.int32)
print(input2)

# 打印结果为1，将1赋值给2，然后进入session后发生真实值得改变
input2 = input1
sess = tf.Session()
print(sess.run(input2))
