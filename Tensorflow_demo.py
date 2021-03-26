import tensorflow as tf
# 设置日志等级（取消warning）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tensorflow_demo():
    a = 2
    b = 3
    c = a + b
    print('Python:\n', c)
    tf.compat.v1.disable_v2_behavior()
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print('c_t:\n', c_t)
    # 开启会话
    # sess = tf.compat.v1.Session()
    # print(sess.run(c_t))
    with tf.compat.v1.Session() as sess:
        print(sess.run(c_t))

    return None


def graph_demo():
    tf.compat.v1.disable_v2_behavior()
    a_t = tf.constant(2, name='a_t')
    b_t = tf.constant(3, name='b_t')
    c_t = tf.add(a_t, b_t, name='c_t')

    # 法一：调用方法
    default_g = tf.compat.v1.get_default_graph()
    print("default graph:\n", default_g)
    # 法二：查看属性
    print("a_t.graph:\n", a_t.graph)
    print("c_t.graph:\n", c_t.graph)
    sess = tf.compat.v1.Session()
    print("sess.graph:\n", sess.graph)

    # 自定义图：
    new_g = tf.Graph()
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print('c_new:\n', c_new)
    new_sess = tf.compat.v1.Session(graph=new_g)
    print('c_new_value:\n', new_sess.run(c_new))
    print("a_new.graph:\n", a_new.graph)
    print("c_new.graph:\n", c_new.graph)
    print("new_sess.graph:\n", new_sess.graph)

    # 图的可视化
    # step1 将图写入本地events文件
    # tf.compat.v1.summary.FileWriter("./tmp/summary", graph=sess.graph)
    # step2 调用tensorboard

    return None


def session_demo():
    tf.compat.v1.disable_v2_behavior()
    a_t = tf.constant(2, name='a_t')
    b_t = tf.constant(3, name='b_t')
    c_t = tf.add(a_t, b_t, name='c_t')
    print('a_t:\n', a_t)
    print('b_t:\n', b_t)
    print('c_t:\n', c_t)

    a_ph = tf.compat.v1.placeholder(tf.float32)
    b_ph = tf.compat.v1.placeholder(tf.float32)
    c_ph = tf.add(a_ph, b_ph)

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as \
            sess:
        print("sess.run(c_t):\n", sess.run(c_t))
        print("c_t.eval(session=sess):\n", c_t.eval(session=sess))
        abc = sess.run([a_t, b_t, c_t])
        print("abc:\n", abc)
        c_ph_value = sess.run(c_ph, feed_dict={a_ph: 3.9, b_ph: 4.8})
        print("c_ph_value:\n", c_ph_value)
    return None


def tensor_demo():
    # 张量的类型修改
    tensor1 = tf.constant(4.0)
    tensor2 = tf.constant([1, 2, 3, 4])
    linear_squares = tf.constant([[4], [9], [16], [25]], dtype=tf.int32)
    print("tensor1:\n", tensor1)
    print("tensor2:\n", tensor2)
    print("linear_squares:\n", linear_squares)
    print("linear_squares_cast:\n", tf.cast(linear_squares, dtype=tf.float32))
    with tf.compat.v1.Session() as sess:
        print("tf.zeros.eval():\n", tf.zeros(shape=[3, 4], dtype=tf.float32).eval())
    # 张量静态形状的修改
    tf.compat.v1.disable_v2_behavior()
    a_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None])
    b_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 10])
    c_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3, 2])
    a_ph.set_shape([2, 3])
    b_ph.set_shape([2, 10])
    b_ph_reshape = tf.reshape(b_ph, shape=[2, 2, 5])
    print('a_ph:\n', a_ph)
    print("b_ph:\n", b_ph)
    print("b_ph_reshape:\n", b_ph_reshape)
    return None


def variable_demo():
    tf.compat.v1.disable_v2_behavior()
    # 创建变量
    with tf.compat.v1.variable_scope("my_scope"):
        a = tf.Variable(initial_value=50)
        b = tf.Variable(initial_value=40)
    with tf.compat.v1.variable_scope("your_scope"):
        c = tf.add(a, b)
    print(a)
    print(c)
    # 创建初始化变量
    init = tf.compat.v1.global_variables_initializer()
    # 开启会话
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        a_value, b_value, c_value = sess.run([a, b, c])
        print("{}   {}   {}".format(a_value, b_value, c_value))
    return None


# tensorflow_demo()
# graph_demo()
# session_demo()
# tensor_demo()
variable_demo()