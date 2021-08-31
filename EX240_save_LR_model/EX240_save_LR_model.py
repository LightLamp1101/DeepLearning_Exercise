import tensorflow as tf

x_data = [1, 2, 3, 4]
y_data = [2, 4, 6, 8]

# -1에서 1사이의 하나의 값을 정규분포로 랜덤하게 할당
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='b')

# 학습데이터를 받아들일 각각의 플레이스홀더
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# y 대신 hypothesis라고 명명 Wx + b 값을 구하고자함
hypothesis = W * X + b

# 정답과 실제의 차이를 구함 코스트 혹은 에러
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#텐서플로 안에 이미 만들어져 있는 옵티마이저를 적용시켜 코스트를 구함
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1) # 0.1로 범위지정

train_op = optimizer.minimize(cost)


with tf.Session() as sess:

	saver = tf.train.Saver()

	#변수 초기화
	sess.run(tf.global_variables_initializer())

	for step in range(1,1001):

		#학습과정
		_, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

		print(step, cost_val, sess.run(W), sess.run(b))


	saver.save(sess, './model/my_model')
	print("\n=== Test ===")
	print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
	print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))