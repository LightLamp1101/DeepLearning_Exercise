# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
# import tensorflowvisu
import mnistdata
import math
import matplotlib.pyplot as plt
import numpy as np
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])	# 사람이 쓴 숫자 이미지(0~255 사이의 스칼라값)
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])			# 최종 라벨값이 들어갈 플레이스 홀더
# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 784])		# X를 한줄(784)로 쫙 펼처줌

# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + b) # 행렬을 곱하여 강조

# loss function: MSE
loss = tf.reduce_mean(tf.squared_difference(Y, Y_)) * 1000 # 손실 함수 (제곱하여 최소값을 찾음) 학습이 원활하게 하도록 1000을 증폭

# accuracy of the trained model, between 0 (worst) and 1 (best)
# Y_ 가 정답, Y가 예측값 정답과 예측값을 비교하여 예측된 숫자의 인덱스값을 추출
# 그것으로 인덱스값을 비교하여 정답인지 확인
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 정확도를 계산하는 부분

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)	# 옵티마이저를 돌림

# init
init = tf.global_variables_initializer()
sess = tf.Session()		# 세션을 통해 CPU나 GPU 할당 
sess.run(init)			# 변수들을 사용할 수 있게 초기화시킴


y1 = []
y2 = []
# run 실행부 
for i in range(2000 + 1) :

	# 배치 X에는 이미지 Y에는 정답이 들어감
    batch_X, batch_Y = mnist.train.next_batch(100)	# 100개씩 샘플링 하는 과정(100/60000)
    
    # 정확도 계산과 loss 오퍼레이션을 실행하는데 그곳에 쓰일 플레이스홀더인 배치X와 Y를 feeddick으로 넘겨줌
    a, c = sess.run([accuracy, loss], feed_dict={X : batch_X, Y_ : batch_Y})
    
    print("training : ", i, a,c)
    y1.append(a)
    
    

    # test_batch_X, test_batch_Y = mnist.test.next_batch(100)  ==> never use mini batch!!
    # sess.run(train_step, feed_dict={X: test_batch_X, Y_: test_batch_Y})  ==> never run train_step on test data!!
    # 학습 완료 후 W와 d값을 가지고 10000장을 테스팅
    a, c = sess.run([accuracy, loss], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
    print("testing  : ",i, a, c)

	y2.append(a)

    # 연산마다 sess.run 을 수행
    sess.run(train_step, feed_dict={X : batch_X, Y_ : batch_Y} )

    # 발간선은 테스트데이터의  파란선은 트레이닝의


x = np.arange(len(y1))
plt.figure(1)
plt.plot(x,y1, label = 'train_acc_list')
plt.plot(x,y2, label = 'train_lost_list')
plt.legend()
plt.show()