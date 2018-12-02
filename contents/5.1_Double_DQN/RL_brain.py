#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
The double DQN based on this paper: https://arxiv.org/abs/1509.06461

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class DoubleDQN:

    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.1,
            replace_target_iter=200,
            memory_size=3000,
            batch_size=32,
            e_greedy_reduction=None,
            output_graph=False,
            double_q=True,
            sess=None,
    ):
        # 行为的数目
        self.n_actions = n_actions
        # 状态特征的数目（S）（简单讲，就是最少用几个数可以描述一个状态）
        self.n_features = n_features
        # 学习率
        self.lr = learning_rate
        # 相邻两步影响的衰减率
        self.gamma = reward_decay
        # 学习过程中最小的epsilon的值（如果变的话，一定是由很大一点点变得很小）
        self.epsilon_min = e_greedy
        # 取代target的训练步数
        self.replace_target_iter = replace_target_iter
        # 抽样池样本的数目
        self.memory_size = memory_size
        # 一次抽样的样本数目
        self.batch_size = batch_size
        # epsilon减少的多少
        self.epsilon_reduction = e_greedy_reduction
        # 这里指的是随机选取的概率（如果e_greedy不减少，那么开始就直接去最小值
        self.epsilon = 1 if e_greedy_reduction is not None else self.epsilon_min

        # 这里的原理似乎是为了学习的稳定
        self.double_q = double_q    # decide to use double q or not
        # 学习次数的记录
        self.learn_step_counter = 0
        # 抽样池初始化
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        # 构造target网络和eval网络
        self._build_net()
        # 获取target网络的所有参数
        t_params = tf.get_collection('target_net_params')
        # 获取eval网络的所有藏书
        e_params = tf.get_collection('eval_net_params')
        # 网络替代的步骤（所有对应参数一一替代）
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # 是否要用已有的训练模型继续训练
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):

        # 创建某一网络的层
        # 基本层采用两层全连接层
        # 以状态为输入，输出所有Q值
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2

            return out

        # ------------------ build evaluate_net ------------------
        # 状态的变量，接受若干组样本以及他们的特征
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        # 目标的变量，若干组样本对应的Q值
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # target网络的输出是q_next
        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    # 存储所有的转变，形成一个抽样池
    def store_transition(self, s, a, r, s_):

        # 初始化抽样池的大小
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 相当于把一个样本拍扁，这样就可以直接索引了

        transition = np.hstack((s, [a, r], s_))
        # 循环记录样本
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        # 尽管循环记录样本，但是计数器并不循环
        self.memory_counter += 1

    # 根据观测抽取行为
    def choose_action(self, observation):

        # 在观测值外面再套一层中括号
        observation = observation[np.newaxis, :]
        # 采用eval网络评估当前状态，获得当前状态下各个action的Q值
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        # 获取最大的Q值对应的索引
        action = np.argmax(actions_value)

        # 相当于初始化
        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0

        # 有点像eligibility trace的记录
        # 但似乎是用来标记学习状况的
        # 因为如果当下的Q值越来越小，那么running_q就会变得越来越小，从而说明这一次的学习效果并不好
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        # 以上那部分代码，完全可以不做，做的原因是为了测试实际表现
        if np.random.uniform() > 1 - self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    # 学习的过程
    def learn(self):

        # 如果训练的次数达到阈值时，那么就更新target网络
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 只有在抽样池的大小达到要求后，才进行正常抽样
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        # 否则只在有效部分进行抽样
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        # 抽一个batch的样本，用batch_memory记录这些样本对应的转变
        batch_memory = self.memory[sample_index, :]

        # 分别计算target网络和eval网络的结果，用状态特征的相反数参与计算
        # 这里代码中的batch_memory[:, -self.n_features:]代表着取样本中的s_
        # 而下一行中的batch_memory[:, :self.n_features]则是代表样本中的s
        # 因为基本元素((s, [a, r], s_))是这种形式

        # 获得target网络对于下一步的判断（直接选择）
        # 获得eval网络对于下一步的判断（两个网络分别运算）
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation

        # 获得每个样本这一步状态下所有的Action的Q值
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        # 这一步只是为了书写的简单
        # 因为最后要极小化的是每个状态下，一个行为对应的Q值之差
        # 但是网络会输出当前状态下所有行为对应的Q值，为了最后极小化的方便，因而先复制一份副本
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # 对应于上面，这里两个分别选取的是action和reward
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            # 根据eval网络中下一步的最大值，选择对应的action
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            # 根据eval中的action获得target中每个样本下一步的Q值
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            # 直接获取这一步每个样本下一步的Q值
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        # 根据上一步获取的下一步的Q值，计算q_target内有意义的值
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # 极小化当前误差，对eval网络进行训练
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})

        # 获取当前误差
        self.cost_his.append(self.cost)

        # 改变epsilon的值，直到其变得很小
        self.epsilon = self.epsilon - self.epsilon_reduction if self.epsilon > self.epsilon_min else self.epsilon_min

        # 学习计数加一
        self.learn_step_counter += 1