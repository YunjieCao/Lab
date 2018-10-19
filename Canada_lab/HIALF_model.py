# -*- coding: utf-8 -*-
# @Time    : 2018/10/5 13:18
# @Author  : Yunjie Cao
# @FileName: HIALF_model.py
# @Software: PyCharm
# @Email   ：Cyj19970823@gmail.com

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import math
import sys
from tensorflow.python import debug as tf_debug


class HIALF:
    def __init__(self):
        self.g = None
        self.user_ids = None
        self.item_ids = None
        self.Bu = None
        self.Bp = None
        self.X = None
        self.Y = None
        self.sequence_info = None

    def view_bar(self, message, num, total):
        rate = num / total
        rate_num = int(rate * 40)
        rate_nums = math.ceil(rate * 100)
        r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total)
        sys.stdout.write(r)
        sys.stdout.flush()

    def get_sequence_info(self):
        """
            get the (u,i,p,p_e,r) information from the historical data
            return : sequence information of aforementioned pair
        """
        print('start preprocessing...')
        file_path = './lfm/data/ratings.csv'
        frame = pd.read_csv(file_path)

        # get the data of each item and sort them by the Timestamp
        item_ids = set(frame['MovieID'])
        item_rating_info = {}
        for item in item_ids:
            item_info = frame.loc[frame['MovieID'] == item].sort_values(by='Timestamp', ascending='True')
            item_rating_info[item] = item_info

        # get the global average rating of all data
        g = frame['Rating'].mean()
        # print(g)
        print('global rating bias is{}'.format(g))

        # get the bias of every user
        user_ids = set(frame['UserID'])
        Bu = np.zeros([max(user_ids), 1])
        for user in user_ids:
            user_score = frame.loc[frame['UserID'] == user]['Rating'].mean()
            Bu[user - 1] = (user_score - g)

        # get the bias of every item
        Bp = np.zeros([max(item_ids), 1])
        for item in item_ids:
            item_score = frame.loc[frame['MovieID'] == item]['Rating'].mean()
            Bp[item - 1] = (item_score - g)

        # try to get the i-th of every product
        # 1. get the length of ratings of every product
        # 2. split the data to train dataset and test dataset 0.8 , 0.2 only choose the product having more than 1 rating
        # 3. get the i-th (product,user) out
        # solve step1,2

        len_of_every_item = {}
        for item in item_ids:
            Length_of_the_item = len(frame.loc[frame['MovieID'] == item])
            if Length_of_the_item > 2:
                len_of_every_item[item] = math.floor(float(Length_of_the_item) * 0.8)
        frame.sort_values(['MovieID', 'Timestamp'], ascending=[1, 1], inplace=True)
        # solve setp3
        # 1.sort the dataframe in item and then timestamp
        Max_len = 2742  # 2742
        sequence_info = {}
        # i : [(product,user,e,r),...]
        # start from the second rating
        for i in range(2, Max_len + 1):
            sequence_info[i] = []
            get_previous_i = frame.groupby(['MovieID']).head(i)
            for item in len_of_every_item.keys():
                if (i <= len_of_every_item[item]):
                    which_item = item
                    which_user = get_previous_i.loc[get_previous_i['MovieID'] == item].iloc[i - 1]['UserID']
                    which_rate = get_previous_i.loc[get_previous_i['MovieID'] == item].iloc[i - 1]['Rating']
                    previous_e = (float)(
                        sum(get_previous_i['Rating'].loc[get_previous_i['MovieID'] == item]) - which_rate) / float(
                        i - 1)
                    sequence_info[i].append([which_item, which_user, previous_e, which_rate])
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.g = g
        self.Bu = Bu
        self.Bp = Bp
        save_preprocess = open('preprocess.pickle', 'wb')
        pickle.dump((self.user_ids, self.item_ids, self.g, self.Bu, self.Bp, sequence_info), save_preprocess)
        save_preprocess.close()
        print('finish preprocessing...')
        return sequence_info

    def get_weight(self, shape, lambda1, mean, stddev):
        var = tf.Variable(tf.random_normal(shape, mean=mean, stddev=stddev), dtype=tf.float32)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
        return var

    def train_model(self):

        """
        train the model using GD algorithm
        :param sequence_info: {i:[product,user,e_rate,rate]}
        :return: None save the model data in the HIALF.model
        """
        sess = tf.Session()
        k = 5
        sequence_info = self.sequence_info
        User_n = max(self.user_ids)
        Item_n = max(self.item_ids)
        # r_ = g + bu + bp + X*Y.T

        # global rating bias
        G = tf.constant(self.g, dtype='float32')

        # bu,bp are parameters of user,item bias
        tf.reshape(self.Bu, [User_n, 1])
        tf.reshape(self.Bp, [Item_n, 1])
        # bu = tf.get_variable(name='bu',shape=[User_n,1],initializer=tf.initialize_variables()  self.Bu,regularizer=tf.contrib.layers.l2_regularizer(0.01))

        bu = tf.Variable(self.Bu, name='bu', dtype='float32', expected_shape=[User_n, 1])
        bp = tf.Variable(self.Bp, name='bp', dtype='float32', expected_shape=[Item_n, 1])
        # bu_onehot bp_onehot are placeholder which tells which user and product are used to train
        bu_OneHot = tf.placeholder(dtype='float32', shape=[User_n, 1], name='bu_OneHot')
        bp_OneHot = tf.placeholder(dtype='float32', shape=[Item_n, 1], name='bp_OneHot')

        # Xu,Yp are matrix describes the user and item latent factor
        Xu = tf.Variable(tf.random_normal(shape=[User_n, k], mean=0.1, stddev=0.001), name='User', dtype='float32')
        Yp = tf.Variable(tf.random_normal(shape=[Item_n, k], mean=0.1, stddev=0.001), name='Item', dtype='float32')

        # Au is the vector describing how easily a user may be affected
        Au = tf.Variable(tf.random_normal(shape=[User_n, 1], mean=0.1, stddev=0.001), name='Au', dtype='float32')

        # g + bu + bp + X*Y.T + Au*f(|i-1|)*B(e-q)
        # q = g + bp + X*Y.T
        # bu + q + fB
        R_predict_first_part = tf.add(G, tf.matmul(tf.transpose(bp_OneHot), bp))  # g + bp
        R_predict_second_part = tf.matmul(tf.matmul(tf.transpose(bu_OneHot), tf.matmul(Xu, tf.transpose(Yp))),
                                          bp_OneHot)  # X*Y.T
        Q_predict = tf.add(R_predict_first_part, R_predict_second_part)
        Bu_predict = tf.matmul(tf.transpose(bu_OneHot), bu)

        # f(|i-1|)
        size_rating = tf.placeholder(dtype='float32', shape=[1], name='size_rating')
        a = tf.Variable(tf.random_normal([1], mean=0.1, stddev=0.001), name='a')
        b = tf.Variable(tf.random_normal([1], mean=0.1, stddev=0.001), name='b')
        f_i = tf.multiply((1.0 + tf.exp(tf.multiply(-size_rating, b))) - 0.5, a)

        # B
        e_l = np.arange(-4, 4.5, 0.5)
        e_pi = tf.placeholder(dtype='float32', name='e_pi')
        # print((Q_predict.shape))
        delta_e_q = tf.subtract(e_pi, Q_predict)
        El = tf.constant(e_l, dtype='float32')
        Vl = tf.Variable(tf.random_normal([len(e_l)], mean=0.1, stddev=0.001), name='Vl')
        Wx = tf.exp(-2.0 * tf.square(delta_e_q - El))
        Bx_up = tf.reduce_sum(tf.multiply(Wx, Vl))
        Bx_down = tf.reduce_sum(Wx)
        Bx = tf.div(Bx_up, Bx_down)

        # F_B = Au*f(|i-1|)B
        F_B = tf.matmul(tf.transpose(Au), bu_OneHot)
        F_B = tf.multiply(tf.multiply(f_i, Bx), F_B)
        F_B = tf.multiply(f_i, Bx)
        # F_B = Bx
        R_predict = tf.add(tf.add(Bu_predict, Q_predict), F_B)
        # R_predict = tf.add(Bu_predict, Q_predict)
        # R_predict = 1.0 / (1.0+tf.exp(-R_predict))

        R = tf.placeholder(dtype='float32', shape=[1], name='R')
        Loss = tf.reduce_sum(tf.square(tf.subtract(R, R_predict)))

        # add regularized parameters to the loss function
        for key in tf.get_default_graph().get_all_collection_keys():
            if key == 'trainable_variables':
                for element_v in tf.get_collection(key):
                    # print(element_v)
                    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0001)(element_v))


        tf.add_to_collection('losses', Loss)

        regularization_loss = tf.add_n(tf.get_collection('losses'))
        lr = 0.005
        lr = tf.Variable(float(lr), trainable=False, dtype='float32')
        lr_decay = lr.assign(lr * 0.9)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(regularization_loss)

        # get the variable information in the GraphKeys
        # for key in tf.get_default_graph().get_all_collection_keys():
        #     print(key)
        #     for ele in tf.get_collection(key):
        #         print(ele)
        # with tf.Session() as sess:


        # sess = tf_debug.LocalCLIDebugWrapperSession(sess,ui_type="readline")
        # sess.add_tensor_filter("has_inf_or_nan",tf_debug.has_inf_or_nan)

        iter_cnt = 100

        # training number

        which_user = np.zeros([User_n, 1])
        which_item = np.zeros([Item_n, 1])
        # tf.train.Saver()
        #saver = tf.train.Saver(max_to_keep=5)
        ALL_VARS = tf.trainable_variables()
        save_v = [v for v in ALL_VARS]


        log_file = open('log.txt', 'w')
        min_loss = 10000
        if os.path.exists('log.txt'):
            #print('exist')
            sess.run(tf.global_variables_initializer())
            get_data = tf.train.import_meta_graph('Model/model-8.meta')
            get_data.restore(sess, 'Model/model-8')
            print('successfully reload trained model')
        saver = tf.train.Saver(save_v)

        sequence_key = sequence_info.keys()
        for iter_ in range(iter_cnt):
            total_loss = 0
            count = 0
            seq_cnt = 0
            for i in sequence_key:
                seq_cnt += 1
                # determine how many historical data are used to training
                if seq_cnt > 100:
                    break
                to_training = sequence_info[i]
                # count = 0
                print('\ntraining on the %dth historical data...' % i)
                for j in range(len(to_training)):
                    # pair [product,user,previous_rating, rating]
                    # print(user_product_pair)
                    # print(to_training)
                    user_product_pair = to_training[j]
                    self.view_bar('finish ', j + 1, len(to_training))
                    which_user[user_product_pair[1] - 1] = 1
                    which_item[user_product_pair[0] - 1] = 1
                    # print(user_product_pair[3])
                    # print(1.0/(1.0+math.exp(-user_product_pair[3])))
                    _, l = sess.run([optimizer, regularization_loss],
                                    feed_dict={bu_OneHot: which_user, bp_OneHot: which_item, size_rating: [i - 1],
                                               e_pi: [user_product_pair[2]], R: [user_product_pair[3]]})
                    # R_truth = 1.0/(1.0+math.exp(-user_product_pair[3]))
                    # R_truth = user_product_pair[3]
                    # _, l = sess.run([optimizer, Loss], feed_dict={bu_OneHot: which_user, bp_OneHot: which_item,R: [R_truth] })
                    which_user[user_product_pair[1] - 1] = 0
                    which_item[user_product_pair[0] - 1] = 0
                    total_loss += l
                    count += 1
                    #print('loss is{}'.format(l))
                print('training loss is {}'.format(total_loss/count))

            if iter_ % 2 == 0:
                log_file.write(str(iter_) + 'loss is :' + str(total_loss) + '\n')
                if (total_loss / count < min_loss):
                    min_loss = total_loss / count
                saver.save(sess, 'Model/model_2', global_step=iter_)
                print('\n****************{}th training iterations loss is {}****************'.format(iter_,
                                                                                                     total_loss / (
                                                                                                         float)(count)))
                sess.run(lr_decay)
        log_file.close()
        sess.close()


    def predict(self, user_id, item_id):
        p = np.mat(self.X.ix[user_id].values)
        q = np.mat(self.Y.ix[item_id].values).T
        r = (p * q).sum()
        logit = 1.0 / (1 + math.exp(-r))
        return logit

    # evaluation part
    # use the trained parameters to evaluate the result of the algorithm
    # graph is stored in meta file
    # weights are stored in data file
    def evaluate(self):
        sequence_info = self.sequence_info
        sess = tf.Session()
        k = 5

        User_n = max(self.user_ids)
        Item_n = max(self.item_ids)
        # r_ = g + bu + bp + X*Y.T

        # global rating bias
        G = tf.constant(self.g, dtype='float32')

        # bu,bp are parameters of user,item bias
        tf.reshape(self.Bu, [User_n, 1])
        tf.reshape(self.Bp, [Item_n, 1])
        # bu = tf.get_variable(name='bu',shape=[User_n,1],initializer=tf.initialize_variables()  self.Bu,regularizer=tf.contrib.layers.l2_regularizer(0.01))

        bu = tf.Variable(self.Bu, name='bu', dtype='float32', expected_shape=[User_n, 1])
        bp = tf.Variable(self.Bp, name='bp', dtype='float32', expected_shape=[Item_n, 1])

        # bu_onehot bp_onehot are placeholder which tells which user and product are used to train
        bu_OneHot = tf.placeholder(dtype='float32', shape=[User_n, 1], name='bu_OneHot')
        bp_OneHot = tf.placeholder(dtype='float32', shape=[Item_n, 1], name='bp_OneHot')

        # Xu,Yp are matrix describes the user and item latent factor
        Xu = tf.Variable(tf.random_normal(shape=[User_n, k], mean=0.1, stddev=0.001), name='User', dtype='float32')
        Yp = tf.Variable(tf.random_normal(shape=[Item_n, k], mean=0.1, stddev=0.001), name='Item', dtype='float32')

        # Au is the vector describing how easily a user may be affected
        Au = tf.Variable(tf.random_normal(shape=[User_n, 1], mean=0.1, stddev=0.001), name='Au', dtype='float32')

        # g + bu + bp + X*Y.T + Au*f(|i-1|)*B(e-q)
        # q = g + bp + X*Y.T
        # bu + q + fB
        R_predict_first_part = tf.add(G, tf.matmul(tf.transpose(bp_OneHot), bp))  # g + bp
        R_predict_second_part = tf.matmul(tf.matmul(tf.transpose(bu_OneHot), tf.matmul(Xu, tf.transpose(Yp))),
                                          bp_OneHot)  # X*Y.T
        Q_predict = tf.add(R_predict_first_part, R_predict_second_part)
        Bu_predict = tf.matmul(tf.transpose(bu_OneHot), bu)

        # f(|i-1|)
        size_rating = tf.placeholder(dtype='float32', shape=[1], name='size_rating')
        a = tf.Variable(tf.random_normal([1], mean=0.1, stddev=0.001), name='a')
        b = tf.Variable(tf.random_normal([1], mean=0.1, stddev=0.001), name='b')
        f_i = tf.multiply((1.0 + tf.exp(tf.multiply(-size_rating, b))) - 0.5, a)

        # B
        e_l = np.arange(-4, 4.5, 0.5)
        e_pi = tf.placeholder(dtype='float32', name='e_pi')
        # print((Q_predict.shape))
        delta_e_q = tf.subtract(e_pi, Q_predict)
        El = tf.constant(e_l, dtype='float32')
        Vl = tf.Variable(tf.random_normal([len(e_l)], mean=0.1, stddev=0.001), name='Vl')
        Wx = tf.exp(-2.0 * tf.square(delta_e_q - El))
        Bx_up = tf.reduce_sum(tf.multiply(Wx, Vl))
        Bx_down = tf.reduce_sum(Wx)
        Bx = tf.div(Bx_up, Bx_down)

        # F_B = Au*f(|i-1|)B
        F_B = tf.matmul(tf.transpose(Au), bu_OneHot)
        F_B = tf.multiply(tf.multiply(f_i, Bx), F_B)
        F_B = tf.multiply(f_i, Bx)
        # F_B = Bx
        R_predict = tf.add(tf.add(Bu_predict, Q_predict), F_B)
        # R_predict = tf.add(Bu_predict, Q_predict)
        # R_predict = 1.0 / (1.0+tf.exp(-R_predict))

        R = tf.placeholder(dtype='float32', shape=[1], name='R')
        Loss = tf.square(tf.subtract(R, R_predict))

        which_user = np.zeros([User_n, 1])
        which_item = np.zeros([Item_n, 1])
        sess.run(tf.global_variables_initializer())
        get_data = tf.train.import_meta_graph('Model/model-8.meta')
        print('sucessfully reload graph')
        # all_v = tf.trainable_variables()
        # for v in all_v:
        #     print(v.name)
        # get_data.restore(sess, tf.train.latest_checkpoint('Model/'))
        get_data.restore(sess,'Model/model-8')
        print('successfullt restore data')
        evaluate_file = open('evaluate.txt','w')
        sequence_key = sequence_info.keys()
        seq_cnt = 0
        for i in sequence_key:
            seq_cnt+=1
            if seq_cnt<=100:
                continue
            elif seq_cnt<105:
                print('\nevaluate on %d data'%seq_cnt)
                to_training = sequence_info[i]
                Total_loss = 0
                cnt = 0
                for j in range(len(to_training)):
                    # pair [product,user,previous_rating, rating]
                    # print(user_product_pair)
                    # print(to_training)
                    cnt+=1
                    user_product_pair = to_training[j]
                    self.view_bar('finish ', j + 1, len(to_training))
                    which_user[user_product_pair[1] - 1] = 1
                    which_item[user_product_pair[0] - 1] = 1
                    l = sess.run([Loss],feed_dict={bu_OneHot: which_user, bp_OneHot: which_item, size_rating: [i-1],e_pi: [user_product_pair[2]], R: [user_product_pair[3]]})
                    which_user[user_product_pair[1] - 1] = 0
                    which_item[user_product_pair[0] - 1] = 0
                    Total_loss+=l[0]

                Mse = math.sqrt(Total_loss/cnt)
                evaluate_file.write(str(i)+'th data, MSE is '+str(Mse)+'\n')
                print('MSE is {}'.format(Mse))
        evaluate_file.close()
        sess.close()
        return


    def test(self):
        top_n = 5
        assert (os.path.exists('HIALF.model'))
        model = open('HIALF.model', 'rb')
        self.X, self.Y = pickle.load(model)
        file_path = './lfm/data/ratings.csv'
        frame = pd.read_csv(file_path)
        user_ids = set(frame['UserID'])
        item_ids = set(frame['MovieID'])
        for user_id in user_ids:
            user_item_ids = set(frame[frame['UserID'] == user_id]['MovieID'])
            other_item_ids = item_ids ^ user_item_ids
            interest_list = [self.predict(user_id, item_id) for item_id in other_item_ids]
            candidates = sorted(zip(list(other_item_ids), interest_list), key=lambda x: x[1], reverse=True)
            print('for user {} recommentations are: {}'.format(user_id, candidates[:top_n]))


if __name__ == "__main__":
    HIALF_MODEL = HIALF()
    sequence_info = None
    if not os.path.exists('preprocess.pickle'):
        sequence_info = self.get_sequence_info()
    else:
        f = open('preprocess.pickle', 'rb')
        HIALF_MODEL.user_ids, HIALF_MODEL.item_ids, HIALF_MODEL.g, HIALF_MODEL.Bu, HIALF_MODEL.Bp, HIALF_MODEL.sequence_info = pickle.load(f)
    ifTrain = True
    ifEvaluate = False
    ifTest = False

    if ifTrain:
        HIALF_MODEL.train_model()
    elif ifEvaluate:
        HIALF_MODEL.evaluate()
    else:
        HIALF_MODEL.test()