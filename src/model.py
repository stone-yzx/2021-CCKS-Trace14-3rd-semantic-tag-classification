#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : NeXtVLADModelLF.py
# @Author: stoneye
# @Date  : 2020/06/24
# @Desc  :
# @license : Copyright(C), Zhenxu Ye
# @Contact : yezhenxu1992@163.com
# @Software : PyCharm

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim
import tensorflow.contrib as tc

# np.random.seed(5)
# tf.set_random_seed(-1)


class NeXtVLAD():
    def __init__(self, feature_size, max_frames, cluster_size, is_training=True, expansion=2, groups=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.cluster_size = cluster_size
        self.expansion = expansion
        self.groups = groups

    def forward(self, input, mask=None):
        input = slim.fully_connected(input, self.expansion * self.feature_size, activation_fn=None,
                                     weights_initializer=slim.variance_scaling_initializer())

        attention = slim.fully_connected(input, self.groups, activation_fn=tf.nn.sigmoid,
                                         weights_initializer=slim.variance_scaling_initializer())
        if mask is not None:
            attention = tf.multiply(attention, tf.expand_dims(mask, -1))
        attention = tf.reshape(attention, [-1, self.max_frames * self.groups, 1])
        tf.summary.histogram("sigmoid_attention", attention)
        feature_size = self.expansion * self.feature_size // self.groups

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.expansion * self.feature_size, self.groups * self.cluster_size],
                                          initializer=slim.variance_scaling_initializer()
                                          )

        # tf.summary.histogram("cluster_weights", cluster_weights)
        reshaped_input = tf.reshape(input, [-1, self.expansion * self.feature_size])
        activation = tf.matmul(reshaped_input, cluster_weights)

        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="cluster_bn",
            fused=False)

        activation = tf.reshape(activation, [-1, self.max_frames * self.groups, self.cluster_size])
        activation = tf.nn.softmax(activation, axis=-1)
        activation = tf.multiply(activation, attention)
        # tf.summary.histogram("cluster_output", activation)
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, feature_size, self.cluster_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(input, [-1, self.max_frames * self.groups, feature_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, self.cluster_size * feature_size])
        vlad = slim.batch_norm(vlad,
                               center=True,
                               scale=True,
                               is_training=self.is_training,
                               scope="vlad_bn",
                               fused=False)

        return vlad


class tag_model(object):

    def __init__(self, args):


        self.task_type=args.task_type # cate tag cate_and_tag
        self.ad_strength = args.ad_strength
        self.lr = args.lr
        self.title_max_len=args.title_max_length 
        self.asr_orc_max_len=args.asr_orc_max_length

        self.cate1_num = args.cate1_num
        self.cate2_num = args.cate2_num

        self.tag_num = args.tag_num
        
        self.pre_train_emb_path = args.pre_trained_emb_path  # 预训练好的词向量路径
        self.embedding_dim = args.embedding_dim  # 词向量维度
        self.hidden_size = args.hidden_size  # 隐藏层的数目
        
        self.max_frames_rgb = args.rgb_frames
        self.max_frames_audio = args.audio_frames

        self.batch_size=args.batch_size
        self.train_sample_nums=args.train_sample_nums

        # 加载预先训练的embeddding
        word_embeddings = self._init_vocab_and_emb(word_vocab=args.word_vocab,pre_train_emb_path=args.pre_trained_emb_path)
        self.word_embeddings = tf.Variable(word_embeddings, name='word_embeddings', dtype=tf.float32)
        
        #初始化 占位符
        self._init_placeholder()
        #构建计算图
        self._build_graph()
        self.global_step = tf.Variable(0, trainable=False, name='global_step')  # 系统维护的一个全局变量，每过一个step（mini-batch），自动+1
        self.cur_lr = tf.train.cosine_decay_restarts(self.lr, self.global_step, 4*self.train_sample_nums//self.batch_size, t_mul=1.5, m_mul=0.5, alpha=1e-7)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.cur_lr).minimize(self.total_loss,global_step=self.global_step)  # 采用Adam优化器进行梯度下降的参数更新迭代
    
    def _init_vocab_and_emb(self, word_vocab, pre_train_emb_path):

        word_id_dict = self._get_word_id_dict(word_vocab)
        print("len(word_id_dict): {}".format(len(word_id_dict)))

        word_embeddings = self._load_pretrained_embedding(word_id_dict=word_id_dict,
                                                          pre_train_emb_path=pre_train_emb_path)

        return word_embeddings

    def exponential_decay_with_warmup(self, warmup_step, learning_rate_base, global_step, learning_rate_step,
                                      learning_rate_decay, staircase=False):
       
        with tf.name_scope("exponential_decay_with_warmup"):
            linear_increase = learning_rate_base * tf.cast(global_step / warmup_step, tf.float32)
            exponential_decay = tf.train.exponential_decay(learning_rate_base,
                                                           global_step - warmup_step,
                                                           learning_rate_step,
                                                           learning_rate_decay,
                                                           staircase=staircase)
            learning_rate = tf.cond(global_step <= warmup_step,
                                    lambda: linear_increase,
                                    lambda: exponential_decay)
            return learning_rate

    def _get_word_id_dict(self, in_file):
        word_id_dict = {}
        with open(in_file, 'r')as fr:
            for line in fr:
                line_split = line.rstrip('\n').split('\t')
                if len(line_split) != 2:
                    continue
                token_text = line_split[0]
                token_id = int(line_split[1])
                if token_text not in word_id_dict:
                    word_id_dict[token_text] = token_id
        print("len(word_id_dict): ", len(word_id_dict))
        return word_id_dict



    def _init_placeholder(self):

        self.title_id_int_list = tf.placeholder(tf.int32, [None, self.title_max_len],name="title_id_int_list")  # [batch,max_len]
        self.title_sequence_length = tf.cast(tf.reduce_sum(tf.sign(self.title_id_int_list), axis=1), tf.int32)  # [batch,]

        self.asr_id_int_list = tf.placeholder(tf.int32, [None, self.asr_orc_max_len], name="asr_id_int_list")  # [batch,max_len]
        self.asr_sequence_length = tf.cast(tf.reduce_sum(tf.sign(self.asr_id_int_list), axis=1),tf.int32)  # [batch,]

        self.ocr_id_int_list = tf.placeholder(tf.int32, [None, self.asr_orc_max_len],name="ocr_id_int_list")  # [batch,max_len]
        self.ocr_sequence_length = tf.cast(tf.reduce_sum(tf.sign(self.ocr_id_int_list), axis=1), tf.int32)  # [batch,]

        self.tag_multi_gt_labels = tf.placeholder(tf.float32, [None, self.tag_num], name="tag_multi_gt_labels")
        self.cate1_gt_labels = tf.placeholder(tf.float32, [None, self.cate1_num], name="cate1_gt_labels")
        self.cate2_gt_labels = tf.placeholder(tf.float32, [None, self.cate2_num], name="cate2_gt_labels")
        
        
        #帧特征
        self.input_rgb_fea = tf.placeholder(tf.float32, shape=[None, self.max_frames_rgb, 512], name='rgb_fea')
        self.input_audio_fea = tf.placeholder(tf.float32, shape=[None, self.max_frames_audio, 128], name='audio_fea')

        self.rgb_fea_true_frame = tf.placeholder(tf.int32, shape=[None, ])
        self.audio_fea_true_frame = tf.placeholder(tf.int32, shape=[None, ])
    
        # 音频特征
        # self.input_audio_fea = tf.placeholder(tf.float32, shape=[None, self.max_frames_audio, 128], name='audio_vggish')
       
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")





    def _load_pretrained_embedding(self, word_id_dict, pre_train_emb_path, embedding_dim=200):
        trained_embedding = {}
        word_emb = 0
        with open(pre_train_emb_path, 'r')as fr:
            num = 0
            for line in fr:
                num += 1
                if num == 1:
                    continue
                contents = line.rstrip('\n').split()
                if len(contents) != 201:
                    print("error emb: ", len(contents))
                    continue
                trained_embedding[contents[0]] = list(map(float, contents[1:]))
        word_embeddings = np.random.standard_normal([len(word_id_dict), embedding_dim])
        # 加载预先训练的词向量
        for token, token_id in word_id_dict.items():
            if token in trained_embedding:
                word_emb += 1
                word_embeddings[token_id] = trained_embedding[token]
        print("word_num_emb_pretrain: ", word_emb)
        word_embeddings[0] = [0.0] * embedding_dim  # padId   0
        word_embeddings[1] = [0.1] * embedding_dim  # unkId   1
        word_embeddings = word_embeddings.astype(np.float32)
        return word_embeddings

    def _BiLSTM_feature(self, embedding_descript, hidden_size, des_sequence_length, dtype=tf.float32, reuse=None):
        '''
        抽取BiLSTM_feature,可以加入该特征
        :param embedding_descript:
        :param hidden_size:
        :param des_sequence_length:
        :param dtype:
        :return:
        '''

        with tf.variable_scope('Bilstm_feature', reuse=reuse):
            with tf.variable_scope("BiLSTM_feature"):
                self.lstm_fw_cell = rnn.BasicLSTMCell(hidden_size)  # hidden_size=128
                self.lstm_bw_cell = rnn.BasicLSTMCell(hidden_size)  # hidden_size=128

                self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(
                    parallel_iterations=520,
                    cell_fw=self.lstm_fw_cell,
                    cell_bw=self.lstm_bw_cell,
                    inputs=embedding_descript,  # [batch,max_len,emb_size=128]
                    sequence_length=des_sequence_length,
                    time_major=False,
                    dtype=dtype
                )

                self.output_fw, self.output_bw = self.outputs
                self.states_fw, self.states_bw = self.states
                self.c_fw, self.h_fw = self.states_fw
                self.c_bw, self.h_bw = self.states_bw
                self.BiLSTM_final_h = tf.concat([self.h_fw, self.h_bw], -1)  # [batch,2h]
                self.BiLSTM_hiddens = tf.concat([self.output_fw, self.output_bw], -1)  # [batch,max_len,2*hidden_size]

            return self.BiLSTM_final_h

    def _cnn_feature(self, embedding_descript,
                     embedding_size, filter_sizes, num_filters, reuse=None, dropout_keep_prob=1.0):
        '''
        抽取app描述文本CNN的特征
        :param embedding_descript:
        :param embedding_size:
        :param filter_sizes:
        :param num_filters:
        :param dropout_keep_prob:
        :return:
        '''

        # embedding_descript [batch,max_len,emb_size]

        with tf.variable_scope('CNN_feature', reuse=reuse):
            self.embedded_descript_expanded = tf.expand_dims(embedding_descript, -1)  # [batch,max_len,emb_size,1]
            self.pooled_outputs_descript = []

            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size) as scope:
                    # Convolution Layer
                    self.filter_shape = [filter_size, embedding_size, 1, num_filters]


                    self.W = tf.get_variable("W", shape=self.filter_shape,
                                             initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
                    self.b = tf.get_variable("b", shape=[num_filters], initializer=tf.constant_initializer(0.1),
                                             dtype=tf.float32)
                    self.embedded_descript_expanded = tf.cast(self.embedded_descript_expanded, dtype=tf.float32)
                    self.conv_descript = tf.nn.conv2d(self.embedded_descript_expanded,
                                                      self.W,
                                                      strides=[1, 1, 1, 1],
                                                      padding="VALID",
                                                      name="conv_descript_feature")
                    # Apply nonlinearity
                    self.app_descript_feature = tf.nn.relu(tf.nn.bias_add(self.conv_descript, self.b), name="relu")
                    # [batch,max_len-filter_size+1,1,num_filters]
                    # Maxpooling over the outputs
                    self.pooled_app_descript_feature = tf.nn.max_pool(self.app_descript_feature,
                                                                      ksize=[1,
                                                                             self.app_descript_feature.get_shape().as_list()[
                                                                                 1], 1, 1],
                                                                      strides=[1, 1, 1, 1],
                                                                      padding='VALID',
                                                                      name="pool_q1")
                    # [batch,1,num_filters]
                    # print "pooled_app_descript_feature: ",self.pooled_app_descript_feature.get_shape()
                    self.pooled_outputs_descript.append(self.pooled_app_descript_feature)
            # Combine all the pooled features
            with tf.variable_scope("combine") as scope:
                num_filters_total = num_filters * len(filter_sizes)
                self.h_pool_descript = tf.concat(self.pooled_outputs_descript, 3)  # [batch,1, num_filters_total]
                # print "h_pool_descript: ",self.h_pool_descript.get_shape()
                self.h_pool_flat_descript = tf.reshape(self.h_pool_descript, [-1, num_filters_total],
                                                       name="descript_flat")
                # [batch,num_filters_total]
            # Add dropout
            with tf.variable_scope("dropout") as scope:
                self.h_drop_app_descript = tf.nn.dropout(self.h_pool_flat_descript, dropout_keep_prob, name="h_drop")

            return self.h_drop_app_descript

    def _SE_module(self, activation, gating_reduction=8, name_scape=''):

        with tf.variable_scope(name_scape):
            hidden1_size = activation.get_shape().as_list()[1]
            gating_weights_1 = tf.get_variable("gating_weights_1",
                                               [hidden1_size, hidden1_size // gating_reduction],
                                               initializer=slim.variance_scaling_initializer())

            gates = tf.matmul(activation, gating_weights_1)

            gates = slim.batch_norm(
                gates,
                center=True,
                scale=True,
                is_training=self.is_training,
                activation_fn=slim.nn.relu,
                scope="gating_bn")

            gating_weights_2 = tf.get_variable("gating_weights_2",
                                               [hidden1_size // gating_reduction, hidden1_size],
                                               initializer=slim.variance_scaling_initializer())
            gates = tf.matmul(gates, gating_weights_2)

            gates = tf.sigmoid(gates)

            activation = tf.multiply(activation, gates)

            return activation

    def neXtvlad_model(self, is_training, fea_input, dropout_keep_prob, fea_type, name_scape,mask=None):

        with tf.variable_scope(name_scape):

            if fea_type == 'rgb':
                max_frames = self.max_frames_rgb
                fea_size = 1024
                cluster_size = 256
                groups = 8

            elif fea_type == 'audio':
                max_frames = self.max_frames_audio
                fea_size = 128
                cluster_size = 128
                groups = 8 // 2
                
            elif fea_type == 'text':
                max_frames = self.asr_orc_max_len
                fea_size = 400
                cluster_size = 200
                groups = 8 // 2
            expansion = 2
            
            
            hidden1_size = 2048
            gating_reduction = 8
            nextvlad_obj = NeXtVLAD(fea_size, max_frames, cluster_size, is_training, groups=groups, expansion=expansion)
            vlad = nextvlad_obj.forward(fea_input, mask=mask)
            vlad = slim.dropout(vlad, keep_prob=dropout_keep_prob, is_training=is_training, scope="vlad_dropout")
            vlad_dim = vlad.get_shape().as_list()[1]
            print("VLAD dimension", vlad_dim)
            hidden1_weights = tf.get_variable("hidden1_weights",
                                              [vlad_dim, hidden1_size],
                                              initializer=slim.variance_scaling_initializer())
            activation = tf.matmul(vlad, hidden1_weights)
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="hidden1_bn",
                fused=False)
            gating_weights_1 = tf.get_variable("gating_weights_1",
                                               [hidden1_size, hidden1_size // gating_reduction],
                                               initializer=slim.variance_scaling_initializer())
            gates = tf.matmul(activation, gating_weights_1)
            gates = slim.batch_norm(
                gates,
                center=True,
                scale=True,
                is_training=is_training,
                activation_fn=slim.nn.relu,
                scope="gating_bn")
            gating_weights_2 = tf.get_variable("gating_weights_2",
                                               [hidden1_size // gating_reduction, hidden1_size],
                                               initializer=slim.variance_scaling_initializer())
            gates = tf.matmul(gates, gating_weights_2)
            gates = tf.sigmoid(gates)
            activation = tf.multiply(activation, gates)
            return activation

    

    def scale_l2(self, x, norm_length):
        # shape(x) = (batch, num_timesteps, d)
        # Divide x by max(abs(x)) for a numerically stable L2 norm.
        # 2norm(x) = a * 2norm(x/a)
        # Scale over the full sequence, dims (1, 2)
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit

    def add_perturbation(self, embedded, loss, norm_length=5):
        """Adds gradient to embedding and recomputes classification loss."""
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        perturb = self.scale_l2(grad, norm_length=norm_length)  # epsilon-->norm_length
        return embedded + perturb

    def cate_mlp_label_layer(self, fea_vector, dropout_keep_prob, cate_num, ml_tag, name_scope):

        with tf.variable_scope(name_scope):
            #########################
            ## define HMC

            # global
            concate_features_SE_drop = fea_vector

            HMC_AG_1 = slim.fully_connected(concate_features_SE_drop, num_outputs=1024, scope="HMC_WG_1")
            HMC_AG_1_drop = tf.nn.dropout(HMC_AG_1, dropout_keep_prob)
            HMC_AG_1_drop_x = tf.concat([HMC_AG_1_drop, concate_features_SE_drop], -1)

            HMC_AG_2 = slim.fully_connected(HMC_AG_1_drop_x, num_outputs=1024, scope="HMC_WG_2")
            HMC_AG_2_drop = tf.nn.dropout(HMC_AG_2, dropout_keep_prob)
            HMC_AG_2_drop_x = tf.concat([HMC_AG_2_drop, HMC_AG_1_drop_x], -1)

            HMC_AG_3 = slim.fully_connected(HMC_AG_2_drop_x, num_outputs=512, scope="HMC_WG_3")
            HMC_AG_3_drop = tf.nn.dropout(HMC_AG_3, dropout_keep_prob)
            HMC_AG_3_drop_x = tf.concat([HMC_AG_3_drop, HMC_AG_2_drop_x], -1)
            HMC_PG_scores = slim.fully_connected(HMC_AG_3_drop_x, num_outputs=cate_num, activation_fn=None,
                                                 normalizer_fn=None,
                                                 biases_initializer=tf.constant_initializer(0.0), scope="HMC_pg")
            HMC_PG_prob = tf.nn.softmax(HMC_PG_scores, name='HMC_pg_softmax')

            # local tag
            HMC_AL_3 = slim.fully_connected(HMC_AG_3_drop, num_outputs=512, scope="HMC_WT_3")
            HMC_AL_3_drop = tf.nn.dropout(HMC_AL_3, dropout_keep_prob)
            HMC_PL_3_scores = slim.fully_connected(HMC_AL_3_drop, num_outputs=cate_num, activation_fn=None,
                                                   normalizer_fn=None,
                                                   biases_initializer=tf.constant_initializer(0.0), scope="HMC_pl_3")
            HMC_PL_3_prob = tf.nn.softmax(HMC_PL_3_scores, name='HMC_pl_3_softmax')

            HMC_PF_prob = 0.5 * HMC_PL_3_prob + 0.5 * HMC_PG_prob  # [:, :tag_num]

            PL_3_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=HMC_PL_3_scores, labels=ml_tag, name="HMC_PL_3_loss"))
            PG_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=HMC_PG_scores, labels=ml_tag, name="HMC_PG_loss"))

            return PL_3_loss + PG_loss, HMC_PF_prob



    def tag_mlp_label_layer(self, fea_vector, dropout_keep_prob, tag_num, ml_tag, name_scope):

        with tf.variable_scope(name_scope):
            #########################
            ## define HMC

            # global
            concate_features_SE_drop = fea_vector

            HMC_AG_1 = slim.fully_connected(concate_features_SE_drop, num_outputs=1024, scope="HMC_WG_1")
            HMC_AG_1_drop = tf.nn.dropout(HMC_AG_1, dropout_keep_prob)
            HMC_AG_1_drop_x = tf.concat([HMC_AG_1_drop, concate_features_SE_drop], -1)

            HMC_AG_2 = slim.fully_connected(HMC_AG_1_drop_x, num_outputs=1024, scope="HMC_WG_2")
            HMC_AG_2_drop = tf.nn.dropout(HMC_AG_2, dropout_keep_prob)
            HMC_AG_2_drop_x = tf.concat([HMC_AG_2_drop, HMC_AG_1_drop_x], -1)

            HMC_AG_3 = slim.fully_connected(HMC_AG_2_drop_x, num_outputs=512, scope="HMC_WG_3")
            HMC_AG_3_drop = tf.nn.dropout(HMC_AG_3, dropout_keep_prob)
            HMC_AG_3_drop_x = tf.concat([HMC_AG_3_drop, HMC_AG_2_drop_x], -1)
            HMC_PG_scores = slim.fully_connected(HMC_AG_3_drop_x, num_outputs=tag_num, activation_fn=None,
                                                 normalizer_fn=None,
                                                 biases_initializer=tf.constant_initializer(0.0), scope="HMC_pg")
            HMC_PG_prob = tf.nn.sigmoid(HMC_PG_scores, name='HMC_pg_sigmoid')

            # local tag
            HMC_AL_3 = slim.fully_connected(HMC_AG_3_drop, num_outputs=512, scope="HMC_WT_3")
            HMC_AL_3_drop = tf.nn.dropout(HMC_AL_3, dropout_keep_prob)
            HMC_PL_3_scores = slim.fully_connected(HMC_AL_3_drop, num_outputs=tag_num, activation_fn=None,
                                                   normalizer_fn=None,
                                                   biases_initializer=tf.constant_initializer(0.0), scope="HMC_pl_3")
            HMC_PL_3_prob = tf.nn.sigmoid(HMC_PL_3_scores, name='HMC_pl_3_sigmoid')

            HMC_PF_prob = 0.5 * HMC_PL_3_prob + 0.5 * HMC_PG_prob  # [:, :tag_num]

            PL_3_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=HMC_PL_3_scores, labels=ml_tag, name="HMC_PL_3_loss"))
            PG_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=HMC_PG_scores, labels=ml_tag, name="HMC_PG_loss"))

            return PL_3_loss + PG_loss, HMC_PF_prob

    def cal_loss(self, rgb_fea,audio_fea, text_emb_fea, reuse=None):

        with tf.variable_scope("cl_loss_from_emb", reuse=reuse):
            embedded_title, embedded_asr, embedded_ocr = text_emb_fea

            #rgb_eff
            rgb_nextvlad_fea = self.neXtvlad_model(is_training=self.is_training,
                                                   fea_input=rgb_fea,
                                                   dropout_keep_prob=self.dropout_keep_prob,
                                                   fea_type='rgb',
                                                   name_scape='rgb_nextvlad_fea',
                                                   mask = tf.sequence_mask(self.rgb_fea_true_frame, self.max_frames_rgb, dtype=tf.float32))

            audio_nextvlad_fea = self.neXtvlad_model(is_training=self.is_training,
                                                   fea_input=audio_fea,
                                                   dropout_keep_prob=self.dropout_keep_prob,
                                                   fea_type='audio',
                                                   name_scape='audio_nextvlad_fea',
                                                   mask = tf.sequence_mask(self.audio_fea_true_frame, self.max_frames_audio, dtype=tf.float32))



            asr_nextvlad_fea = self.neXtvlad_model(is_training=self.is_training,
                                                   fea_input=embedded_asr,
                                                   dropout_keep_prob=self.dropout_keep_prob,
                                                   fea_type='text',
                                                   name_scape='asr_nextvlad_fea',
                                                   mask = tf.sequence_mask(self.asr_sequence_length, self.asr_orc_max_len, dtype=tf.float32))

            ocr_nextvlad_fea = self.neXtvlad_model(is_training=self.is_training,
                                                   fea_input=embedded_ocr,
                                                   dropout_keep_prob=self.dropout_keep_prob,
                                                   fea_type='text',
                                                   name_scape='ocr_nextvlad_fea',
                                                   mask=tf.sequence_mask(self.ocr_sequence_length, self.asr_orc_max_len,dtype=tf.float32))
            

           
            BiLSTM_title_feature = self._BiLSTM_feature(
                embedding_descript=embedded_title,
                hidden_size=self.hidden_size,
                des_sequence_length=self.title_sequence_length,
                dtype=tf.float32,
                reuse=reuse)
            
            textCNN_title_feature = self._cnn_feature(
                embedding_descript=embedded_title,
                embedding_size=self.embedding_dim,
                filter_sizes=list(map(int, "1,2,3,4".split(","))),
                num_filters=100,
                reuse=reuse,
                dropout_keep_prob=self.dropout_keep_prob
            )
            
            title_fea = tf.concat([BiLSTM_title_feature,textCNN_title_feature,asr_nextvlad_fea,ocr_nextvlad_fea ], axis=1)

            title_fea_drop = slim.dropout(title_fea, keep_prob=self.dropout_keep_prob, is_training=self.is_training,scope="title_fea_drop")
            title_fea_dense_fea = slim.fully_connected(inputs=title_fea_drop,
                                                   num_outputs=2048,
                                                   activation_fn=None,
                                                   scope="title_fea_dense")

            # title_fea_dense_bn = slim.batch_norm(
            #     title_fea_dense_bn,
            #     center=True,
            #     scale=True,
            #     is_training=self.is_training,
            #     scope="title_fea_dense_bn",
            #     fused=False)

            with slim.arg_scope([slim.fully_connected],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': self.is_training, 'center': True,'scale': True}):
                common_merge_fea = tf.concat([rgb_nextvlad_fea,audio_nextvlad_fea, title_fea_dense_fea], -1)
                self.concate_features_SE = self._SE_module(common_merge_fea,
                                                           name_scape="concat_SE")


                self.concate_features_SE_drop = slim.dropout(self.concate_features_SE,
                                                             keep_prob=self.dropout_keep_prob,
                                                             is_training=self.is_training,
                                                             scope="concate_features_SE_drop")


                # 定义分类器

                if self.task_type=='cate1':
                    cate1_total_loss, cate1_total_prob = self.cate_mlp_label_layer(fea_vector=self.concate_features_SE_drop,
                                                                              dropout_keep_prob=self.dropout_keep_prob,
                                                                              cate_num=self.cate1_num,
                                                                              ml_tag=self.cate1_gt_labels,
                                                                              name_scope='cate1_total_loss')

                    tag_total_prob=tf.zeros_like(self.tag_multi_gt_labels)
                    cate2_total_prob=tf.zeros_like(self.cate2_gt_labels)
                    return  cate1_total_loss,tag_total_prob,cate1_total_prob,cate2_total_prob

                elif self.task_type == 'cate2':
                    cate2_total_loss, cate2_total_prob = self.cate_mlp_label_layer(
                                                                    fea_vector=self.concate_features_SE_drop,
                                                                    dropout_keep_prob=self.dropout_keep_prob,
                                                                    cate_num=self.cate2_num,
                                                                    ml_tag=self.cate2_gt_labels,
                                                                    name_scope='cate2_total_loss')

                    tag_total_prob = tf.zeros_like(self.tag_multi_gt_labels)
                    cate1_total_prob = tf.zeros_like(self.cate1_gt_labels)

                    return cate2_total_loss, tag_total_prob, cate1_total_prob,cate2_total_prob

                elif self.task_type == 'cate1_cate2':

                    cate1_total_loss, cate1_total_prob = self.cate_mlp_label_layer(
                                                                    fea_vector=self.concate_features_SE_drop,
                                                                    dropout_keep_prob=self.dropout_keep_prob,
                                                                    cate_num=self.cate1_num,
                                                                    ml_tag=self.cate1_gt_labels,
                                                                    name_scope='cate1_total_loss')

                    cate2_total_loss, cate2_total_prob = self.cate_mlp_label_layer(
                                                                    fea_vector=self.concate_features_SE_drop,
                                                                    dropout_keep_prob=self.dropout_keep_prob,
                                                                    cate_num=self.cate2_num,
                                                                    ml_tag=self.cate2_gt_labels,
                                                                    name_scope='cate2_total_loss')

                    tag_total_prob = tf.zeros_like(self.tag_multi_gt_labels)


                    return cate1_total_loss+cate2_total_loss, tag_total_prob, cate1_total_prob,cate2_total_prob




                elif self.task_type=='tag':
                    tag_total_loss, tag_total_prob = self.tag_mlp_label_layer(fea_vector=self.concate_features_SE_drop,
                                                                        dropout_keep_prob=self.dropout_keep_prob,
                                                                        tag_num=self.tag_num,
                                                                        ml_tag=self.tag_multi_gt_labels,
                                                                        name_scope='tag_total_loss')

                    cate1_total_prob = tf.zeros_like(self.cate1_gt_labels)
                    cate2_total_prob = tf.zeros_like(self.cate2_gt_labels)


                    return tag_total_loss, tag_total_prob, cate1_total_prob,cate2_total_prob

                elif self.task_type=='cate1_cate2_tag':
                    cate1_total_loss, cate1_total_prob = self.cate_mlp_label_layer(fea_vector=self.concate_features_SE_drop,
                                                                                    dropout_keep_prob=self.dropout_keep_prob,
                                                                                    cate_num=self.cate1_num,
                                                                                    ml_tag=self.cate1_gt_labels,
                                                                                    name_scope='cate1_total_loss')

                    cate2_total_loss, cate2_total_prob = self.cate_mlp_label_layer(fea_vector=self.concate_features_SE_drop,
                                                                                    dropout_keep_prob=self.dropout_keep_prob,
                                                                                    cate_num=self.cate2_num,
                                                                                    ml_tag=self.cate2_gt_labels,
                                                                                    name_scope='cate2_total_loss')

                    tag_total_loss, tag_total_prob = self.tag_mlp_label_layer(fea_vector=self.concate_features_SE_drop,
                                                                              dropout_keep_prob=self.dropout_keep_prob,
                                                                              tag_num=self.tag_num,
                                                                              ml_tag=self.tag_multi_gt_labels,
                                                                              name_scope='tag_total_loss')
                    return cate1_total_loss+cate2_total_loss+tag_total_loss, tag_total_prob,cate1_total_prob,cate2_total_prob

    def _build_graph(self,):


        embedded_ocr = tf.nn.embedding_lookup(self.word_embeddings,self.ocr_id_int_list)  # [batch,ocr_seq_len,emb_size]

        embedded_asr = tf.nn.embedding_lookup(self.word_embeddings,self.asr_id_int_list)  # [batch,asr_seq_len,emb_size]

        embedded_title = tf.nn.embedding_lookup(self.word_embeddings,self.title_id_int_list)  # [batch,title_seq_len,emb_size]



        # ###############################################

        cl_loss, self.tag_prob,self.cate1_prob,self.cate2_prob = self.cal_loss(rgb_fea=self.input_rgb_fea,
                                                                              audio_fea=self.input_audio_fea,
                                                                              text_emb_fea=[embedded_title,embedded_asr,embedded_ocr],
                                                                              reuse=None)


        rgb_fea_perturbated = self.add_perturbation(embedded=self.input_rgb_fea,
                                                    loss=cl_loss,
                                                    norm_length=self.ad_strength)

        audio_fea_perturbated = self.add_perturbation(embedded=self.input_audio_fea,
                                                    loss=cl_loss,
                                                    norm_length=self.ad_strength)

        embedded_ocr_perturbated = self.add_perturbation(embedded=embedded_ocr,
                                                          loss=cl_loss,
                                                          norm_length=self.ad_strength)

        embedded_asr_perturbated = self.add_perturbation(embedded=embedded_asr,
                                                    loss=cl_loss,
                                                    norm_length=self.ad_strength)

        embedded_title_perturbated = self.add_perturbation(embedded=embedded_title,
                                                    loss=cl_loss,
                                                    norm_length=self.ad_strength)



        ad_loss, self.tag_ad_prob,self.cate1_ad_prob,self.cate2_ad_prob = self.cal_loss(rgb_fea=rgb_fea_perturbated,
                                                                                        audio_fea=audio_fea_perturbated,
                                                                                        text_emb_fea=[embedded_title_perturbated,embedded_asr_perturbated,embedded_ocr_perturbated],
                                                                                        reuse=True)

        self.tag_prob_predict_topk, self.tag_index_predict_topk = tf.nn.top_k(self.tag_prob, self.tag_num)
        self.cate1_prob_predict_topk, self.cate1_index_predict_topk = tf.nn.top_k(self.cate1_prob, self.cate1_num)
        self.cate2_prob_predict_topk, self.cate2_index_predict_topk = tf.nn.top_k(self.cate2_prob, self.cate2_num)

        self.total_loss = cl_loss + ad_loss
