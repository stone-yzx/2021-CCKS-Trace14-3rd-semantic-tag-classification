#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : NeXtVLADModelLF.py
# @Author: stoneye
# @Date  : 2020/06/24
# @Desc  :
# @license : Copyright(C), Zhenxu Ye
# @Contact : yezhenxu1992@163.com
# @Software : PyCharm

import sys
from model import tag_model
from data import  Data_itertool
import numpy as np
import tensorflow as tf
import argparse
import os
import time
import json

from eval_metrics import cal_eval_metrics,calculate_gap

# np.random.seed(5)
# tf.set_random_seed(-1)

parser = argparse.ArgumentParser()
parser.add_argument("-model_type", "--model_type", help="the model for [train | predict |train ]",
                    default="train", type=str)  # 控制开关，即训练or测试
parser.add_argument("-segment_or_padding", "--segment_or_padding", help="segment or padding",
                    default="segment", type=str)  # 控制开关，即训练or测试


parser.add_argument("-task_type", "--task_type", help="cate tag cate_and_tag]",
                    default="cate", type=str)  # 控制开关，即训练or测试

parser.add_argument("-ad_strength", "--ad_strength", help="ad_strength",
                    default=0.5, type=float)  # 对抗扰动系数

parser.add_argument("-tag_num", "--tag_num", help="the num of tags ",
                    default=64080, type=int)  # keep tag_num
parser.add_argument("-cate1_num", "--cate1_num", help="the num of tags ",
                    default=33, type=int)  # keep tag_num
parser.add_argument("-cate2_num", "--cate2_num", help="the num of tags ",
                    default=310, type=int)  # keep tag_num


parser.add_argument("-train_sample_nums", "--train_sample_nums", help="the num of trainset ",
                    default=9000*4, type=int)  # keep tag_num
parser.add_argument("-asr_orc_max_length", "--asr_orc_max_length", help="the num of tags ",
                    default=10, type=int)  # title_max_len
parser.add_argument("-title_max_length", "--title_max_length", help="the num of tags ",
                    default=10, type=int)  # title_max_len

parser.add_argument("-rgb_frames", "--rgb_frames", help="the size of batch ",
                    default=300, type=int)  # rgb segment_num
parser.add_argument("-audio_frames", "--audio_frames", help="the size of batch ",
                    default=300, type=int)  # rgb segment_num


parser.add_argument("-batch_size", "--batch_size", help="the size of batch ",
                    default=2, type=int)  # mini-batch的数目
parser.add_argument("-lr", "--lr", help="learning rate",
                    default=0.0001, type=float)  # 学习率
parser.add_argument("-keep_dropout", "--keep_dropout", help="dropout keep rate",
                    type=float, default=0.8)  # 学习率
parser.add_argument("-train_tfrecord_data_path", "--train_tfrecord_data_path", help="the data path of train_rgb",
                    type=str, default="/data/pcg_ceph/ccks/tfrecord/clip_vgg_fea_extractor/vgg_clip_tfrecord/trainset")
parser.add_argument("-test_a_tfrecord_data_path", "--test_a_tfrecord_data_path", help="the data path of train_rgb",
                    type=str, default="/data/pcg_ceph/ccks/tfrecord/clip_vgg_fea_extractor/vgg_clip_tfrecord/test_a_set")



parser.add_argument("-word_vocab", "--word_vocab", help="the data path of word vocab",
                    type=str, default="/data/pcg_ceph/stoneye/new_test/dataset/vocab/tencent_ailab_50w_200_emb_vocab.txt")

parser.add_argument("-cate1_cate2_mapping_vocab_path", "--cate1_cate2_mapping_vocab_path", help="the data path of test_rgb",
                    type=str, default="/data/pcg_ceph/ccks/vocab/cate1_cate2_mapping_vocab.txt")


parser.add_argument("-train_vid2acr_ocr_infos_path", "--train_vid2acr_ocr_infos_path", help="the data path of test_rgb",
                    type=str, default="/data/pcg_ceph/ccks/cleaned_data/trainset/train_vid2acr_ocr_infos_cleaned.txt")

parser.add_argument("-train_vid2cate_tag_infos_path", "--train_vid2cate_tag_infos_path", help="the data path of test_rgb",
                    type=str, default="/data/pcg_ceph/ccks/cleaned_data/trainset/train_vid2cate_tag_infos_cleaned.txt")

parser.add_argument("-test_vid2cate_tag_infos_path", "--test_vid2cate_tag_infos_path", help="the data path of test_rgb",
                    type=str, default="/data/pcg_ceph/ccks/cleaned_data/test_a_set/test_a_vid2cate_tag_infos_cleaned.txt")

parser.add_argument("-test_vid2acr_ocr_infos_path", "--test_vid2acr_ocr_infos_path", help="the data path of test_rgb",
                    type=str, default="/data/pcg_ceph/ccks/cleaned_data/test_a_set/test_a_vid2acr_ocr_infos_cleaned.txt")

parser.add_argument("-train_vid_split_infos", "--train_vid_split_infos", help="the data path of test_rgb",
                    type=str, default="/data/pcg_ceph/ccks/cleaned_data/vid_split_infos/train_split_prefix_infos.txt")

parser.add_argument("-pre_Train_path", "--pre_Train_path", help="the data path of pre_Train_path",
                    type=str, default="/data/pcg_ceph/ccks/vocab/head10.emb")  # 预训练好的词向量
parser.add_argument("-report_freq", "--report_freq", help="frequency to report loss",
                    type=int, default=1)  # 打印频率，每隔300个step，即经过300个mini-batch,输出训练的loss
parser.add_argument("-one_epoch_step_eval_num", "--one_epoch_step_eval_num", help="frequency to do validation",
                    type=int, default=4)  # 验证频率，每隔300个step，即经过300个mini-batch,验证当前验证集合的loss
parser.add_argument("-embedding_dim", "--embedding_dim", help="the size of word embedding",
                    type=int, default=200)  # 词向量的维度
parser.add_argument("-hidden_size", "--hidden_size", help="the units of hidden",
                    type=int, default=10)  # 隐藏层的数目
parser.add_argument("-epoch", "--epoch", help="the number of epoch",
                    type=int, default=10)  # 轮数，即训练集合过几遍模型。
parser.add_argument("-model_dir", "--model_dir", help="the dir of save model",
                    type=str, default="./vocab")  # 保存ckpt模型的路径

parser.add_argument("-valid_data_prefix", "--valid_data_prefix", help="valid_data_prefix",
                    type=str, default="split_1")  # 测试文件的预测输出路径
parser.add_argument("-predict_cate_output", "--predict_cate_output", help="the data path of predict_out",
                    type=str, default="./predict_testSet_cate_infos.txt")  # 测试文件的预测输出路径
parser.add_argument("-predict_tag_output", "--predict_tag_output", help="the data path of predict_out",
                    type=str, default="./predict_testSet_tag_infos.txt")  # 测试文件的预测输出路径


args = parser.parse_args()


def creat_model(session, args):
    '''
        模型计算图的构建
        :param session:            会话实例
        :param args:               配置参数
        :return: _model：模型对象
        '''
    model_obj = tag_model(args)
    ckpt = tf.train.get_checkpoint_state(args.model_dir)

    # 判断是否存在模型
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess=session, save_path=ckpt.model_checkpoint_path)  # 调用saver接口，将各个tensor变量的值赋给对应的tensor
        print('model_obj.global_step: ', session.run(model_obj.global_step))
    else:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        print("Created new model parameters..")
        session.run(tf.global_variables_initializer())
    return model_obj


def train():
    '''
        模型，训练任务
        :return:
    '''

    print("train process is doing !")
    valid_cate1_max_f1,valid_cate2_max_f1,valid_tag_max_f1 = -999,-999,-999
    early_stop_num = 0
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  # 如果有gpu，则优先用gpu，否则走cpu资源
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # 训练集迭代器，每次返回mini-batch个样本
        train_data_obj = Data_itertool(sess=sess,
                                 batch_size=args.batch_size,
                                 cate1_num=args.cate1_num,
                                 cate2_num=args.cate2_num,
                                 tag_num=args.tag_num,
                                 rgb_max_frame=args.rgb_frames,
                                 audio_max_frame=args.audio_frames,
                                 title_max_length=args.title_max_length,
                                 asr_orc_max_length=args.asr_orc_max_length,
                                 cate1_cate2_mapping_vocab_path=args.cate1_cate2_mapping_vocab_path,
                                 train_vid2acr_ocr_infos_path=args.train_vid2acr_ocr_infos_path,
                                 train_vid2cate_tag_infos_path=args.train_vid2cate_tag_infos_path,
                                 test_vid2cate_tag_infos_path=args.test_vid2cate_tag_infos_path,
                                 test_vid2acr_ocr_infos_path=args.test_vid2acr_ocr_infos_path,
                                 tfrecord_train_data_path=args.train_tfrecord_data_path,
                                 tfrecord_test_data_path=args.test_a_tfrecord_data_path,
                                 valid_data_prefix=args.valid_data_prefix,
                                 train_vid_split_infos=args.train_vid_split_infos,
                                 segment_or_padding=args.segment_or_padding,
                                 mode='train')

        valid_data_obj = Data_itertool(sess=sess,
                                 batch_size=args.batch_size*2,
                                 cate1_num=args.cate1_num,
                                 cate2_num=args.cate2_num,
                                 tag_num=args.tag_num,
                                 rgb_max_frame=args.rgb_frames,
                                 audio_max_frame=args.audio_frames,
                                 title_max_length=args.title_max_length,
                                 asr_orc_max_length=args.asr_orc_max_length,
                                 cate1_cate2_mapping_vocab_path=args.cate1_cate2_mapping_vocab_path,
                                 train_vid2acr_ocr_infos_path=args.train_vid2acr_ocr_infos_path,
                                 train_vid2cate_tag_infos_path=args.train_vid2cate_tag_infos_path,
                                 test_vid2cate_tag_infos_path=args.test_vid2cate_tag_infos_path,
                                 test_vid2acr_ocr_infos_path=args.test_vid2acr_ocr_infos_path,
                                 tfrecord_train_data_path=args.train_tfrecord_data_path,
                                 tfrecord_test_data_path=args.test_a_tfrecord_data_path,
                                 valid_data_prefix=args.valid_data_prefix,
                                 train_vid_split_infos=args.train_vid_split_infos,
                                 segment_or_padding=args.segment_or_padding,
                                 mode='valid')



        model = creat_model(sess, args)  # 构建模型计算图
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)  # max_to_keep 表征只保留最好的3个模型
        print("Begin training..")

        # total_total_dataset_num = train_iter.total_dataset_num
        total_total_dataset_num = args.train_sample_nums
        one_epoch_contain_step_nums = int(total_total_dataset_num / args.batch_size)

        print("total_total_dataset_num:{}; one_epoch_contain_step_nums:{}".format(total_total_dataset_num,
                                                                                  one_epoch_contain_step_nums))


        for epoch in range(args.epoch):
            old = time.time()
            if early_stop_num >= 50:
                print("early_stop !!! model.global_step.eval(): ", model.global_step.eval())
                break

            total_loss_list = []
            tag_train_total_p_list,tag_train_total_r_list,tag_train_total_f1_list = [],[],[]
            cate1_train_total_p_list, cate1_train_total_r_list, cate1_train_total_f1_list = [], [], []
            cate2_train_total_p_list, cate2_train_total_r_list, cate2_train_total_f1_list = [], [], []


            for samples in train_data_obj._yield_one_batch_data():
                vid_np, padding_rgb_clip_np, rgb_true_frame_np, padding_audio_vggish_np, audio_true_frame_np, \
                cate1_gt_label_np, cate2_gt_label_np, tag_multi_gt_label_np, padded_title_list_np, padded_asr_list_np, padded_ocr_list_np = tuple(
                    samples)

                # 为了防止bn层不稳定
                if vid_np.shape[0] < args.batch_size//3:
                    print('vid_np.shape[0]: ', vid_np.shape[0])
                    continue

                feed = dict(
                    zip([model.title_id_int_list, model.asr_id_int_list,model.ocr_id_int_list,
                         model.tag_multi_gt_labels, model.cate1_gt_labels,model.cate2_gt_labels,
                         model.input_rgb_fea,model.rgb_fea_true_frame,
                         model.input_audio_fea,model.audio_fea_true_frame,
                         model.dropout_keep_prob, model.is_training],
                        [padded_title_list_np, padded_asr_list_np,padded_ocr_list_np,
                         tag_multi_gt_label_np, cate1_gt_label_np,cate2_gt_label_np,
                         padding_rgb_clip_np,rgb_true_frame_np,
                         padding_audio_vggish_np, audio_true_frame_np,
                         args.keep_dropout, True]))

                tag_probs,cate1_probs,cate2_probs, total_loss, curr_lr, _ = sess.run(
                    [model.tag_prob,model.cate1_prob,model.cate2_prob, model.total_loss, model.cur_lr, model.optimizer], feed)

                total_loss_list.append(total_loss)


                #cal tag p,r,f
                tag_gl_tag_dense = [np.where(x == 1)[0].tolist() for x in tag_multi_gt_label_np]
                tag_pred_tag_dense = [np.where(x > 0.5)[0].tolist() for x in tag_probs]

                tag_total_p, tag_total_r, tag_total_f1 = cal_eval_metrics(tag_gl_tag_dense, tag_pred_tag_dense)
                tag_train_total_p_list.append(tag_total_p)
                tag_train_total_r_list.append(tag_total_r)
                tag_train_total_f1_list.append(tag_total_f1)

                # cal cate p,r,f
                cate1_gl_tag_dense = [[np.argmax(x)] for x in cate1_gt_label_np]
                cate1_pred_tag_dense = [[np.argmax(x)] for x in cate1_probs]

                cate2_gl_tag_dense = [[np.argmax(x)] for x in cate2_gt_label_np]
                cate2_pred_tag_dense = [[np.argmax(x)] for x in cate2_probs]



                cate1_total_p, cate1_total_r, cate1_total_f1 = cal_eval_metrics(cate1_gl_tag_dense, cate1_pred_tag_dense)
                cate2_total_p, cate2_total_r, cate2_total_f1 = cal_eval_metrics(cate2_gl_tag_dense, cate2_pred_tag_dense)


                cate1_train_total_p_list.append(cate1_total_p)
                cate1_train_total_r_list.append(cate1_total_r)
                cate1_train_total_f1_list.append(cate1_total_f1)

                cate2_train_total_p_list.append(cate2_total_p)
                cate2_train_total_r_list.append(cate2_total_r)
                cate2_train_total_f1_list.append(cate2_total_f1)


                # print("******************")
                sys.stdout.flush()
                if model.global_step.eval() % args.report_freq == 0:
                    print("report_freq: ", args.report_freq)
                    print('Train-->  Epoch:{}, Step:{}, lr:{:.6f}, Loss:{:.4f}  cate1--> t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f}  cate2--> t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f} tag-->t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f}'.format(
                        epoch, model.global_step.eval(), curr_lr,1.0 * np.sum(total_loss_list) / len(total_loss_list),
                        1.0 * np.sum(cate1_train_total_p_list) / len(cate1_train_total_p_list),
                        1.0 * np.sum(cate1_train_total_r_list) / len(cate1_train_total_r_list),
                        1.0 * np.sum(cate1_train_total_f1_list) / len(cate1_train_total_f1_list),
                        1.0 * np.sum(cate2_train_total_p_list) / len(cate2_train_total_p_list),
                        1.0 * np.sum(cate2_train_total_r_list) / len(cate2_train_total_r_list),
                        1.0 * np.sum(cate2_train_total_f1_list) / len(cate2_train_total_f1_list),
                        1.0 * np.sum(tag_train_total_p_list) / len(tag_train_total_p_list),
                        1.0 * np.sum(tag_train_total_r_list) / len(tag_train_total_r_list),
                        1.0 * np.sum(tag_train_total_f1_list) / len(tag_train_total_f1_list)))

                    total_loss_list = []
                    tag_train_total_p_list, tag_train_total_r_list, tag_train_total_f1_list = [], [], []
                    cate1_train_total_p_list, cate1_train_total_r_list, cate1_train_total_f1_list = [], [], []
                    cate2_train_total_p_list, cate2_train_total_r_list, cate2_train_total_f1_list = [], [], []

                # 验证集上做验证
                if model.global_step.eval() % int(one_epoch_contain_step_nums // args.one_epoch_step_eval_num) == 0:

                    cate1_total_p, cate1_total_r, cate1_total_f1,cate2_total_p, cate2_total_r, cate2_total_f1,\
                    tag_total_p, tag_total_r, tag_total_f1,\
                    valid_total_loss_list_aver = eval(model=model,data_obj=valid_data_obj,sess=sess)
                    print('\n***********************')
                    print('Valid--> Epoch:{}; Step:{}; Valid: loss:{:.4f}\ncate1--> t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f}\ncate2--> t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f}\ntag-->t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f}'.format(
                        epoch,model.global_step.eval(),valid_total_loss_list_aver,
                        cate1_total_p, cate1_total_r, cate1_total_f1,
                        cate2_total_p, cate2_total_r, cate2_total_f1,
                        tag_total_p, tag_total_r, tag_total_f1
                    ))
                    print('***********************\n')


                    # 保存当前验证集合准确率最高的模型


                    if args.task_type=='cate1':
                        if cate1_total_f1 > valid_cate1_max_f1 :
                            print("save the model, step= : ", model.global_step.eval())
                            valid_cate1_max_f1 = cate1_total_f1
                            checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                            saver.save(sess=sess, save_path=checkpoint_path, global_step=model.global_step.eval())
                            early_stop_num = 0
                        else:
                            early_stop_num += 1

                    if args.task_type in ['cate2','cate1_cate2','cate1_cate2_tag'] :
                        if cate2_total_f1 > valid_cate2_max_f1 :
                            print("save the model, step= : ", model.global_step.eval())
                            valid_cate2_max_f1 = cate2_total_f1
                            checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                            saver.save(sess=sess, save_path=checkpoint_path, global_step=model.global_step.eval())
                            early_stop_num = 0
                        else:
                            early_stop_num += 1



                    if args.task_type=='tag':
                        if tag_total_f1>valid_tag_max_f1 :
                            print("save the model, step= : ", model.global_step.eval())
                            valid_tag_max_f1=tag_total_f1
                            checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                            saver.save(sess=sess, save_path=checkpoint_path, global_step=model.global_step.eval())
                            early_stop_num = 0
                        else:
                            early_stop_num += 1




                    print("valid processing is finished!")

            new = time.time()
            print("{}th epoch run {:.3f} minutes:".format(epoch, (new - old) / 60))


def eval(model,data_obj,sess):
    print("Valid infer is process !")
    total_loss_list,tag_gt_list,tag_pt_list,cate1_gt_list,cate1_pt_list,cate2_gt_list,cate2_pt_list = [] ,[],[],[],[],[],[]

    for samples in data_obj._yield_one_batch_data():
        vid_np, padding_rgb_clip_np, rgb_true_frame_np, padding_audio_vggish_np, audio_true_frame_np, \
        cate1_gt_label_np, cate2_gt_label_np, tag_multi_gt_label_np, padded_title_list_np, padded_asr_list_np, padded_ocr_list_np = tuple(
            samples)

        feed = dict(
            zip([model.title_id_int_list, model.asr_id_int_list, model.ocr_id_int_list,
                 model.tag_multi_gt_labels, model.cate1_gt_labels, model.cate2_gt_labels,
                 model.input_rgb_fea, model.rgb_fea_true_frame,
                 model.input_audio_fea, model.audio_fea_true_frame,
                 model.dropout_keep_prob, model.is_training],
                [padded_title_list_np, padded_asr_list_np, padded_ocr_list_np,
                 tag_multi_gt_label_np, cate1_gt_label_np, cate2_gt_label_np,
                 padding_rgb_clip_np, rgb_true_frame_np,
                 padding_audio_vggish_np, audio_true_frame_np,
                 1.0, False]))

        total_loss, tag_probs,cate1_probs,cate2_probs = sess.run(
            [model.total_loss, model.tag_prob,model.cate1_prob,model.cate2_prob], feed)


        total_loss_list.append(total_loss)
        tag_gt_list.extend([np.where(x == 1)[0].tolist() for x in tag_multi_gt_label_np])
        tag_pt_list.extend([np.where(x > 0.5)[0].tolist() for x in tag_probs])

        cate1_gt_list.extend([[np.argmax(x)] for x in cate1_gt_label_np])
        cate1_pt_list.extend([[np.argmax(x)] for x in cate1_probs])

        cate2_gt_list.extend([[np.argmax(x)] for x in cate2_gt_label_np])
        cate2_pt_list.extend([[np.argmax(x)] for x in cate2_probs])




    total_loss_list_aver = 1.0 * np.sum(total_loss_list) / len(total_loss_list)
    tag_total_p, tag_total_r, tag_total_f1 = cal_eval_metrics(tag_gt_list, tag_pt_list)
    cate1_total_p, cate1_total_r, cate1_total_f1 = cal_eval_metrics(cate1_gt_list, cate1_pt_list)
    cate2_total_p, cate2_total_r, cate2_total_f1 = cal_eval_metrics(cate2_gt_list, cate2_pt_list)

    return cate1_total_p, cate1_total_r, cate1_total_f1,\
           cate2_total_p, cate2_total_r, cate2_total_f1, \
           tag_total_p, tag_total_r, tag_total_f1,total_loss_list_aver


def get_mapping_dict(in_file):

    mapping_dict={}

    with open(in_file,'r')as fr:
        for line in fr:
            line_split=line.rstrip('\n').split('\t')
            if len(line_split)!=2:
                continue
            token =line_split[0]
            id=int(line_split[1])
            if id not in mapping_dict:
                mapping_dict[id]=token

    return mapping_dict

def predict():
    '''
    :return:
    '''
    print("predict process is doing !")

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  # 如果有gpu，则优先用gpu，否则走cpu资源
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        test_data_obj = Data_itertool(sess=sess,
                                 batch_size=args.batch_size*2,
                                 cate1_num=args.cate1_num,
                                 cate2_num=args.cate2_num,
                                 tag_num=args.tag_num,
                                 rgb_max_frame=args.rgb_frames,
                                 audio_max_frame=args.audio_frames,
                                 title_max_length=args.title_max_length,
                                 asr_orc_max_length=args.asr_orc_max_length,
                                 cate1_cate2_mapping_vocab_path=args.cate1_cate2_mapping_vocab_path,
                                 train_vid2acr_ocr_infos_path=args.train_vid2acr_ocr_infos_path,
                                 train_vid2cate_tag_infos_path=args.train_vid2cate_tag_infos_path,
                                 test_vid2cate_tag_infos_path=args.test_vid2cate_tag_infos_path,
                                 test_vid2acr_ocr_infos_path=args.test_vid2acr_ocr_infos_path,
                                 tfrecord_train_data_path=args.train_tfrecord_data_path,
                                 tfrecord_test_data_path=args.test_a_tfrecord_data_path,
                                 valid_data_prefix=args.valid_data_prefix,
                                 train_vid_split_infos=args.train_vid_split_infos,
                                 segment_or_padding=args.segment_or_padding,
                                 mode='test')

        model = creat_model(sess, args)  # 构建模型计算图



        for samples in test_data_obj._yield_one_batch_data():
            vid_np, padding_rgb_clip_np, rgb_true_frame_np, padding_audio_vggish_np, audio_true_frame_np, \
            cate1_gt_label_np, cate2_gt_label_np, tag_multi_gt_label_np, padded_title_list_np, padded_asr_list_np, padded_ocr_list_np = tuple(
                samples)

            feed = dict(
                zip([model.title_id_int_list, model.asr_id_int_list, model.ocr_id_int_list,
                     model.tag_multi_gt_labels, model.cate1_gt_labels, model.cate2_gt_labels,
                     model.input_rgb_fea, model.rgb_fea_true_frame,
                     model.input_audio_fea, model.audio_fea_true_frame,
                     model.dropout_keep_prob, model.is_training],
                    [padded_title_list_np, padded_asr_list_np, padded_ocr_list_np,
                     tag_multi_gt_label_np, cate1_gt_label_np, cate2_gt_label_np,
                     padding_rgb_clip_np, rgb_true_frame_np,
                     padding_audio_vggish_np, audio_true_frame_np,
                     1.0, False]))



            if args.task_type=='cate':
                _,predict_probs = sess.run(
                    [model.cur_lr, model.cate_prob], feed)
                with open(args.predict_cate_output,'a+')as fw:
                    for vid ,prob in zip(vid_np.tolist(),predict_probs.tolist()):
                        prob=['{:.4f}'.format(token) for token in prob]
                        fw.write(vid+'\t'+' '.join(prob)+'\n')


            if args.task_type=='tag':
                _,predict_probs = sess.run(
                    [model.cur_lr, model.tag_prob], feed)
                with open(args.predict_tag_output,'a+')as fw:
                    for vid ,prob in zip(vid_np.tolist(),predict_probs.tolist()):
                        prob=['{:.4f}'.format(token) for token in prob]
                        fw.write(vid+'\t'+' '.join(prob)+'\n')

            if args.task_type=='cate1_cate2_tag':
                _,predict_probs,tag_probs = sess.run(
                    [model.cur_lr, model.cate2_prob, model.tag_prob], feed)
                with open(args.predict_cate_output,'a+')as fw:
                    for vid ,prob in zip(vid_np.tolist(),predict_probs.tolist()):

                        prob=['{:.4f}'.format(token) for token in prob]
                        fw.write(str(vid.decode())+'\t'+' '.join(prob)+'\n')

                with open(args.predict_tag_output,'a+')as fw:
                    for vid ,prob in zip(vid_np.tolist(),tag_probs.tolist()):
                        prob=['{:.4f}'.format(token) for token in prob]
                        fw.write(vid+'\t'+' '.join(prob)+'\n')



def predict_valid():
    '''
    :return:
    '''
    print("predict valid process is doing !")

    num=0
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  # 如果有gpu，则优先用gpu，否则走cpu资源
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        valid_data_obj = Data_itertool(sess=sess,
                                       batch_size=args.batch_size,
                                       cate_num=args.cate_num,
                                       tag_num=args.tag_num,
                                       title_max_length=args.title_max_length,
                                       asr_orc_max_length=args.asr_orc_max_length,
                                       train_rgb_fea_base_dir=args.train_rgb_fea_base_dir,
                                       test_rgb_fea_base_dir=args.test_rgb_fea_base_dir,
                                       train_vid2acr_ocr_infos_path=args.train_vid2acr_ocr_infos_path,
                                       train_vid2cate_tag_infos_path=args.train_vid2cate_tag_infos_path,
                                       test_vid2cate_tag_infos_path=args.test_vid2cate_tag_infos_path,
                                       test_vid2acr_ocr_infos_path=args.test_vid2acr_ocr_infos_path,
                                       tfrecord_data_path=args.train_tfrecord_data_path,
                                       valid_data_prefix=args.valid_data_prefix,
                                       train_vid_split_infos=args.train_vid_split_infos,
                                       mode='valid')

        model = creat_model(sess, args)  # 构建模型计算图

        total_loss_list, tag_gt_list, tag_pt_list, cate_gt_list, cate_pt_list = [], [], [], [], []
        for samples in valid_data_obj._yield_one_batch_data():
            vid_batch_np, rgb_fea_batch_np, cate_label_batch_np, tag_label_batch_np, title_fea_bacth_np, asr_fea_batch_np, ocr_fea_batch_np = tuple(
                samples)

            feed = dict(
                zip([model.title_id_int_list, model.asr_id_int_list, model.ocr_id_int_list,
                     model.tag_multi_gt_labels, model.cate_gt_labels, model.input_rgb_fea,
                     model.dropout_keep_prob, model.is_training],
                    [title_fea_bacth_np, asr_fea_batch_np, ocr_fea_batch_np,
                     tag_label_batch_np, cate_label_batch_np, rgb_fea_batch_np,
                     1.0, False]))

            total_loss, tag_probs, cate_probs = sess.run(
                [model.total_loss, model.tag_prob, model.cate_prob], feed)

            total_loss_list.append(total_loss)
            tag_gt_list.extend([np.where(x == 1)[0].tolist() for x in tag_label_batch_np])
            tag_pt_list.extend([np.where(x > 0.5)[0].tolist() for x in tag_probs])



            num+=1

            if num==1:
                print("vid_batch_np: ",vid_batch_np)
                print("[np.where(x == 1)[0].tolist() for x in tag_label_batch_np]: ",[np.where(x == 1)[0].tolist() for x in tag_label_batch_np])
                print("[np.where(x > 0.5)[0].tolist() for x in tag_probs]: ",[np.where(x > 0.5)[0].tolist() for x in tag_probs])
                print('tag_probs: ',tag_probs.tolist())




            cate_gt_list.extend([[np.argmax(x)] for x in cate_label_batch_np])
            cate_pt_list.extend([[np.argmax(x)] for x in cate_probs])



            if args.task_type=='cate':

                with open(args.predict_cate_output,'a+')as fw:
                    for vid ,prob in zip(vid_batch_np.tolist(),cate_probs.tolist()):
                        prob=['{:.4f}'.format(token) for token in prob]
                        fw.write(vid+'\t'+' '.join(prob)+'\n')


            if args.task_type=='tag':

                pass
                # with open(args.predict_tag_output,'a+')as fw:
                #     for vid ,prob in zip(vid_batch_np.tolist(),tag_probs.tolist()):
                #         prob=['{:.4f}'.format(token) for token in prob]
                #         fw.write(vid+'\t'+' '.join(prob)+'\n')



        print("tag_gt_list: ",tag_gt_list)
        print("tag_pt_list: ",tag_pt_list)
        total_loss_list_aver = 1.0 * np.sum(total_loss_list) / len(total_loss_list)
        tag_total_p, tag_total_r, tag_total_f1 = cal_eval_metrics(tag_gt_list, tag_pt_list)
        cate_total_p, cate_total_r, cate_total_f1 = cal_eval_metrics(cate_gt_list, cate_pt_list)

        print('\n***********************')
        print(
            'predict Valid--> loss:{:.4f}\ncate--> t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f}\ntag-->t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f}'.format(
                total_loss_list_aver,
                cate_total_p, cate_total_r, cate_total_f1,
                tag_total_p, tag_total_r, tag_total_f1
            ))
        print('***********************\n')



def main():
    if args.model_type == "train":
        train()

    elif args.model_type == "predict":
        predict()


    elif args.model_type == "predict_valid":
        predict_valid()

if __name__ == '__main__':
    main()
