#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/25 18:33
# @Author  : stonye
# @Email   : stoneye@tencent.com
# @File    : data.py


import os
import numpy as np
from numpy.random import randint
import tensorflow as tf
import random


class Data_itertool(object):

    def __init__(self,
                 sess,
                 batch_size,
                 cate1_num,
                 cate2_num,
                 tag_num,
                 title_max_length,
                 asr_orc_max_length,
                 cate1_cate2_mapping_vocab_path=None,
                 train_vid2acr_ocr_infos_path=None,
                 train_vid2cate_tag_infos_path=None,
                 test_vid2cate_tag_infos_path=None,
                 test_vid2acr_ocr_infos_path=None,
                 tfrecord_train_data_path=None,
                 tfrecord_test_data_path=None,
                 valid_data_prefix='split_1',
                 train_vid_split_infos=None,
                 mode='train',
                 rgb_max_frame=64,
                 audio_max_frame=64,
                 segment_or_padding='padding'):
        self.sess = sess
        self.cate1_num = cate1_num
        self.cate2_num = cate2_num
        self.tag_num = tag_num
        self.title_max_length = title_max_length
        self.asr_orc_max_length = asr_orc_max_length

        self.batch_size = batch_size
        self.mode = mode
        self.rgb_max_frame = rgb_max_frame
        self.audio_max_frame = audio_max_frame
        self.segment_or_padding = segment_or_padding


        self.cate2_to_cate1_mapping_dict=self.get_cate2_to_cate1_dict(in_file=cate1_cate2_mapping_vocab_path)

        if self.mode == 'train' or self.mode == 'valid':
            self.train_vid_set, \
            self.valid_vid_set = self.parse_train_vid_split_infos(
                in_file=train_vid_split_infos, valid_data_prefix=valid_data_prefix)
            self.train_vid2cate_tag_title_dict = self.parse_vid2cate_tag_title_dict(
                in_file=train_vid2cate_tag_infos_path,mode='train')
            self.train_vid2asr_ocr_dict, _ = self.parse_vid2asr_ocr_dict(in_file=train_vid2acr_ocr_infos_path)

            print(
                "len(train_vid_set):{};len(valid_vid_set):{};len(train_vid2cate_tag_title_dict):{};len(train_vid2asr_ocr_dict):{}".format(
                    len(self.train_vid_set), len(self.valid_vid_set), len(self.train_vid2cate_tag_title_dict),
                    len(self.train_vid2asr_ocr_dict)))

        elif self.mode == 'test':

            self.test_vid2cate_tag_title_dict = self.parse_vid2cate_tag_title_dict(in_file=test_vid2cate_tag_infos_path,mode='test')
            self.test_vid2asr_ocr_dict, self.test_vid_set = self.parse_vid2asr_ocr_dict(
                in_file=test_vid2acr_ocr_infos_path)

            print("len(test_vid_set):{};len(test_vid2cate_tag_title_dict):{};len(test_vid2asr_ocr_dict):{}".format(
                len(self.test_vid_set), len(self.test_vid2cate_tag_title_dict), len(self.test_vid2asr_ocr_dict)))

        if self.mode=='test':
            tfrecord_data_path=tfrecord_test_data_path
        else:
            tfrecord_data_path=tfrecord_train_data_path
        print("self.mode:{}; tfrecord_data_path:{}".format(self.mode,tfrecord_data_path))
        self.parse_tfrecord(data_path=tfrecord_data_path)

    def get_cate2_to_cate1_dict(self,in_file):
        '''
        
        :param in_file: 
        :return: 
        '''
        cate2_to_cate1_dict={}
        with open(in_file,'r')as fr:
            for line in fr:
                line_split=line.rstrip('\n').split('\t')
                # cate1_text=line_split[0]
                cate1_id=int(line_split[1])
                # cate2_text=line_split[2]
                cate2_id=int(line_split[3])
                cate2_to_cate1_dict[cate2_id]=cate1_id
        return cate2_to_cate1_dict
    
    def parse_tfrecord(self, data_path):
        '''

        :param data_path:
        :return:
        '''

        mode = self.mode
        filenames = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.endswith('.tfrecord')]
        num_parallel_reads = min(6, len(filenames))
        print("mode={}; len(filenames)={}; num_parallel_reads={}".format(mode, len(filenames), num_parallel_reads))
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_parallel_reads)

        if mode == 'train' or mode == 'valid':
            dataset = dataset.filter(self.filter_fun)
            dataset = dataset.apply(tf.contrib.data.ignore_errors())
        dataset = dataset.map(self.do_parse_func, num_parallel_calls=num_parallel_reads)

        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=self.batch_size * 20)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        self.next_element = iterator.get_next()
        self.dataset_init_op = iterator.make_initializer(dataset)

    def parse_other_fea_infos(self, vid):
        vid = vid.decode()
        mode = self.mode
        if mode == 'train' or mode == 'valid':
            vid2cate_infos = self.train_vid2cate_tag_title_dict
            vid2asr_infos = self.train_vid2asr_ocr_dict
        elif mode == 'test':
            vid2cate_infos = self.test_vid2cate_tag_title_dict
            vid2asr_infos = self.test_vid2asr_ocr_dict
        else:
            raise Exception('mode must be in train,valid,test')

        cate1_id,cate2_id, tags, title = vid2cate_infos[vid]
        asr, ocr = vid2asr_infos[vid]

        # asr、ocr fea
        asr_list = [int(token) for token in asr.split(';') if token]
        ocr_list = [int(token) for token in ocr.split(';') if token]
        padded_asr_list = self.padding_id_list(text_id_list=asr_list, max_num=self.asr_orc_max_length)
        padded_ocr_list = self.padding_id_list(text_id_list=ocr_list, max_num=self.asr_orc_max_length)

        # cate label


        cate1_gt_label = self.parse_cate2_gt_label(cate_id=cate1_id, cate_num=self.cate1_num)
        cate2_gt_label = self.parse_cate2_gt_label(cate_id=cate2_id,cate_num=self.cate2_num)


        # tag multi label
        tag_list = [int(token) for token in tags.split(';') if token]
        tag_multi_gt_label = self.parse_tag_multi_gt_label(tag_list=tag_list)

        # text2id fea
        title_list = [int(token) for token in title.split(';') if token]
        padded_title_list = self.padding_id_list(text_id_list=title_list, max_num=self.title_max_length)

        return np.asarray(cate1_gt_label, dtype=np.float32),np.asarray(cate2_gt_label, dtype=np.float32), np.asarray(tag_multi_gt_label, dtype=np.float32), \
               np.asarray(padded_title_list, dtype=np.int32), np.asarray(padded_asr_list, dtype=np.int32), np.asarray(
            padded_ocr_list, dtype=np.int32)

    def filter_fun(self, example_proto):
        dics = {
            'vid': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'vggish_wav_emb': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'clip_frames_rgb_emb': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'shape_vggish_wav_emb': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            'shape_clip_frames_rgb_emb': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
        }

        features = tf.parse_single_example(example_proto, dics)

        vid = features['vid']

        return tf.py_func(func=self.do_filter_by_vid, inp=[vid],
                          Tout=tf.bool)

    def do_filter_by_vid(self, vid):

        vid = vid.decode()
        if self.mode == 'train':
            dataset = self.train_vid_set
        elif self.mode == 'valid':
            dataset = self.valid_vid_set
        if vid in dataset:
            return True
        else:
            return False

    def parse_cate2_gt_label(self, cate_id,cate_num):
        '''
        :param cate_id:
        :return:
        '''
        cate2_gt_label = [0.0] * cate_num
        if self.mode=='test':
            return cate2_gt_label
        else:
            cate2_gt_label[cate_id] = 1.0
            return cate2_gt_label

    def parse_tag_multi_gt_label(self, tag_list):
        '''

        :param cate_id:
        :return:
        '''
        tag_multi_gt_label = [0.0] * self.tag_num
        for tag_id in tag_list:
            if tag_id >= self.tag_num:
                continue
            tag_multi_gt_label[tag_id] = 1.0
        return tag_multi_gt_label

    def parse_vid2cate_tag_title_dict(self, in_file,mode='test'):
        '''

        :param in_file:
        :return:
        '''
        vid2cate_tag_title_dict = {}
        with open(in_file, 'r')as fr:
            for line in fr:
                line_split = line.rstrip('\n').split('\t')
                vid = line_split[0]
                cate2_id = int(line_split[1])

                if mode=='test':
                    cate1_id=-1
                else:
                    cate1_id=self.cate2_to_cate1_mapping_dict[cate2_id]
                tags = line_split[2]
                title = line_split[3]
                vid2cate_tag_title_dict[vid] = (cate1_id,cate2_id, tags, title)
        return vid2cate_tag_title_dict

    def parse_vid2asr_ocr_dict(self, in_file):
        '''

        :param in_file:
        :return:
        '''
        vid2asr_ocr_dict = {}
        vid_set = set()
        with open(in_file, 'r')as fr:
            for line in fr:
                line_split = line.rstrip('\n').split('\t')
                vid = line_split[0]
                asr = line_split[1]
                ocr = line_split[2]
                vid2asr_ocr_dict[vid] = (asr, ocr)
                vid_set.add(vid)
        return vid2asr_ocr_dict, vid_set

    def parse_train_vid_split_infos(self, in_file, valid_data_prefix):
        '''

        :param in_file:
        :param valid_data_prefix:
        :return:
        '''
        train_data_set = set()
        valid_data_set = set()
        with open(in_file, 'r')as fr:
            for line in fr:
                line_split = line.rstrip('\n').split('\t')
                vid = line_split[0]
                split_prefix = line_split[1]

                if split_prefix == valid_data_prefix:
                    valid_data_set.add(vid)
                else:
                    train_data_set.add(vid)
        return train_data_set, valid_data_set

    def padding_id_list(self, text_id_list, max_num=25):
        '''

        :param text_id_list:
        :param max_num:
        :return:
        '''

        if len(text_id_list) > max_num:
            text_id_list = text_id_list[:max_num]
        else:
            for i in range(max_num - len(text_id_list)):
                text_id_list.append(0)  # padID=0; unkID=1
        return text_id_list

    def segment_fea(self, fea_list, segment_num=3):
        '''

        :param fea_list:
        :param segment_num:
        :param mode:
        :return:
        '''
        fea_len = len(fea_list)
        if self.mode == 'train' and fea_len > segment_num:
            average_duration = fea_len // segment_num
            begin_index = np.multiply(list(range(segment_num)), average_duration)
            random_index = randint(average_duration, size=segment_num)
            indexs = list(begin_index + random_index)
            segment_fea_list_x = [fea_list[_index] for _index in indexs]
        else:
            tick = (fea_len) / float(segment_num)
            indexs = np.array([int(tick / 2.0 + tick * x) for x in range(segment_num)])
            segment_fea_list_x = [fea_list[_index] for _index in indexs]
        return np.asarray(segment_fea_list_x), np.asarray(segment_num, dtype=np.int32)

    def padding_max_frames(self, fea_list, max_frame):

        feature_size = fea_list.shape[1]
        fea_list = list(fea_list)
        if len(fea_list) > max_frame:
            fea_true_frame = max_frame
            fea_list = fea_list[:max_frame]
        else:
            fea_true_frame = len(fea_list)
            fea_list = np.vstack((fea_list, np.zeros((max_frame - len(fea_list), feature_size))))
        return np.asarray(fea_list, dtype=np.float32), np.asarray(fea_true_frame, dtype=np.int32)

    def do_parse_func(self, example_proto):
        dics = {
            'vid': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'vggish_wav_emb': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'clip_frames_rgb_emb': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'shape_vggish_wav_emb': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            'shape_clip_frames_rgb_emb': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
        }

        features = tf.parse_single_example(example_proto, dics)

        vid = features['vid']
        clip_frames_rgb_emb_tensor = tf.reshape(
            tf.cast(tf.decode_raw(features['clip_frames_rgb_emb'], tf.float16), tf.float32),
            features['shape_clip_frames_rgb_emb'])
        vggish_wav_emb_tensor = tf.reshape(tf.cast(tf.decode_raw(features['vggish_wav_emb'], tf.float16), tf.float32),
                                           features['shape_vggish_wav_emb'])

        if self.segment_or_padding == 'segment':
            padding_rgb_clip, rgb_true_frame = tf.py_func(func=self.segment_fea,
                                                          inp=[clip_frames_rgb_emb_tensor, self.rgb_max_frame],
                                                          Tout=[tf.float32, tf.int32])

            padding_audio_vggish, audio_true_frame = tf.py_func(func=self.segment_fea,
                                                                inp=[vggish_wav_emb_tensor, self.audio_max_frame],
                                                                Tout=[tf.float32, tf.int32])



        elif self.segment_or_padding == 'padding':
            padding_rgb_clip, rgb_true_frame = tf.py_func(func=self.padding_max_frames,
                                                          inp=[clip_frames_rgb_emb_tensor, self.rgb_max_frame],
                                                          Tout=[tf.float32, tf.int32])

            padding_audio_vggish, audio_true_frame = tf.py_func(func=self.padding_max_frames,
                                                                inp=[vggish_wav_emb_tensor, self.audio_max_frame],
                                                                Tout=[tf.float32, tf.int32])

        else:
            raise Exception("self.segment_or_padding must be segment or padding")

        cate1_gt_label,cate2_gt_label, tag_multi_gt_label, padded_title_list, \
        padded_asr_list, padded_ocr_list = tf.py_func(func=self.parse_other_fea_infos, inp=[vid],
                                                      Tout=[tf.float32,tf.float32, tf.float32, tf.int32, tf.int32, tf.int32])

        return vid, padding_rgb_clip, rgb_true_frame, padding_audio_vggish, audio_true_frame, \
               cate1_gt_label,cate2_gt_label, tag_multi_gt_label, padded_title_list, padded_asr_list, padded_ocr_list

    def _yield_one_batch_data(self,):

        num=0
        self.sess.run(self.dataset_init_op)
        while True:
            try:
                vid, padding_rgb_clip, rgb_true_frame, padding_audio_vggish, audio_true_frame, \
                cate1_gt_label, cate2_gt_label, tag_multi_gt_label, padded_title_list, padded_asr_list, padded_ocr_list = self.sess.run(
                    self.next_element)

                num+=vid.shape[0]
                yield [vid, padding_rgb_clip, rgb_true_frame, padding_audio_vggish, audio_true_frame,
                      cate1_gt_label,cate2_gt_label, tag_multi_gt_label, padded_title_list, padded_asr_list, padded_ocr_list]

            except tf.errors.OutOfRangeError:
                print("mode:{}; samples:{}".format(self.mode,num))
                print("End of dataset")
                break


def main():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  # 如果有gpu，则优先用gpu，否则走cpu资源
    config.gpu_options.allow_growth = True
    num = 0
    with tf.Session(config=config) as sess:
        train_iter = Data_itertool(sess=sess,
                                   batch_size=2,
                                   cate1_num=33,
                                   cate2_num=310,
                                   tag_num=64080,
                                   title_max_length=10,
                                   asr_orc_max_length=10,
                                   rgb_max_frame=10,
                                   audio_max_frame=10,
                                   cate1_cate2_mapping_vocab_path='/data/pcg_ceph/ccks/vocab/cate1_cate2_mapping_vocab.txt',
                                   train_vid2acr_ocr_infos_path='/data/pcg_ceph/ccks/cleaned_data/trainset/train_vid2acr_ocr_infos_cleaned.txt',
                                   train_vid2cate_tag_infos_path='/data/pcg_ceph/ccks/cleaned_data/trainset/train_vid2cate_tag_infos_cleaned.txt',
                                   test_vid2cate_tag_infos_path='/data/pcg_ceph/ccks/cleaned_data/test_a_set/test_a_vid2cate_tag_infos_cleaned.txt',
                                   test_vid2acr_ocr_infos_path='/data/pcg_ceph/ccks/cleaned_data/test_a_set/test_a_vid2acr_ocr_infos_cleaned.txt',
                                   tfrecord_train_data_path='/data/pcg_ceph/ccks/tfrecord/clip_vgg_fea_extractor/vgg_clip_tfrecord/trainset',
                                   tfrecord_test_data_path='/data/pcg_ceph/ccks/tfrecord/clip_vgg_fea_extractor/vgg_clip_tfrecord/test_a_set',
                                   valid_data_prefix='split_1',
                                   train_vid_split_infos='/data/pcg_ceph/ccks/cleaned_data/vid_split_infos/train_split_prefix_infos.txt',
                                   segment_or_padding='segment',
                                   mode='valid')
        for samples in train_iter._yield_one_batch_data():
            vid_np, padding_rgb_clip_np, rgb_true_frame_np, padding_audio_vggish_np, audio_true_frame_np, \
            cate1_gt_label_np, cate2_gt_label_np, tag_multi_gt_label_np, padded_title_list_np, padded_asr_list_np, padded_ocr_list_np = tuple(
                samples)

            print("vid_np: ", vid_np)
            print("padding_rgb_clip_np.shape: ", padding_rgb_clip_np.shape)
            print("rgb_true_frame_np: ", rgb_true_frame_np)
            print("padding_audio_vggish_np.shape: ", padding_audio_vggish_np.shape)
            print("audio_true_frame_np: ", audio_true_frame_np)
            print("cate1_gt_label_np: ", cate1_gt_label_np.shape,
                  [np.where(x == 1)[0].tolist() for x in cate1_gt_label_np])
            print("cate2_gt_label_np: ", cate2_gt_label_np.shape,
                  [np.where(x == 1)[0].tolist() for x in cate2_gt_label_np])
            print("tag_multi_gt_label_np: ", tag_multi_gt_label_np.shape,
                  [np.where(x == 1)[0].tolist() for x in tag_multi_gt_label_np])
            print("padded_title_list_np: ", padded_title_list_np.shape)
            print("padded_asr_list_np: ", padded_asr_list_np.shape)
            print("padded_ocr_list_np: ", padded_ocr_list_np.shape)

            num += 1

            if num > 3:
                break


if __name__ == "__main__":
    main()
