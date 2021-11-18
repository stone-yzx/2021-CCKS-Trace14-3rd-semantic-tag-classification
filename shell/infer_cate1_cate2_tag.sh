
cd ../src

python train.py \
-model_type predict \
-task_type cate1_cate2_tag \
-valid_data_prefix split_1 \
-segment_or_padding segment \
-ad_strength 0.5 \
-tag_num 1134 \
-cate1_num 33 \
-cate2_num 310 \
-train_sample_nums 36000 \
-asr_orc_max_length 150 \
-title_max_length 50 \
-rgb_frames 80 \
-audio_frames 80 \
-batch_size 64 \
-lr 0.001 \
-keep_dropout 0.8 \
-word_vocab /data/pcg_ceph/stoneye/new_test/dataset/vocab/tencent_ailab_50w_200_emb_vocab.txt \
-cate1_cate2_mapping_vocab_path /data/pcg_ceph/ccks/vocab/cate1_cate2_mapping_vocab.txt \
-train_tfrecord_data_path /data/pcg_ceph/ccks/tfrecord/clip_vgg_fea_extractor/vgg_clip_tfrecord/trainset \
-test_a_tfrecord_data_path /data/pcg_ceph/ccks/tfrecord/clip_vgg_fea_extractor/vgg_clip_tfrecord/test_b_set \
-train_vid2acr_ocr_infos_path  /data/pcg_ceph/ccks/cleaned_data/trainset/train_vid2acr_ocr_infos_cleaned.txt \
-train_vid2cate_tag_infos_path /data/pcg_ceph/ccks/cleaned_data/trainset/train_vid2cate_tag_infos_cleaned.txt \
-test_vid2cate_tag_infos_path /data/pcg_ceph/ccks/cleaned_data/test_b_set/test_b_vid2cate_tag_infos_cleaned.txt \
-test_vid2acr_ocr_infos_path /data/pcg_ceph/ccks/cleaned_data/test_b_set/test_b_vid2acr_ocr_infos_cleaned.txt \
-train_vid_split_infos /data/pcg_ceph/ccks/cleaned_data/vid_split_infos/train_split_prefix_infos.txt \
-pre_Train_path /data/pcg_ceph/stoneye/new_test/dataset/vocab/head_50w_Tencent_AILab_emb_200.txt \
-report_freq 100 \
-one_epoch_step_eval_num 2 \
-embedding_dim 200 \
-hidden_size 512 \
-epoch 80 \
-model_dir  /data/pcg_ceph/ccks/my_baseline_clip/models/tag_1134_cate1_cate2_base \
-predict_cate_output ./predict_out/test_b_predict_cate2_infos.txt \
-predict_tag_output  ./predict_out/test_b_predict_tag_infos.txt |tee  ./log/pretest_tag_1134_cate1_cate2_base.log
