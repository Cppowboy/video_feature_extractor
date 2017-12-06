# coding: utf-8
import os

a_feature_size = 2048  # 表观特征的大小
m_feature_size = 4096  # 运动特征的大小
# feature_size = a_feature_size + m_feature_size  # 最终特征大小
feature_size = a_feature_size  # 最终特征大小
max_frames = 28  # 图像序列的最大长度
width = 7
height = 7

DATA_DIR = '/home/sensetime/data'
# 数据相关的参数
# 提供两个数据集：MSR-VTT和MSVD
msrvtt_video_root = os.path.join(DATA_DIR, 'msr-vtt/train-video')
msrvtt_anno_json_path = os.path.join(DATA_DIR, 'msr-vtt/videodatainfo_2017.json')
msrvtt_video_sort_lambda = lambda x: int(x[5:-4])
msrvtt_train_range = (0, 6512)
msrvtt_val_range = (6513, 7009)
msrvtt_test_range = (7010, 9999)
msrvtt_save_path = os.path.join(DATA_DIR, 'msr-vtt/npy')

msvd_video_root = os.path.join(DATA_DIR, 'msvd/YouTubeClips')
msvd_csv_path = os.path.join(DATA_DIR, 'msvd/MSR_Video_Description_Corpus.csv')  # 手动修改一些数据集中的错误
msvd_video_name2id_map = './datasets/MSVD/youtube_mapping.txt'
msvd_anno_json_path = './datasets/MSVD/annotations.json'  # MSVD并未提供这个文件，需要自己写代码生成（build_msvd_annotation.py）
msvd_video_sort_lambda = lambda x: int(x[3:-4])
msvd_train_range = (0, 1200)
msvd_val_range = (1200, 1300)
msvd_test_range = (1300, 1970)
msvd_map_dict_path = os.path.join(DATA_DIR, 'msvd/youtube2text_iccv15/dict_youtube_mapping.pkl')
msvd_save_path = os.path.join(DATA_DIR, 'msvd/npy3')

dataset = {
    'msr-vtt': [msrvtt_video_root, msrvtt_video_sort_lambda, msrvtt_anno_json_path,
                msrvtt_train_range, msrvtt_val_range, msrvtt_test_range, msrvtt_save_path],
    'msvd': [msvd_video_root, msvd_video_sort_lambda, msvd_anno_json_path,
             msvd_train_range, msvd_val_range, msvd_test_range, msvd_save_path]
}

# 用video_root和anno_json_path这两个变量来切换所使用的数据集
# video_sort_lambda用来对视频按照名称进行排序
# ds = 'msvd'
ds = 'msr-vtt'
video_root, video_sort_lambda, anno_json_path, \
train_range, val_range, test_range, save_dir = dataset[ds]

# checkpoint相关的超参数
# resnet_checkpoint = '/home/sensetime/model/resnet/resnet50-19c8e357.pth'  # 直接用pytorch训练的模型
resnet_checkpoint = '/home/sensetime/model/resnet/resnet152-b121ed2d.pth'  # 直接用pytorch训练的模型
