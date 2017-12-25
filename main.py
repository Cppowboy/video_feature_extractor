# coding: utf-8
'''
载入显著图特征，用vgg16提取
'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import cv2, torch
import numpy as np
import skimage
from torch.autograd import Variable
from model import AppearanceEncoder
from args import ds
from args import video_root, save_dir, msvd_map_dict_path
from args import max_frames, feature_size, width, height
import cPickle as pickle


def sample_frames(video_path, train=True):
    '''
    对视频帧进行采样，减少计算量。等间隔地取max_frames帧
    '''
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        # 把BGR的图片转换成RGB的图片，因为之后的模型用的是RGB格式
        if ret is False:
            break
        frame = frame[:, :, ::-1]
        frames.append(frame)
        frame_count += 1
    print('frame count', frame_count)
    indices = np.linspace(8, frame_count - 7, max_frames, endpoint=False, dtype=int)

    frames = np.array(frames)
    frame_list = frames[indices]
    clip_list = []
    for index in indices:
        clip_list.append(frames[index - 8: index + 8])
    clip_list = np.array(clip_list)
    return frame_list, clip_list, frame_count


def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # 把单通道的灰度图复制三遍变成三通道的图片
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height),
                                           target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,
                        cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height,
                                           int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:
        resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    # 根据在ILSVRC数据集上的图像的均值（RGB格式）进行白化
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def recur_dir(aencoder, base_dir, feature_dir, sub_dir):
    full_dir = os.path.join(base_dir, sub_dir)
    for f in os.listdir(full_dir):
        subpath = os.path.join(sub_dir, f)
        if os.path.isfile(os.path.join(base_dir, subpath)):
            file_name, file_ext = os.path.splitext(subpath)
            if file_ext == '.avi':
                try:
                    video_path = os.path.join(base_dir, subpath)
                    feature_path = os.path.join(feature_dir, file_name)
                    print(video_path, feature_path)
                    frame_list, clip_list, frame_count = sample_frames(video_path, train=True)
                    print(frame_count)
                    frame_list = np.array([preprocess_frame(x) for x in frame_list])
                    frame_list = frame_list.transpose((0, 3, 1, 2))
                    frame_list = Variable(torch.from_numpy(frame_list), volatile=True).cuda()
                    af = aencoder(frame_list)
                    af = af.data.cpu().numpy()
                    if not os.path.exists(os.path.dirname(feature_path)):
                        os.makedirs(os.path.dirname(feature_path))
                    np.save(open(feature_path, 'w'), af)
                    print('saving feature to', af.shape, feature_path)
                except Exception as e:
                    print('Error:', e, file_name)
        else:
            recur_dir(aencoder, base_dir, feature_dir, os.path.join(sub_dir, f))


def extract_features(aencoder):
    # 读取视频列表，让视频按照id升序排列
    videos = sorted(os.listdir(video_root))
    nvideos = len(videos)

    if ds == 'msvd':
        map_dict = pickle.load(open(msvd_map_dict_path, 'r'))

    for i, video in enumerate(videos):
        print(video, end=' ')
        video_path = os.path.join(video_root, video)
        # 提取视频帧以及视频小块
        frame_list, clip_list, frame_count = sample_frames(video_path, train=True)
        print(frame_count)

        # 把图像做一下处理，然后转换成（batch, channel, height, width）的格式
        frame_list = np.array([preprocess_frame(x) for x in frame_list])
        frame_list = frame_list.transpose((0, 3, 1, 2))
        frame_list = Variable(torch.from_numpy(frame_list), volatile=True).cuda()

        # 视频特征的shape是max_frames x (2048 + 4096)
        # 如果帧的数量小于max_frames，则剩余的部分用0补足
        # feats = np.zeros((max_frames * width * height, feature_size), dtype='float32')

        # 先提取表观特征
        af = aencoder(frame_list)

        # 合并表观和动作特征
        # feats[:frame_count, :] = torch.cat([af, mf], dim=1).data.cpu().numpy()
        af = af.data.cpu().numpy()
        # af = af.transpose(0, 2, 3, 1)
        # af = af.reshape(-1, af.shape[-1])
        # feats[:frame_count * width * height, :] = af
        # print(feats.shape)
        feats = af
        if ds == 'msvd':
            vid = map_dict[video[:-4]]
            np.save(open(os.path.join(save_dir, vid + '.npy'), 'w'), feats)
            print('saving', vid, video)
        elif ds == 'msr-vtt':
            np.save(open(os.path.join(save_dir, video[:-4] + '.npy'), 'w'), feats)
            print('saving', video)


def main():
    aencoder = AppearanceEncoder()
    aencoder.eval()
    aencoder.cuda()
    ## uncommment these lines to extract msvd or msr-vtt features
    # extract_features(aencoder)
    ## extract mpii-md feature
    recur_dir(aencoder, '/mnt/lustre/panyinxu/data/mpii-md', '/mnt/lustre/panyinxu/data/mpii-md/npy', '')


if __name__ == '__main__':
    main()
