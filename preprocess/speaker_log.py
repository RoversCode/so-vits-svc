#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   speaker_log.py
@Time    :   2024/09/23 22:40:38
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
'''

# here put the import lib
import os


def traverse_dir_files(root_dir, ext="dic"):
    paths_list=[]
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith('.'):  # 去除隐藏文件
                continue
            if ext:  # 根据后缀名搜索
                if name.endswith(tuple(ext)):
                    # names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                # names_list.append(name)
                paths_list.append(os.path.join(parent, name))

    return paths_list



# 版本要求 modelscope version 升级至最新版本 funasr 升级至最新版本

from modelscope.pipelines import pipeline
sd_pipeline = pipeline(
    task='speaker-diarization',
    model='/mnt/e/Workspace/modeldata/torch/audio/iic/speech_campplus_speaker-diarization_common',
    model_revision='master'
)
input_wav = '/mnt/e/Workspace/growth/audio/so-vits-svc/preprocess/raw_data/蜡笔小新人声分离/1_1_(Vocals).wav'
result = sd_pipeline(input_wav)
print(result)
# 如果有先验信息，输入实际的说话人数，会得到更准确的预测结果
# result = sd_pipeline(input_wav, oracle_num=2)
# print(result)
