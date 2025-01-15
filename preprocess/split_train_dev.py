#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   preprocess_flist_config.py
@Time    :   2024/11/11 22:32:03
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import os
import sys

# 获取当前文件的绝对路径
cur_dir = os.path.dirname(os.path.abspath(__file__))
# 拿到父目录
parent_dir = os.path.dirname(cur_dir)
print(parent_dir)
sys.path.append(parent_dir)
import argparse
import json
import re
import wave
import yaml
import logging
from random import shuffle
from tqdm import tqdm
from pathlib import Path
import utils

pattern = re.compile(r"^[\.a-zA-Z0-9_\/]+$")

logger = logging.getLogger(__name__)


def get_wav_duration(file_path):
    try:
        with wave.open(str(file_path), "rb") as wav_file:
            # 获取音频帧数
            n_frames = wav_file.getnframes()
            # 获取采样率
            framerate = wav_file.getframerate()
            # 计算时长（秒）
            return n_frames / float(framerate)
    except Exception as e:
        logger.error(f"Reading {file_path}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="path to source dir")
    parser.add_argument(
        "--speech_encoder",
        type=str,
        default="vec768l12",
        help="choice a speech encoder|'vec768l12','vec256l9','hubertsoft','whisper-ppg','cnhubertlarge','dphubert','whisper-ppg-large','wavlmbase+'",
    )
    parser.add_argument(
        "--vol_aug",
        action="store_true",
        help="Whether to use volume embedding and volume augmentation",
    )
    parser.add_argument(
        "--train_type",
        type=str,
        default="finetune",
        help="Whether to use volume embedding and volume augmentation",
    )
    parser.add_argument(
        "--tiny", action="store_true", help="Whether to train sovits tiny"
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    spk_dict = json.load(open(data_dir / "spk_info.json"))

    train = []
    val = []
    idx = 0
    spk_id = 0

    audio_dir = data_dir / "audio_slice"
    subdirs = [x.name for x in audio_dir.iterdir() if x.is_dir()]

    for spk_name in subdirs:
        if spk_name not in spk_dict:
            assert ValueError(f"{spk_name} not in spk_info.json")
        file_paths = (audio_dir / spk_name).glob("*.wav")
        wavs = []
        for file_path in file_paths:
            if not file_path.name.endswith("wav"):
                continue
            if file_path.name.startswith("."):
                continue
            if get_wav_duration(file_path) < 0.3:
                logger.info("Skip too short audio: " + str(file_path))
                continue
            wavs.append(file_path)
        shuffle(wavs)
        train += wavs[2:]
        val += wavs[:2]

    shuffle(train)
    shuffle(val)

    logger.info("Writing " + str(data_dir / "train.list"))
    with open(data_dir / "train.list", "w") as f:
        for fname in tqdm(train):
            f.write(str(fname).strip() + "\n")

    logger.info("Writing " + str(data_dir / "val.list"))
    with open(data_dir / "val.list", "w") as f:
        for fname in tqdm(val):
            f.write(str(fname).strip() + "\n")

    config_path = Path("configs") / data_dir.name
    config_path.mkdir(parents=True, exist_ok=True)  # 创建文件夹

    # 读取base_config

    sovtis_base_config = utils.get_hparams_from_file(
        config_path.parent / "sovits_base_config.yaml"
    )
    diff_base_config = utils.get_hparams_from_file(
        config_path.parent / "diffusion_base_config.yaml"
    )

    diff_base_config.model.n_spk = len(spk_dict)
    diff_base_config.data.encoder = args.speech_encoder
    diff_base_config.data.spk = spk_dict

    sovtis_base_config.data.spk = spk_dict
    sovtis_base_config.model.n_speakers = len(spk_dict)
    sovtis_base_config.model.speech_encoder = args.speech_encoder

    sovtis_base_config.data.training_files = str(data_dir / "train.list")
    sovtis_base_config.data.validation_files = str(data_dir / "val.list")
    diff_base_config.data.training_files = str(data_dir / "train.list")
    diff_base_config.data.validation_files = str(data_dir / "val.list")
    
    sovtis_base_config.train.train_type = args.train_type  # 训练类型
    diff_base_config.train.train_type = args.train_type

    if (
        args.speech_encoder == "vec768l12"
        or args.speech_encoder == "dphubert"
        or args.speech_encoder == "wavlmbase+"
    ):
        sovtis_base_config.model.ssl_dim = sovtis_base_config.model.filter_channels = (
            sovtis_base_config.model.gin_channels
        ) = 768
        diff_base_config.data.encoder_out_channels = 768
    elif args.speech_encoder == "vec256l9" or args.speech_encoder == "hubertsoft":
        sovtis_base_config.model.ssl_dim = sovtis_base_config.model.gin_channels = 256
        diff_base_config.data.encoder_out_channels = 256
    elif args.speech_encoder == "whisper-ppg" or args.speech_encoder == "cnhubertlarge":
        sovtis_base_config.model.ssl_dim = sovtis_base_config.model.filter_channels = (
            sovtis_base_config.model.gin_channels
        ) = 1024
        diff_base_config.data.encoder_out_channels = 1024
    elif args.speech_encoder == "whisper-ppg-large":

        sovtis_base_config.model.ssl_dim = sovtis_base_config.model.filter_channels = (
            sovtis_base_config.model.gin_channels
        ) = 1280
        diff_base_config.data.encoder_out_channels = 1280

    if args.vol_aug:
        sovtis_base_config.train.vol_aug = sovtis_base_config.model.vol_embedding = sovtis_base_config.data.vol_aug = True
    else:
        sovtis_base_config.train.vol_aug = sovtis_base_config.model.vol_embedding = sovtis_base_config.data.vol_aug = False
    

    if args.tiny:
        sovtis_base_config.model.filter_channels = 512

    logger.info("Writing to configs/config.json")

    # 保存为yaml文件
    print(config_path / "sovits_config.yaml")

    utils.save_config(config_path / "sovits.yaml", sovtis_base_config)
    utils.save_config(config_path / "diffusion.yaml", diff_base_config)
