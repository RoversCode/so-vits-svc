#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   timbre_multi_gpu.py
@Time    :   2024/09/03 17:53:13
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import multiprocessing.queues
import sys
import os
import logging
import multiprocessing
import time
import numpy as np
import torch
import torchaudio
from eres2net.eres2net import ERes2NetV2
import torchaudio.compliance.kaldi as Kaldi


logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("/mnt/e/Workspace/growth/audio/so-vits-svc/preprocess/logs/style.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

# cpu_results
cpu_queue = multiprocessing.Manager().Queue(256)
# 共享的结束标志
cpu_finished = multiprocessing.Value("b", False)


class FBank(object):
    def __init__(
        self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr == self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0] == 1
        feat = Kaldi.fbank(
            wav, num_mel_bins=self.n_mels, sample_frequency=sr, dither=dither
        )
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat


def load_wav(wav_file, obj_fs=16000):
    wav, fs = torchaudio.load(wav_file)
    if fs != obj_fs:
        wav, fs = torchaudio.sox_effects.apply_effects_tensor(
            wav, fs, effects=[["rate", str(obj_fs)]]
        )
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
    return wav


def cpu_text_process(audio_path, save_root):
    files = os.listdir(audio_path)
    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
    for file_name in files:
        path = os.path.join(audio_path, file_name)
        # 检查是否已存在npy文件
        if os.path.exists(os.path.join(save_root, f"{file_name}.npy")):
            continue
        wav = load_wav(path)
        if wav.shape[1] < 16000:
            continue
        input_data = feature_extractor(wav).unsqueeze(0)
        cpu_queue.put([input_data, file_name])

    # 设置CPU完成标志
    with cpu_finished.get_lock():
        cpu_finished.value = True
    logger.info(
        f"CPU process finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )


def generate_feature(device_id, save_root):
    device_id = str(device_id)

    @torch.no_grad()
    def _gpu_process():
        device = torch.device("cuda:" + device_id)
        model_config = {
            "feat_dim": 80,
            "embedding_size": 192,
            "baseWidth": 26,
            "scale": 2,
            "expansion": 2,
        }
        model = ERes2NetV2(**model_config)
        model.load_state_dict(
            torch.load(
                "/mnt/e/Workspace/growth/audio/so-vits-svc/preprocess/eres2net/eres2netv.pth"
            ),
            strict=True,
        )
        model = model.eval().to(device)
        logger.info(
            f"GPU Start at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
        )
        while True:
            # 使用非阻塞的get，添加超时
            try:
                data = cpu_queue.get(timeout=1)
                input_data, file_name = data
            except Exception as e:
                # 如果队列为空且CPU已完成，则退出循环
                if cpu_finished.value and cpu_queue.empty():
                    break
                continue
            input_data = input_data.to(device)
            timbre_embedding = model(input_data).detach().cpu().numpy()
            np.save(os.path.join(save_root, f"{file_name}.npy"), timbre_embedding)
        logger.info(
            f'GPU audio process on device {device_id} finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
        )

    return _gpu_process


if __name__ == "__main__":
    save_root = (
        "/mnt/e/Workspace/growth/audio/so-vits-svc/preprocess/raw_data/timbre"
    )
    audio_path = "/mnt/e/Workspace/growth/audio/so-vits-svc/preprocess/raw_data/蜡笔小新切片"
    # 打印开始时间
    start_time = time.perf_counter()
    logger.info(f"CPU Start at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    cpu_process = multiprocessing.Process(
        target=cpu_text_process, args=(audio_path, save_root)
    )
    cpu_process.start()

    # 启动GPU进程
    gpu_processes = []
    for i in range(0, 1):
        for _ in range(4):
            p = multiprocessing.Process(target=generate_feature(i, save_root))
            p.start()
            gpu_processes.append(p)

    # 等待所有进程完成
    cpu_process.join()
    for p in gpu_processes:
        p.join()

    # 打印结束时间
    total_time = time.perf_counter() - start_time
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info("All processes finished.")
