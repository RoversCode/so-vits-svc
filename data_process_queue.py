#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_process_queue.py
@Time    :   2024/11/13 18:00:30
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import os

# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["WANDB_DISABLED"] = "true"
import multiprocessing
import torch
import torchaudio
import random
import yaml
import time
import json
import threading
import numpy as np
from io import BytesIO
from multiprocessing.managers import BaseManager
from vencoder.ContentVec768L12 import ContentVec768L12
from modules.F0Predictor.RMVPEF0Predictor import RMVPEF0Predictor


def data_producer(queue, data_path="Data/sovits_svc/train.list"):
    # spk
    spks = json.load(open("Data/sovits_svc/sing_spk_info.json"))
    # 读取数据
    all_raw_data = []
    with open(data_path, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip().split("|")
            all_raw_data.append(line)

    print("开始生产数据")
    while True:
        for line in all_raw_data:
            # 读取数据
            try:
                sample = {}
                if line[2] not in spks:  # 不在spk中，直接跳过
                    continue
                sample["utt"] = line[0]
                if sample["utt"].endswith(".wav"):
                    sample["utt"] = sample["utt"].replace(".wav", "")
                sample["language"] = line[1]
                sample["dur"] = float(line[-1])
                sample["spk"] = line[2]
                queue.put(sample)
            except Exception as ex:
                print(f"生产数据出错{ex}")
        # # 刷新
        # if data_path.endswith("dev.list"):
        #     time.sleep(60)
        # print("一轮结束，刷新数据")
        random.shuffle(all_raw_data)


def processer(
    device_id, source_queue, target_queue, hop_length=256, sampling_rate=44100
):
    device_id = str(device_id)

    # data = open(utt2wav[utt], "rb").read()
    @torch.no_grad()
    def _processer():
        device = "cuda:" + device_id
        # num = 1
        # ssl
        # rmpve
        content_model = ContentVec768L12(device=device)
        f0_predictor_object = RMVPEF0Predictor(
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            dtype=torch.float32,
            device=device,
            threshold=0.05,
        )
        while True:
            try:
                results = {}
                sample = source_queue.get()  # 获取样本

                if sample["dur"] > 15:  # 15秒以上的音频不处理
                    continue
                print(f"处理进程 {device_id} 处理音频{sample['utt']}")
                # 判断音频文件在哪
                if os.path.exists(
                    os.path.join("Data/22050raw", sample["utt"] + ".wav")
                ):
                    file_path = os.path.join("Data/22050raw", sample["utt"] + ".wav")
                elif os.path.exists(
                    os.path.join("Data/wavetts_22050", sample["utt"] + ".wav")
                ):
                    file_path = os.path.join(
                        "Data/wavetts_22050", sample["utt"] + ".wav"
                    )
                elif os.path.exists(
                    os.path.join("Data/sing_raw", sample["utt"] + ".wav")
                ):
                    file_path = os.path.join(
                        "Data/sing_raw", sample["utt"] + ".wav"
                    )
                else:
                    assert False, f"找不到音频文件{sample['utt']}"

                ssl_t_path = os.path.join(
                    "Data/sovits_svc/ssls", sample["utt"] + ".wav_ssl.pt"
                )
                f0_t_path = os.path.join("Data/sovits_svc/f0s", sample["utt"] + ".wav_f0.npy")
                ssl_path = os.path.join(
                    "Data/sovits_svc/ssls", sample["utt"] + "_ssl.pt"
                )
                f0_path = os.path.join("Data/sovits_svc/f0s", sample["utt"] + "_f0.npy")
                if os.path.exists(ssl_t_path):
                    # 改名
                    os.rename(ssl_t_path, ssl_path)
                if os.path.exists(f0_t_path):
                    os.rename(f0_t_path, f0_path)
                audio_rb = open(file_path, "rb").read()
                results["audio_data"] = audio_rb
                if not os.path.exists(ssl_path) or not os.path.exists(f0_path):
                    audio, sr = torchaudio.load(BytesIO(audio_rb))
                    if audio.dim() == 2 and audio.size(0) > 1:
                        audio = audio.mean(0, keepdim=True)  # 单通道
                    if sr != 16000:  # 重新采样
                        wav16 = torchaudio.transforms.Resample(sr, 16000)(audio)

                if os.path.exists(ssl_path):
                    ssl_feature = torch.load(ssl_path)
                else:
                    # ssl特征
                    ssl_feature = content_model.encoder(
                        wav16.squeeze(0).to(device)
                    )  # ssl
                    ssl_feature = ssl_feature.cpu()
                    torch.save(ssl_feature, ssl_path)
                results["ssl_feature"] = ssl_feature.numpy()

                if os.path.exists(f0_path):
                    f0, uv = np.load(f0_path, allow_pickle=True)
                else:
                    f0, uv = f0_predictor_object.compute_f0_uv(wav16.squeeze(0), 16000)
                    np.save(
                        os.path.join("Data/sovits_svc/f0s", sample["utt"] + "_f0.npy"),
                        np.asanyarray((f0, uv), dtype=object),
                    )
                results["f0"] = f0
                results["uv"] = uv
                results.update(sample)

                # 保存
                target_queue.put(results)

            except Exception as ex:
                print(f"处理进程 {device_id} 处理数据错误: {ex}")
                continue

    return _processer


# 存入encodec的内容
data_product = multiprocessing.Queue(512)
final_queue = multiprocessing.Queue(512)
dev_product = multiprocessing.Queue(10)
dev_queue = multiprocessing.Queue(10)
# 读取配置
config = yaml.safe_load(open("configs/sovits_base_config.yaml", "r"))["data"]

if __name__ == "__main__":
    # 读取音频
    threading.Thread(
        target=data_producer, args=(data_product, "Data/sovits_svc/train.list")
    ).start()
    threading.Thread(
        target=data_producer, args=(dev_product, "Data/sovits_svc/dev.list")
    ).start()

    for i in range(0, 1):
        for _ in range(3):
            time.sleep(1)
            multiprocessing.Process(
                target=processer(
                    i,
                    data_product,
                    final_queue,
                    hop_length=config["hop_length"],
                    sampling_rate=config["sampling_rate"],
                )
            ).start()

    for i in range(0, 1):
        for _ in range(1):
            time.sleep(1)
            multiprocessing.Process(
                target=processer(
                    i,
                    dev_product,
                    dev_queue,
                    hop_length=config["hop_length"],
                    sampling_rate=config["sampling_rate"],
                )
            ).start()

    class QueueManager(BaseManager):
        pass

    QueueManager.register("get_train_queue", callable=lambda: final_queue)
    QueueManager.register("get_dev_queue", callable=lambda: dev_queue)
    # 127.0.0.1
    m = QueueManager(address=("127.0.0.1", 12345), authkey=b"liujunjieabracadabra")
    s = m.get_server()
    print("服务启动")
    s.serve_forever()
