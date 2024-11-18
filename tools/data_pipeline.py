#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_pipeline.py
@Time    :   2024/11/13 21:06:39
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import torch
import json
import torchaudio
import utils
import random
import numpy as np
from io import BytesIO
from torch.nn.utils.rnn import pad_sequence
from modules.mel_processing import spectrogram_torch

try:
    spk2id = json.load(open("Data/sovits_svc/speaker_map.json", "r"))
    # 重新排列speaker id
    spk2id = {k: i for i, k in enumerate(spk2id.keys())}
except Exception as ex:
    print("Failed to load speaker map, ex info {}".format(ex))
    spk2id = None


# 第一道流水线: 过滤
def filter(data, configs):
    """Give url or local file, return file descriptor"""
    for sample in data:
        # assert "src" in sample
        # url = sample["src"]  # parquet文件路径
        try:
            if sample["dur"] < 0.5:
                continue
            if sample["dur"] > 30:
                continue
            yield {**sample}

        except Exception as ex:
            print("Failed to filter {}, ex info {}".format(sample["utt"], ex))


# c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded, volume_padded
def gen_spec(data, configs):
    """Give url or local file, return file descriptor"""
    for sample in data:
        # assert "src" in sample
        # url = sample["src"]  # parquet文件路径
        try:
            audio, sr = torchaudio.load(BytesIO(sample["audio_data"]))
            if audio.dim() == 2 and audio.size(0) > 1:
                audio = audio.mean(0, keepdim=True)  # 单通道
            if sr != configs.sampling_rate:
                audio = torchaudio.transforms.Resample(
                    sr, configs.sampling_rate)(audio)
            audio = audio / torch.max(torch.abs(audio))
            del sample["audio_data"]
            sample["audio"] = audio
            spec = spectrogram_torch(
                audio,
                configs.filter_length,
                configs.sampling_rate,
                configs.hop_length,
                configs.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)  # [freq, time]
            sample["spec"] = spec

            yield sample

        except Exception as ex:
            print("Failed to gen_spec {}, ex info {}".format(
                sample["utt"], ex))


def gen_vol(data, configs):
    for sample in data:
        try:
            if configs.vol_embedding:
                audio = sample["audio"]
                n_frames = int(audio.size(-1) // configs.hop_length)
                audio2 = audio**2
                audio2 = torch.nn.functional.pad(
                    audio2,
                    (int(configs.hop_length // 2),
                     int((configs.hop_length + 1) // 2)),
                    mode="reflect",
                )
                volume = torch.nn.functional.unfold(
                    audio2[:, None, None, :],
                    (1, configs.hop_length),
                    stride=configs.hop_length,
                )[:, :, :n_frames].mean(dim=1)[0]
                volume = torch.sqrt(volume)
                sample["volume"] = volume
            yield sample
        except Exception as ex:
            print("Failed to gen_vol {}, ex info {}".format(sample["utt"], ex))


def align(data, configs):
    for sample in data:
        try:
            if 'spk' in configs:
                sample["spk_id"] = configs.spk[sample["spk"]]
            else:
                sample["spk_id"] = spk2id[sample["spk"]]

            f0 = torch.FloatTensor(sample["f0"].astype(np.float32))
            uv = torch.FloatTensor(sample["uv"].astype(np.float32))
            ssl = torch.FloatTensor(sample["ssl_feature"].astype(np.float32))
            spec = sample["spec"]
            audio = sample["audio"]
            ssl = utils.repeat_expand_2d(ssl.squeeze(0),
                                         f0.shape[0],
                                         mode=configs.unit_interpolate_mode)
            lmin = min(ssl.size(-1), sample["spec"].size(-1))  # NOTE: 对齐
            spec, ssl, f0, uv = spec[:, :lmin], ssl[:, :
                                                    lmin], f0[:lmin], uv[:lmin]

            audio = audio[:, :lmin * configs.hop_length]

            # 如果sample中有volume，那么也要对volume进行截取
            if "volume" in sample:
                volume = sample["volume"]
                volume = volume[:lmin]

            # 增强
            if random.choice([True, False
                              ]) and configs.vol_aug and "volume" in sample:
                max_amp = float(torch.max(torch.abs(audio))) + 1e-5
                max_shift = min(1, np.log10(1 / max_amp))
                log10_vol_shift = random.uniform(-1, max_shift)
                audio = audio * (10**log10_vol_shift)
                volume = volume * (10**log10_vol_shift)
                spec = spectrogram_torch(
                    audio,
                    configs.filter_length,
                    configs.sampling_rate,
                    configs.hop_length,
                    configs.win_length,
                    center=False,
                )[0]
            if spec.shape[1] > 800:
                start = random.randint(0, spec.shape[1] - 800)
                end = start + 790
                spec, ssl, f0, uv = (
                    spec[:, start:end],
                    ssl[:, start:end],
                    f0[start:end],
                    uv[start:end],
                )
                audio = audio[:, start * configs.hop_length:end *
                              configs.hop_length]
                if volume is not None:
                    volume = volume[start:end]

            sample["spec"] = spec
            sample["ssl_feature"] = ssl
            sample["f0"] = f0
            sample["uv"] = uv
            sample["audio"] = audio
            if "volume" in sample:
                sample["volume"] = volume

            yield sample
        except Exception as ex:
            print("Failed to align {}, ex info {}".format(sample["utt"], ex))


def shuffle(data, configs):
    """Local shuffle the data
      数据"蓄水池"，乱序。
    Args:
        data: Iterable[{key, feat, label}]
        shuffle_size: buffer size for shuffle

    Returns:
        Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        try:
            buf.append(sample)
            if len(buf) >= configs.shuffle_size:
                random.shuffle(buf)
                for x in buf:
                    yield x
                buf = []
        except Exception as ex:
            print("Failed to shuffle {}, ex info {}".format(sample["utt"], ex))
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


# 第八道流水线 -> 长度差不多的数据放在一起
def sort(data, configs):
    """Sort the data by feature length.
    Sort is used after shuffle and before batch, so we can group
    utts with similar lengths into a batch, and `sort_size` should
    be less than `shuffle_size`

    Args:
        data: Iterable[{key, feat, label}]
        sort_size: buffer size for sort

    Returns:
        Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        try:
            buf.append(sample)
            if len(buf) >= configs.sort_size:
                buf.sort(key=lambda x: x["spec"].size(1))  # 看一下spec的维度
                for x in buf:
                    yield x
                buf = []
        except Exception as ex:
            print("Failed to sort {}, ex info {}".format(sample["utt"], ex))
    # The sample left over
    buf.sort(key=lambda x: x["spec"].size(1))
    for x in buf:
        yield x


# 第九道流水线
def batch(data,
          configs,
          batch_type='dynamic',
          max_frames_in_batch=3500,
          batch_size=1):
    """Wrapper for static/dynamic batch

    配置文件默认 ->  batch_type: dynamic, max_frames_in_batch: 2000
    """
    if batch_type == "static":
        return static_batch(data, batch_size)
    elif batch_type == "dynamic":
        return dynamic_batch(data,
                             configs,
                             max_frames_in_batch=max_frames_in_batch)
    else:
        print("Unsupported batch type {}".format(configs.batch_type))


def static_batch(data, batch_size=1):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, configs, max_frames_in_batch=12000):
    """Dynamic batch the data until the total frames in batch
    reach `max_frames_in_batch`

    Args:
        data: Iterable[{key, feat, label}]
        max_frames_in_batch: max_frames in one batch

    Returns:
        Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        try:
            assert "spec" in sample
            assert isinstance(sample["spec"], torch.Tensor)
            new_sample_frames = sample["spec"].size(1)
            longest_frames = max(longest_frames, new_sample_frames)
            frames_after_padding = longest_frames * (
                len(buf) + 1)  # 乘以batch_size表示当前batch的总帧数
            if frames_after_padding > max_frames_in_batch:
                yield buf
                buf = [sample]
                longest_frames = new_sample_frames
            else:
                buf.append(sample)
        except Exception as ex:
            print("Failed to dynamic batch {}, ex info {}".format(
                sample["utt"], ex))
    if len(buf) > 0:
        yield buf


# 第十道流水线 -> 填充长度
def padding(data, configs):
    """Padding the data into training data

    Args:
        data: Iterable[List[{key, feat, label}]]
    # c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded, volume_padded
    Returns:
        Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        try:
            assert isinstance(sample, list)
            ssl_feature_len = torch.tensor(
                [x["ssl_feature"].size(1) for x in sample],
                dtype=torch.int32)  # 每个样本的mel长度
            order = torch.argsort(ssl_feature_len, descending=True)  # 降序

            utts = [sample[i]["utt"] for i in order]

            spec = [sample[i]["spec"].transpose(1, 0)
                    for i in order]  # [t, num_mels]
            spec = pad_sequence(spec, batch_first=True, padding_value=0)

            # ssl
            ssl_feature = [
                sample[i]["ssl_feature"].transpose(1, 0) for i in order
            ]
            ssl_length = torch.tensor([i.size(0) for i in ssl_feature],
                                      dtype=torch.int32)  #
            ssl_feature = pad_sequence(ssl_feature,
                                       batch_first=True,
                                       padding_value=0)
            # f0
            f0 = [sample[i]["f0"] for i in order]
            f0 = pad_sequence(f0, batch_first=True, padding_value=0)

            # uv_padded
            uv = [sample[i]["uv"] for i in order]
            uv = pad_sequence(uv, batch_first=True, padding_value=0)

            # volume_padded
            volume = [sample[i]["volume"] for i in order]
            volume = pad_sequence(volume, batch_first=True, padding_value=0)

            # wav
            audio = [sample[i]["audio"].squeeze(0) for i in order]
            audio = pad_sequence(audio, batch_first=True, padding_value=0)

            # spk
            spk_id = [sample[i]["spk_id"] for i in order]
            spk_id = torch.tensor(spk_id, dtype=torch.int64)

            batch = {
                "utts": utts,
                "spec": spec.permute(0, 2, 1),  # [b, t, nff]
                "ssl_feature": ssl_feature.permute(0, 2, 1),
                "ssl_length": ssl_length,
                "f0": f0,
                "uv": uv,
                "volume": volume,
                "audio": audio.unsqueeze(1),
                "spk_id": spk_id.unsqueeze(1),
            }
            yield batch
        except Exception as ex:
            print("Failed to padding {}, ex info {}".format(sample, ex))
