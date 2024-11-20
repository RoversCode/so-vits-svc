#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   diff_data_pipeline.py
@Time    :   2024/11/19 19:49:32
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import torch
import json
import torchaudio
import torch
import random
import numpy as np
import utils
from io import BytesIO
from torch.nn.utils.rnn import pad_sequence
from vdecoder.nsf_hifigan.models import load_config
from vdecoder.nsf_hifigan.nvSTFT import STFT

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
            # 检查filter是否存在变量spk2id
            # spk_id
            if "spk" in configs.data:
                sample["spk_id"] = configs.data.spk[sample["spk"]]
            else:
                sample["spk_id"] = spk2id[sample["spk"]]
            yield {**sample}

        except Exception as ex:
            print("Failed to filter {}, ex info {}".format(sample["utt"], ex))


def gen_spec(data, configs):
    """Give url or local file, return file descriptor"""
    vocoder_config = load_config(configs.vocoder.ckpt)
    stft = STFT(
        vocoder_config.sampling_rate,
        vocoder_config.num_mels,
        vocoder_config.n_fft,
        vocoder_config.win_size,
        vocoder_config.hop_size,
        vocoder_config.fmin,
        vocoder_config.fmax,
    )
    for sample in data:
        try:
            # vocoder的mel
            audio, sr = torchaudio.load(BytesIO(sample["audio_data"]))
            if audio.dim() == 2 and audio.size(0) > 1:
                audio = audio.mean(0, keepdim=True)  # 单通道
            if sr != configs.data.sampling_rate:
                audio = torchaudio.transforms.Resample(
                    sr, vocoder_config.sampling_rate
                )(audio)
            audio = audio / torch.max(torch.abs(audio))
            del sample["audio_data"]
            sample["audio"] = audio

            mel = stft.get_mel(audio, keyshift=0).transpose(1, 2)  # B, n_frames, bins
            sample["mel_spec"] = mel.squeeze()

            # aug mel
            max_amp = float(torch.max(torch.abs(audio))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            keyshift = random.uniform(-5, 5)
            aug_mel_t = stft.get_mel(audio * (10**log10_vol_shift), keyshift=keyshift).transpose(1, 2)
            
            sample['keyshift'] = keyshift
            sample["aug_mel_spec"] = aug_mel_t.squeeze()
            sample["log10_vol_shift"] = log10_vol_shift
            yield sample

        except Exception as ex:
            print("Failed to gen_spec {}, ex info {}".format(sample["utt"], ex))


def gen_vol(data, configs):
    def extract_volume(audio, hop_length):
        n_frames = int(audio.size(-1) // hop_length)
        audio2 = audio**2
        audio2 = torch.nn.functional.pad(
            audio2,
            (int(hop_length // 2), int((hop_length + 1) // 2)),
            mode="reflect",
        )
        volume = torch.nn.functional.unfold(
            audio2[:, None, None, :],
            (1, hop_length),
            stride=hop_length,
        )[:, :, :n_frames].mean(dim=1)[0]
        volume = torch.sqrt(volume)
        return volume

    for sample in data:
        try:
            audio = sample["audio"]
            sample["volume"] = extract_volume(audio, configs.data.hop_length)

            # aug vol
            log10_vol_shift = sample["log10_vol_shift"]
            sample["aug_volume"] = extract_volume(
                audio * (10**log10_vol_shift), configs.data.hop_length
            )
            del sample["log10_vol_shift"]
            yield sample
        except Exception as ex:
            print("Failed to gen_vol {}, ex info {}".format(sample["utt"], ex))


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
            if len(buf) >= configs.data.shuffle_size:
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
            if len(buf) >= configs.data.sort_size:
                buf.sort(key=lambda x: x["mel_spec"].size(0))  # 看一下spec的维度
                for x in buf:
                    yield x
                buf = []
        except Exception as ex:
            print("Failed to sort {}, ex info {}".format(sample["utt"], ex))
    # The sample left over
    buf.sort(key=lambda x: x["mel_spec"].size(1))
    for x in buf:
        yield x


def batch(data, configs, batch_type="dynamic", max_frames_in_batch=3500, batch_size=1):
    """Wrapper for static/dynamic batch

    配置文件默认 ->  batch_type: dynamic, max_frames_in_batch: 2000
    """
    if batch_type == "static":
        return static_batch(data, batch_size)
    elif batch_type == "dynamic":
        return dynamic_batch(data, configs, max_frames_in_batch=max_frames_in_batch)
    else:
        print("Unsupported batch type {}".format(configs.data.batch_type))


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
            assert "mel_spec" in sample
            assert isinstance(sample["mel_spec"], torch.Tensor)
            new_sample_frames = sample["mel_spec"].size(0)
            longest_frames = max(longest_frames, new_sample_frames)
            frames_after_padding = longest_frames * (
                len(buf) + 1
            )  # 乘以batch_size表示当前batch的总帧数
            if frames_after_padding > max_frames_in_batch:
                yield buf
                buf = [sample]
                longest_frames = new_sample_frames
            else:
                buf.append(sample)
        except Exception as ex:
            print("Failed to dynamic batch {}, ex info {}".format(sample["utt"], ex))
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
        
            mel=mel,
            f0=f0_frames,
            volume=volume_frames,
            units=units,
            spk_id=spk_id,
            aug_shift=aug_shift,
            name=name,
            name_ext=name_ext,
    """
    for sample in data:
        try:
            assert isinstance(sample, list)
            ssl_feature_len = torch.tensor(
                [x["mel_spec"].size(0) for x in sample], dtype=torch.int32
            )  # 每个样本的mel长度
            order = torch.argsort(ssl_feature_len, descending=False)  # 降序
            # dur = [sample[i]["dur"] for i in order]
            keyshift = [sample[i]["keyshift"] for i in order]
            # 随机增强
            if random.uniform(0, 1) < 0.5:
                aug_flag = True
                # vol mel 
                mel_spec = [sample[i]["aug_mel_spec"] for i in order]  # [t, num_mels]
                volume = [sample[i]["aug_volume"] for i in order]
            else:
                aug_flag = False
                mel_spec = [sample[i]["mel_spec"] for i in order]
                volume = [sample[i]["volume"] for i in order]
                
            f0 = [torch.FloatTensor(sample[i]["f0"].astype(np.float32)) for i in order]
            ssl = [sample[i]["ssl_feature"] for i in order]  # [768, t]
            frame_num = mel_spec[0].shape[0]
            
            for i in range(0, len(sample)):
                cur_frame_num = mel_spec[i].shape[0]
                if cur_frame_num == frame_num:  # 等长
                    if aug_flag:
                        f0[i] = 2 ** (keyshift[i] / 12) * f0[i]
                    continue
                # 随机筛选frame_num
                start_idx = np.random.randint(0, cur_frame_num-frame_num)
                
                mel_spec[i] = mel_spec[i][start_idx:start_idx+frame_num, :]
                volume[i] = volume[i][start_idx:start_idx+frame_num]
                ssl[i] = ssl[i][:, start_idx:start_idx+frame_num]
                if aug_flag:
                    f0[i] = 2 ** (keyshift[i] / 12) * f0[i][start_idx : start_idx + frame_num]
                else:
                    f0[i] = f0[i][start_idx:start_idx+frame_num]
            spk_id = [sample[i]["spk_id"] for i in order]
            spk_id = torch.tensor(spk_id, dtype=torch.int64)
            
            keyshift = torch.tensor(keyshift, dtype=torch.float32)
            f0 = torch.stack(f0, dim=0)
            ssl = torch.stack(ssl, dim=0)
            volume = torch.stack(volume, dim=0)
            mel_spec = torch.stack(mel_spec, dim=0)
            
            batch = {
                "mel_spec": mel_spec,  # [b, t, nff]
                "ssl_feature": ssl,
                "f0": f0,
                "volume": volume,
                "spk_id": spk_id.unsqueeze(1),
                "keyshift": keyshift.unsqueeze(1).unsqueeze(1),
            }
            yield batch
        except Exception as ex:
            print("Failed to padding {}, ex info {}".format(sample, ex))
