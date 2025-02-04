#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_structure.py
@Time    :   2024/11/11 20:52:54
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
sys.path.append(parent_dir)
import numpy as np
import librosa
import soundfile
import json
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser


def main(args):
    """
    # 切片并重新命名
    """
    audio_dir = Path(args.audio_dir)
    # 创建切片文件夹
    output_dir = audio_dir.parent / "audio_slice"
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建文件夹
    spk_info = {}
    # 检测里面是否含有音频文件夹
    subdirs = [x.name for x in audio_dir.iterdir() if x.is_dir()]
    print(subdirs)
    if len(subdirs) == 0:
        # 创建一个audio_dir.parent.name的文件夹
        (audio_dir / audio_dir.parent.name).mkdir(parents=True, exist_ok=True)
        subdirs = [audio_dir.parent.name]

    slicer = Slicer(
        sr=args.sr,
        threshold=args.db_thresh,
        min_length=args.sliced_min_length,
        audio_min_length=args.audio_min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept,
    )
    slice_grain = Slicer(
        sr=args.sr,
        threshold=args.db_thresh,
        min_length=args.sliced_min_length,
        min_interval=args.min_interval * 0.5,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept,
    )
    # 遍历说话人文件夹
    for spk_name in tqdm(subdirs):
        spk_info[spk_name] = len(spk_info)
        dir_path = audio_dir / spk_name
        out_spk_dir = output_dir / spk_name
        out_spk_dir.mkdir(parents=True, exist_ok=True)  # 创建文件夹
        files = dir_path.glob("*.wav")  # 仅支持wav格式，这个给的完整路径
        file_idx = 0  # 文件序列
        for file_name in files:
            audio, sr = librosa.load(file_name, sr=args.sr)
            chunks = slicer.slice(audio)
            for chunk in chunks:
                # NOTE: 如果chunk大于10s，下面硬切硬切5s
                time_inter = 10
                max_length = 15
                if len(chunk) > max_length * sr:
                    second_chunks = slice_grain.slice(chunk)
                    for c in second_chunks:
                        if len(c) > max_length * sr:
                            num = len(c) // (time_inter * sr)
                            for n in range(num):
                                soundfile.write(
                                    out_spk_dir / f"{spk_name}_{file_idx:06d}.wav",
                                    c[n * time_inter * sr : (n + 1) * time_inter * sr],
                                    sr,
                                )
                                file_idx += 1
                            if (
                                len(c[num * time_inter * sr :])
                                > args.audio_min_length * sr
                            ):
                                soundfile.write(
                                    out_spk_dir / f"{spk_name}_{file_idx:06d}.wav",
                                    c[num * time_inter * sr :],
                                    sr,
                                )
                                file_idx += 1
                        else:
                            soundfile.write(
                                out_spk_dir / f"{spk_name}_{file_idx:06d}.wav",
                                c,
                                sr,
                            )
                            file_idx += 1
                else:
                    soundfile.write(
                        out_spk_dir / f"{spk_name}_{file_idx:06d}.wav",
                        chunk,
                        sr,
                    )
                    file_idx += 1

    # 保存说话人信息
    json.dump(
        spk_info,
        open(audio_dir.parent / "spk_info.json", "w"),
        ensure_ascii=False,
        indent=4,
    )


def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        audio_min_length: int = 3000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 500,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )
        self.sr = sr
        self.audio_min_length = audio_min_length / 1000
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(
            sr * min_length / 1000 / self.hop_size
        )  # NOTE: 能被切片的最小长度
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[
                :, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)
            ]
        else:
            return waveform[
                begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)
            ]

    # @timeit
    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if (samples.shape[0] + self.hop_size - 1) // self.hop_size <= self.min_length:
            return [waveform]
        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(
            0
        )  # NOTE: 每帧的能量
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:  # NOTE: 分贝
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start
                >= self.min_interval  # NOTE: 静音片段超过设置min_interval
                and i - clip_start
                >= self.min_length  # NOTE: 被切分的片段超过设置min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None  # 不用切，继续走
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:  # 如果静音片段小于max_sil_kept
                pos = (
                    rms_list[silence_start : i + 1].argmin() + silence_start
                )  # 寻找从silence_start到i这一段时间内，哪个位置的音量最小，也就是最静的地方。
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif (
                i - silence_start <= self.max_sil_kept * 2
            ):  # 如果静音片段小于max_sil_kept*2
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    self._apply_slice(waveform, sil_tags[-1][1], total_frames)
                )
            # 如果单个片段小于audio_min_length，就合并到前一个片段
            new_chunks = []
            curent_idx = 0
            while curent_idx < len(chunks):
                chunk = chunks[curent_idx]
                while len(chunk) < self.audio_min_length * self.sr:
                    if curent_idx + 1 < len(chunks):
                        chunk = np.concatenate([chunk, chunks[curent_idx + 1]])
                        curent_idx += 1
                    else:
                        break
                new_chunks.append(chunk)
                curent_idx += 1

            return new_chunks


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--audio_dir", type=str, help="The audio to be sliced")
    parser.add_argument("--sr", type=int, default=44100, help="The audio to be sliced")
    parser.add_argument(
        "--db_thresh",
        type=float,
        required=False,
        default=-40,
        help="The dB threshold for silence detection",
    )
    parser.add_argument(
        "--sliced_min_length",
        type=int,
        required=False,
        default=10000,
        help="当一个音频片段大于这个值，这个片段才会被切分",
    )
    parser.add_argument(
        "--audio_min_length",
        type=int,
        required=False,
        default=2000,
        help="切片后的音频片段最小长度",
    )
    parser.add_argument(
        "--min_interval",
        type=int,
        required=False,
        default=300,
        help="静音部分被切分的最小毫秒数",
    )
    parser.add_argument(
        "--hop_size",
        type=int,
        required=False,
        default=10,
        help="Frame length in milliseconds",
    )
    parser.add_argument(
        "--max_sil_kept",
        type=int,
        required=False,
        default=500,
        help="切片两端能保留静音的最大长度，毫秒为单位",
    )
    args = parser.parse_args()
    main(args)
