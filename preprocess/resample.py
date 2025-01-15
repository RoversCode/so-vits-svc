#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   resample.py
@Time    :   2024/10/09 15:51:16
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import sys

sys.path.append("/datadisk/liujunjie/cmm-live-tts")
import os
import librosa
import soundfile
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser


def process(item):
    root_dir, wav_name, output_dir, sr = item
    wav_path = os.path.join(root_dir, wav_name)
    target_path = os.path.join(output_dir, wav_name)
    try:
        if os.path.exists(wav_path) and wav_name.endswith(".wav"):
            wav, sr = librosa.load(wav_path, sr=sr)
            soundfile.write(os.path.join(output_dir, wav_name), wav, sr)
    except Exception as e:

        print(f"{wav_path} error! {e}")


def main(args):
    root_dir = args.audio_dir
    out_dir = root_dir
    sr = args.sr
    processes = args.processes
    pool = Pool(processes=processes)

    tasks = []
    files = os.listdir(root_dir)
    for file_name in files:
        twople = (root_dir, file_name, out_dir, sr)
        tasks.append(twople)
    for _ in tqdm(
        pool.imap_unordered(process, tasks),
    ):
        pass

    pool.close()
    pool.join()

    # 修改filelist
    print("音频重采样完毕!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True,  help="The audio to be sliced")
    parser.add_argument("--sr", type=int, required=True, help="The audio to be sliced")
    parser.add_argument("--processes", type=int, default=5, help="The audio to be sliced")
    args = parser.parse_args()
    main(args)
