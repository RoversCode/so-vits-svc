#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   gener_spk_info.py
@Time    :   2024/09/09 09:45:32
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import os
from tqdm import tqdm
import torch
import json
import torch.multiprocessing as mp
import numpy as np


def traverse_dir_files(root_dir, ext="dic"):
    paths_list = []
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith("."):  # 去除隐藏文件
                continue
            if ext:  # 根据后缀名搜索
                if name.endswith(tuple(ext)):
                    # names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                # names_list.append(name)
                paths_list.append(os.path.join(parent, name))

    return paths_list


def cosine_similarity_batch_torch(arr1, arr2):
    # 确保输入的张量形状为[batch_size, feature_size]
    if arr1.shape != arr2.shape:
        arr1 = arr1[: arr2.shape[0], :]

    # 计算余弦相似度
    return torch.nn.functional.cosine_similarity(arr1, arr2)


def process_batch(spk1, batch_files, processed, device):
    try:
        spk1 = spk1.to(device)
        names = [
            p
            for p in batch_files
            if p.split("/")[-1].replace(".npy", "") not in processed
        ]
        spk2s = [np.load(p) for p in names]
        if len(spk2s) == 0:
            return [], []
        spk2s = torch.from_numpy(np.squeeze(np.stack(spk2s, axis=0))).to(device)
        if len(spk2s.shape) == 1:
            spk2s = spk2s.unsqueeze(0)
        x = cosine_similarity_batch_torch(spk1, spk2s)
        indexs = torch.where(x > 0.75)[0]
        similar_names = [
            names[index].split("/")[-1].replace(".npy", "") for index in indexs
        ]
        return similar_names
    except Exception as e:
        print(f"报错了: {e}")
        return [], []


def process_results(results):
    for result in results:
        similar_names = result
        for name in similar_names:
            if name not in processed:
                spk_info[spk_name].append(name)
                new_processed[name] = True

all_files = traverse_dir_files('/mnt/e/Workspace/growth/audio/so-vits-svc/preprocess/raw_data/timbre', ext='npy')

if __name__ == "__main__":
    import time

    # Set the start method to 'spawn'
    mp.set_start_method("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        spk_info = json.load(
            open(
                "/mnt/e/Workspace/growth/audio/so-vits-svc/preprocess/raw_data/spk_info.json"
            )
        )
        processed = json.load(
            open(
                "/mnt/e/Workspace/growth/audio/so-vits-svc/preprocess/raw_data/processed.json"
            )
        )
    except:
        spk_info = {}
        processed = {}
    print(f"Original data count: {len(all_files)}")
    all_files = [
        path
        for path in all_files
        if path.split("/")[-1].replace(".npy", "") not in processed
        and path.split("/")[-1].replace(".npy", "") not in spk_info
    ]
    print(f"Remaining data count: {len(all_files)}")
    new_processed = {}
    num_processes = 20  # 指定进程数
    save_iter = 2
    with mp.Pool(processes=num_processes) as pool:
        for i in range(len(all_files)):
            print(f"i: {i}/{len(all_files)}")
            spk_name = all_files[i].split("/")[-1].replace(".npy", "")
            if spk_name in new_processed:
                continue
            if spk_name not in spk_info:
                spk_info[spk_name] = [spk_name]
            else:
                continue
            new_processed[spk_name] = True
            sub_file = all_files[i + 1 :]
            batch_size = 128
            spk1 = torch.from_numpy(np.load(all_files[i]))
            spk1 = spk1.repeat(batch_size, 1)
            start = time.perf_counter()
            async_result = pool.starmap_async(
                process_batch,
                [
                    (spk1, sub_file[j : j + batch_size], new_processed, device)
                    for j in range(i + 1, len(sub_file), batch_size)
                ],
                callback=process_results,
            )
            async_result.wait()  # 等待任务完成
            end = time.perf_counter()
            print(f"spk_name: {spk_name}, time: {end - start}")
            if i % save_iter == 0:
                processed.update(new_processed)
                json.dump(
                    spk_info, open("raw_data/spk_info.json", "w"), ensure_ascii=False, indent=4
                )
                json.dump(
                    processed, open("raw_data/processed.json", "w"), ensure_ascii=False, indent=4
                )
        processed.update(new_processed)
        json.dump(spk_info, open("raw_data/spk_info.json", "w"), ensure_ascii=False, indent=4)
        json.dump(processed, open("raw_data/processed.json", "w"), ensure_ascii=False, indent=4)
