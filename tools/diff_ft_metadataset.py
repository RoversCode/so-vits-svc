#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   diff_metadataset.py
@Time    :   2024/11/19 20:20:18
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import random
import math
import torch
import librosa
import numpy as np
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import IterableDataset
from multiprocessing.managers import BaseManager

# 跟sovits隔开是因为，不想耦合那么严重，方便后续调整


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """Sample data according to rank/world_size/num_workers

        Args:
            data(List): input data list

        Returns:
            List: data list after sample
        """
        data = list(range(len(data)))
        # force datalist even
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            if len(data) < self.world_size:
                data = data * math.ceil(self.world_size / len(data))
                data = data[: self.world_size]
            data = data[self.rank :: self.world_size]
        if len(data) < self.num_workers:
            data = data * math.ceil(self.num_workers / len(data))
            data = data[: self.num_workers]
        data = data[self.worker_id :: self.num_workers]
        return data


import utils


class DataList(IterableDataset):

    def __init__(self, lists, configs, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)
        self.configs = configs

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            index = Path(self.lists[index])
            sample = {}
            with open(index, "rb") as f:
                sample["audio_data"] = f.read()

            # f0
            sample["f0"], sample["uv"] = np.load(
                index.parent / ("_".join(index.stem.split("_")[:2]) + "_f0.npy"),
                allow_pickle=True,
            )
            # ssl
            ssl = torch.load(
                index.parent / ("_".join(index.stem.split("_")[:2]) + "_ssl.pt")
            )
            ssl = utils.repeat_expand_2d(
                ssl.squeeze(0),
                sample["f0"].shape[0],
                self.configs.data.unit_interpolate_mode,
            )
            sample["ssl_feature"] = ssl

            # dur ，获取index 音频时长
            wav, sr = librosa.load(index, sr=None)
            sample["dur"] = len(wav) / sr
            # spk
            sample["spk"] = index.stem.split("_")[0]
            # utt
            sample["utt"] = index.name
            # audio_data
            sample.update(sampler_info)
            yield sample


def Dataset(data_list_file, data_pipeline, configs, shuffle=True, partition=True):
    """Construct dataset from arguments

    We have two shuffle stage in the Dataset. The first is global
    shuffle at shards tar/raw file level. The second is global shuffle
    at training samples level.

    Args:
        data_type(str): raw/shard
        tokenizer (BaseTokenizer): tokenizer to tokenize
        partition(bool): whether to do data partition in terms of rank
    """
    # 读取数据
    lists = []
    with open(data_list_file) as f:
        for line in f:
            lists.append(line.strip())

    dataset = DataList(lists, configs, shuffle=shuffle, partition=partition)
    for func in data_pipeline:
        dataset = Processor(dataset, func, configs=configs)

    return dataset


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """Return an iterator over the source dataset processed by the
        given processor.
        self.source是上一个Processor对象(或者是最初的DataList对象),self.f是当前的数据处理函数。通过调用iter(self.source),我们获得了上一个Processor对象的迭代器。然后,我们将这个迭代器作为参数传递给当前的数据处理函数self.f,并将处理后的结果返回。
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)
