#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   metadataset.py
@Time    :   2024/11/13 21:40:25
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
'''

# here put the import lib
import random
import json
import math
import queue
import torch
import time
import torch.distributed as dist
from functools import partial
from torch.utils.data import IterableDataset
from multiprocessing.managers import BaseManager

class QueueManager(BaseManager):
    pass


def QueueDatasetPipeline(
    queue_address=("127.0.0.1", 12345),
    queue_authkey=b"liujunjieabracadabra",
    queue_name='get_train_queue',
    data_pipeline=None,
    configs=None,
    partition=True,
    max_retries=5,
    retry_delay=5,
    buffer_size=1000,
):
    """构造支持分布式训练的队列数据集处理流水线"""
    
    # 创建队列数据集
    dataset = QueueDataset(
        address=queue_address,
        authkey=queue_authkey,
        queue_name=queue_name,
        partition=partition,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    dataset.buffer_size = buffer_size

    # 应用数据处理流水线
    for func in data_pipeline:
        dataset = Processor(dataset, func, configs=configs)

    return dataset


class DistributedSampler:

    def __init__(self, partition=True):
        self.epoch = -1
        self.update()
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
            self.worker_id = worker_info.id  # 当前进程的rank
            self.num_workers = worker_info.num_workers  # 总进程数
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch


class QueueDataset(IterableDataset):
    """从Queue Manager接收数据的数据集"""

    """支持分布式训练的队列数据集"""

    def __init__(
        self,
        address=("", 6789),
        authkey=b"liujunjieabracadabra",
        queue_name='get_train_queue',
        partition=True,
        max_retries=5,
        retry_delay=5,
        buffer_size=1000,
    ):
        super().__init__()
        self.address = address
        self.authkey = authkey
        self.sampler = DistributedSampler(partition)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.manager = None
        self.queue = None
        self.buffer_size = buffer_size  # 缓冲区大小
        self.queue_name = queue_name

    def connect(self):
        """建立与Queue Manager的连接"""
        if self.manager is not None:
            try:
                self.manager.shutdown()
            except:
                pass
        retries = 0
        while retries < self.max_retries:
            try:
                QueueManager.register(self.queue_name)
                self.manager = QueueManager(address=self.address, authkey=self.authkey)
                self.manager.connect()
                self.queue = getattr(self.manager, self.queue_name)()
                print("Successfully connected to Queue Manager")
                return True
            except Exception as e:
                retries += 1
                print(f"Connection attempt {retries} failed: {str(e)}")
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
                continue

        raise ConnectionError(f"Failed to connect after {self.max_retries} attempts")

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        """返回迭代器，确保数据被正确分片到各个GPU"""
        sampler_info = self.sampler.update()

        if self.queue is None:
            self.connect()  # 获取链接
        while True:
            try:
                # 抽出一个蓄水池，往pipeline里面传递
                sample = self.queue.get()
                sample.update(sampler_info)
                yield sample
                
            except Exception as e:
                print(f"Error in iteration: {str(e)}")
                # 尝试重连
                if self.queue is None:
                    try:
                        print("Attempting to reconnect to Queue Manager")
                        self.connect()
                    except:
                        assert False, "Failed to reconnect to Queue Manager"
                continue


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