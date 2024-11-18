#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_utils.py
@Time    :   2024/11/18 22:48:48
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
'''

# here put the import lib
from functools import partial
from tools.metadataset import QueueDatasetPipeline
from tools.ft_metadataset import Dataset
from tools.data_pipeline import (
    filter,
    gen_spec,
    gen_vol,
    align,
    shuffle,
    sort,
    batch,
    padding,
)
from torch.utils.data import DataLoader

def init_dataloader(hps, rank):
    if hps.train.train_type == "base":
        train_dataset = QueueDatasetPipeline(
            queue_address=("127.0.0.1", 12345),
            queue_name="get_train_queue",
            data_pipeline=[
                filter,
                gen_spec,
                gen_vol,
                align,
                shuffle,
                sort,
                partial(
                    batch,
                    batch_type=hps.data.batch_type,
                    max_frames_in_batch=hps.data.max_frames_in_batch,
                ),
                padding,
            ],
            configs=hps.data,
        )
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=None,
            pin_memory=hps.train.pin_memory,
            num_workers=hps.train.num_workers,
            prefetch_factor=hps.train.prefetch,
        )

        if rank == 0:
            hps.data.batch_type = "static"
            hps.data.batch_size = 1
            dev_dataset = QueueDatasetPipeline(
                queue_address=("127.0.0.1", 12345),
                queue_name="get_dev_queue",
                data_pipeline=[
                    filter,
                    gen_spec,
                    gen_vol,
                    align,
                    partial(
                        batch,
                        batch_type='static',
                        batch_size=1,
                    ),
                    padding,
                ],
                configs=hps.data,
            )
            dev_data_loader = DataLoader(
                dev_dataset,
                batch_size=None,
                # pin_memory=hps.train.pin_memory,
                # num_workers=hps.train.num_workers,
                # prefetch_factor=hps.train.prefetch,
            )
        else:
            dev_data_loader = None
            dev_dataset = None
    else:
        train_dataset = Dataset(
            hps.data.training_files,
            data_pipeline=[
                filter,
                gen_spec,
                gen_vol,
                align,
                shuffle,
                sort,
                partial(
                    batch,
                    batch_type=hps.data.batch_type,
                    max_frames_in_batch=hps.data.max_frames_in_batch,
                ),
                padding,
            ],
            configs=hps.data,
        )
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=None,
            pin_memory=hps.train.pin_memory,
            num_workers=hps.train.num_workers,
            prefetch_factor=hps.train.prefetch,
        )
        if rank == 0:
            hps.data.batch_type = "static"
            hps.data.batch_size = 1
            dev_dataset = Dataset(
                hps.data.validation_files,
                data_pipeline=[
                    filter,
                    gen_spec,
                    gen_vol,
                    align,
                    partial(
                        batch,
                        batch_type='static',
                        batch_size=1,
                    ),
                    padding,
                ],
                configs=hps.data,
            )
            dev_data_loader = DataLoader(
                dev_dataset,
                batch_size=None,
                # pin_memory=hps.train.pin_memory,
                # num_workers=hps.train.num_workers,
                # prefetch_factor=hps.train.prefetch,
            )
        else:
            dev_data_loader = None
            dev_dataset = None
    return train_dataset, train_data_loader, dev_dataset, dev_data_loader