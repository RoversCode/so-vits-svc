#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/11/19 16:50:35
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import json
import os
import torch
import yaml
from pathlib import Path


def traverse_dir(
    root_dir,
    extensions,
    amount=None,
    str_include=None,
    str_exclude=None,
    is_pure=False,
    is_sort=False,
    is_ext=True,
):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir) + 1 :] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list

                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue

                if not is_ext:
                    ext = pure_path.split(".")[-1]
                    pure_path = pure_path[: -(len(ext) + 1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 递归转换所有嵌套的字典为 DotDict
        for key, value in self.items():
            if type(value) is dict:
                self[key] = DotDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        # 如果设置的值是字典，递归转换为 DotDict
        if type(value) is dict:
            value = DotDict(value)
        self[key] = value

    __delattr__ = dict.__delitem__


def get_network_paras_amount(model_dict):
    info = dict()
    for model_name, model in model_dict.items():
        # all_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info[model_name] = trainable_params
    return info


def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    # print(args)
    return args


def save_config(path_config, config):
    config = dict(config)
    with open(path_config, "w") as f:
        yaml.dump(config, f)


def to_json(path_params, path_json):
    params = torch.load(path_params, map_location=torch.device("cpu"))
    raw_state_dict = {}
    for k, v in params.items():
        val = v.flatten().numpy().tolist()
        raw_state_dict[k] = val

    with open(path_json, "w") as outfile:
        json.dump(raw_state_dict, outfile, indent="\t")


def convert_tensor_to_numpy(tensor, is_squeeze=True):
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def load_model(model_dir, model, optimizer):
    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)

    f_list = list(model_dir.glob("*.pt"))
    f_list.sort(
        key=lambda f: int("".join(filter(str.isdigit, f)))
    )  # 根据提取的数据，排列


    if len(f_list) > 0:
        ckpt_path = f_list[-1]
        print(" [*] restoring model from", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt["model"], strict=True)
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
    else:
        global_step = 1

    return global_step, model, optimizer
