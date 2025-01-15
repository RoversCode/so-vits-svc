#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train_index.py
@Time    :   2024/12/30 00:26:50
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import pickle
import utils
from pathlib import Path


if __name__ == "__main__":
    hps = utils.get_hparams()
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--root_dir", type=str, default="dataset/44k", help="path to root dir"
    # )
    # parser.add_argument('-c', '--config', type=str, default="./configs/config.json",
    #                 help='JSON file for configuration')
    # parser.add_argument(
    #     "--output_dir", type=str, default="logs/44k", help="path to output dir"
    # )

    # args = parser.parse_args()
    # hps = utils.get_hparams_from_file(args.config)
    spk_dic = hps.data.spk
    result = {}

    for k, v in spk_dic.items():
        print(f"now, index {k} feature...")
        index = utils.train_index(k, Path("ckpts")/hps.args.exp_name/'audio_slice')
        result[v] = index
    output_dir = Path("ckpts")/hps.args.exp_name/'index'/'feature_and_index.pkl'
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(output_dir, "wb") as f:
        pickle.dump(result, f)
