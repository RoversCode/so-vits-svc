#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train_diff.py
@Time    :   2024/11/17 21:26:42
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import logging
import datetime
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from diffusion.unit2mel import Unit2Mel
from utils import get_hparams, get_logger, latest_checkpoint_path, load_checkpoint
from tools.diff_train_utils import init_dataloader
from vdecoder.nsf_hifigan.models import load_config
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

global_step = 0


def run():
    global global_step
    hps = get_hparams()
    for env_name, env_value in hps.train.train_env.items():  # Debug
        if env_name not in os.environ.keys():
            print("加载config中的配置{}".format(str(env_value)))
            os.environ[env_name] = str(env_value)
    print(
        "加载环境变量 \nMASTER_ADDR: {},\nMASTER_PORT: {},\nWORLD_SIZE: {},\nRANK: {},\nLOCAL_RANK: {}".format(
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
            os.environ["LOCAL_RANK"],
        )
    )

    # for pytorch on win, backend use gloo
    dist.init_process_group(
        backend="nccl" if os.name == "nt" else "nccl",
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )
    rank = dist.get_rank()
    # local_rank = int(os.environ["LOCAL_RANK"])
    # n_gpus = dist.get_world_size()
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    if rank == 0:
        logger = get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir / "tensorboard")
        writer_eval = SummaryWriter(log_dir=hps.model_dir / "tensorboard/eval")
    else:
        logger = None
        writer = None
        writer_eval = None

    # datasets
    train_dataset, train_data_loader, dev_dataset, dev_data_loader = init_dataloader(
        hps, 0
    )

    try:
        if hps.train.train_type == "base":
            spk2id = json.load(open("Data/sovits_svc/sing_spk_info.json", "r"))
            hps.model.n_speakers = len(spk2id)
            del spk2id
        else:
            hps.model.n_speakers = len(hps.data.spk)
    except Exception:
        pass

    # vocoder_config
    vocoder_config = load_config(hps.vocoder.ckpt)

    # load model
    model = Unit2Mel(
        hps.data.encoder_out_channels,
        hps.model.n_spk,
        hps.model.use_pitch_aug,
        vocoder_config.num_mels,
        hps.model.n_layers,
        hps.model.n_chans,
        hps.model.n_hidden,
        hps.model.timesteps,
        hps.model.k_step_max,
    ).cuda(rank)

    if rank == 0:
        logger.info(
            f" > Now model timesteps is {model.timesteps}, and k_step_max is {model.k_step_max}"
        )

    # load parameters
    optimizer = torch.optim.AdamW(model.parameters())

    try:
        model_path = latest_checkpoint_path(hps.model_dir, "diff_*.pt")
        _, _, epoch_str = load_checkpoint(
            model_path,
            model,
            optimizer,
            hps.train.skip_optimizer,
        )
        global_step = int(model_path.stem.split("_")[1]) + 1
    except:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0

    # 检查可训练参数
    if rank == 0:  # NOTE：计算目前要训练的东西
        all_params = 0
        total_params = 0
        for net in [model]:
            print(
                f"Model: {net.__class__.__name__}, params: {sum(p.numel() for p in net.parameters())}"
            )
            total_params += sum(p.numel() for p in net.parameters() if p.requires_grad)
            all_params += sum(p.numel() for p in net.parameters())

        print(f"全部参数: {all_params}")
        print(f"要训练参数: {total_params}")

    model = DDP(model, device_ids=[rank])

    # scheduler = lr_scheduler.ExponentialLR(
    #     optimizer,
    #     gamma=hps.train.gamma,
    #     last_epoch=epoch_str - 2,  # 如果epoch是从1开始计数
    # )
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=100000, gamma=hps.train.gamma, last_epoch=epoch_str - 2
    )
    scaler = GradScaler(enabled=hps.train.fp16_run)
    for epoch in range(epoch_str, hps.train.epochs):
        train_dataset.set_epoch(epoch)  # 重新设置随机方式
        dist.barrier()  # 确保所有进程都完成了上一个epoch的工作,并准备好开始新的epoch。
        group_join = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=300)
        )  # args.dist_backend

        train_and_evaluate(
            rank,
            epoch,
            hps,
            model,
            optimizer,
            scaler,
            [train_data_loader, dev_data_loader],
            logger,
            [writer, writer_eval],
            group_join,
        )
        scheduler.step()
        dist.destroy_process_group(group_join)


def sovits_join(group_join, batch_idx):
    # 分布式训练中检测和处理不均匀的工作负载分配。
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    if batch_idx != 0:
        # we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
        try:
            dist.monitored_barrier(
                group=group_join, timeout=group_join.options._timeout
            )
            return False
        except RuntimeError as e:
            logging.info(
                "Detected uneven workload distribution: {}\n".format(e)
                + "Break current worker to manually join all workers, "
                + "world_size {}, current rank {}, current local_rank {}\n".format(
                    world_size, rank, local_rank
                )
            )
            return True
    else:
        return False


def train_and_evaluate(
    rank,
    epoch,
    hps,
    model,
    optimizer,
    scaler,
    loaders,
    logger,
    writers,
    group_join,
):
    global global_step
    model.train()
    train_loader, eval_loader = loaders
    writer, writer_eval = writers

    with model.join():
        for batch_idx, items in enumerate(train_loader):
            if hps.train.train_type != "finetune":
                if (
                    batch_idx > 0 and batch_idx % hps.train.per_step_epoch == 0
                ):  # yi epoch结束
                    break

            if sovits_join(group_join, batch_idx):
                break
            optimizer.zero_grad()
            mel_spec = items["mel_spec"].cuda(rank)
            f0 = items["f0"].cuda(rank)  #  f0.unsqueeze(2)
            volume = items["volume"].cuda(rank)  #  f0.unsqueeze(2)
            ssl_feature = items["ssl_feature"].cuda(
                rank
            )  # ssl_feature.permute(0,2,1)
            spk_id = items["spk_id"].cuda(rank)  # unsqueeze(1)
            keyshift = items["keyshift"].cuda(rank)  # unsqueeze(1)
            loss = model(
                ssl_feature.float(),
                f0,
                volume,
                spk_id,
                aug_shift=keyshift,
                gt_spec=mel_spec.float(),
                infer=False,
                k_step=(
                    model.module.k_step_max
                    if hasattr(model, "module")
                    else model.k_step_max
                ),
            )            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                if global_step % hps.train.log_interval == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"epoch: {epoch} | {batch_idx}/{hps.train.per_step_epoch} | loss: {loss.item()}"
                    )
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", current_lr, global_step)

            global_step += 1
    try:
        dist.barrier()
    except RuntimeError as e:
        logger.info("except RuntimeError as e: {}".format(e))

    if rank == 0 and epoch % hps.train.per_epoch_save == 0:

        evaluate(rank, hps, model, eval_loader, writer_eval)
        # 保存模型
        checkpoint_path = hps.model_dir / f"diff_{global_step}.pt"
        logger.info(
            "Saving model and optimizer state at iteration {} to {}".format(
                epoch, checkpoint_path
            )
        )
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(
            {
                "model": state_dict,
                "iteration": epoch,
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_path,
        )


def evaluate(rank, hps, model, eval_loader, writer_eval):
    global global_step
    model.eval()
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            mel_spec = items["mel_spec"].cuda(rank)
            f0 = items["f0"].cuda(rank)
            volume = items["volume"].cuda(rank)
            ssl_feature = items["ssl_feature"].cuda(rank)
            spk_id = items["spk_id"].cuda(rank)
            # keyshift = items["keyshift"].cuda(rank, non_blocking=False)
            gen_mel = model(
                ssl_feature,
                f0,
                volume,
                spk_id,
                gt_spec=mel_spec,
                infer=True,
                infer_speedup=hps.infer.speedup,
                method=hps.infer.method,
                k_step=(
                    model.module.k_step_max
                    if hasattr(model, "module")
                    else model.k_step_max
                ),
            )
            vmin = -14
            vmax = 3.5
            spec_cat = torch.cat(
                [(gen_mel - mel_spec).abs() + vmin, mel_spec, gen_mel], -1
            )
            spec = spec_cat[0]
            if isinstance(spec, torch.Tensor):
                spec = spec.cpu().numpy()
            fig = plt.figure(figsize=(12, 9))
            plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
            plt.tight_layout()
            writer_eval.add_figure(f"spec_{batch_idx}", fig, global_step)
            plt.close(fig)  # 添加这行，及时清理图像
    model.train()


if __name__ == "__main__":
    run()
