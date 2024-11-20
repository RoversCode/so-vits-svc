#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2024/11/13 14:01:34
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import time
import torch
import datetime
import torch.distributed as dist
import modules.commons as commons
import utils
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from tools.train_utils import init_dataloader
import json



logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


def run():
    global global_step
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()
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
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir/'tensorboard')
        writer_eval = SummaryWriter(log_dir=hps.model_dir/'tensorboard/eval')

    # c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded, volume_padded
    train_dataset, train_data_loader, dev_dataset, dev_data_loader = init_dataloader(hps, rank)

    try:
        spk2id = json.load(open("Data/sovits_svc/speaker_map.json", "r"))   # 训练底模
        hps.model.n_speakers = len(spk2id)
        del spk2id
    except Exception:
        pass
    
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    net_g = DDP(net_g, device_ids=[rank])  # , find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank])

    try:
        _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
            net_g,
            optim_g,
            hps.train.skip_optimizer,
        )
        _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
            net_d,
            optim_d,
            hps.train.skip_optimizer,
        )
        epoch_str = max(epoch_str, 1)
        name = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        global_step = int(name[name.rfind("_") + 1 : name.rfind(".")]) + 1
        # global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0

    if hps.train.skip_optimizer:
        epoch_str = 1
        global_step = 0

    warmup_epoch = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):  # epoch_str >= 1
        train_dataset.set_epoch(epoch)  # 重新设置随机方式s
        dist.barrier()  # 确保所有进程都完成了上一个epoch的工作,并准备好开始新的epoch。
        group_join = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=300)
        )  # args.dist_backend
        # set up warm-up learning rate
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group["lr"] = hps.train.learning_rate / warmup_epoch * epoch
            for param_group in optim_d.param_groups:
                param_group["lr"] = hps.train.learning_rate / warmup_epoch * epoch
                

        # training
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                scaler,
                [train_data_loader, dev_data_loader],
                logger,
                [writer, writer_eval],
                group_join
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                scaler,
                [train_data_loader, None],
                None,
                None,
                group_join
            )
        # update learning rate
        scheduler_g.step()
        scheduler_d.step()
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
    rank, epoch, hps, nets, optims, scaler, loaders, logger, writers, group_join
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    half_type = torch.bfloat16 if hps.train.half_type == "bf16" else torch.float16

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    with net_g.join():  # 这里的操作会确保所有进程都执行完成后才继续
        for batch_idx, items in enumerate(train_loader):
            if hps.train.train_type != 'finetune':
                if batch_idx > 0 and batch_idx % hps.train.per_step_epoch == 0:  # yi epoch结束
                    break
            
            if sovits_join(group_join, batch_idx):
                break
            
            spec = items["spec"]
            c = items["ssl_feature"]
            lengths = items["ssl_length"]
            f0 = items["f0"]
            uv = items["uv"]
            volume = items["volume"]
            y = items["audio"]
            spk = items["spk_id"]
            # c, f0, spec, y, spk, lengths, uv, volume = items
            g = spk.cuda(rank, non_blocking=True)
            spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
            c = c.cuda(rank, non_blocking=True)
            f0 = f0.cuda(rank, non_blocking=True)
            uv = uv.cuda(rank, non_blocking=True)
            lengths = lengths.cuda(rank, non_blocking=True)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            with autocast(enabled=hps.train.fp16_run, dtype=half_type):
                (
                    y_hat,
                    ids_slice,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    pred_lf0,
                    norm_lf0,
                    lf0,
                ) = net_g(
                    c,
                    f0,
                    uv,
                    spec,
                    g=g,
                    c_lengths=lengths,
                    spec_lengths=lengths,
                    vol=volume,
                )

                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y = commons.slice_segments(
                    y, ids_slice * hps.data.hop_length, hps.train.segment_size
                )  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

                with autocast(enabled=False, dtype=half_type):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc

            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=hps.train.fp16_run, dtype=half_type):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                with autocast(enabled=False, dtype=half_type):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_lf0 = (
                        F.mse_loss(pred_lf0, lf0)
                        if net_g.module.use_automatic_f0_prediction
                        else 0
                    )
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            if rank == 0:
                if global_step % hps.train.log_interval == 0:
                    lr = optim_g.param_groups[0]["lr"]
                    losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                    reference_loss = 0
                    for i in losses:
                        reference_loss += i
                    logger.info(
                        f"Train Epoch: {epoch} step: {batch_idx}/{hps.train.per_step_epoch} "
                    )
                    logger.info(
                        f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}"
                    )

                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc_all,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                            "loss/g/kl": loss_kl,
                            "loss/g/lf0": loss_lf0,
                        }
                    )

                    # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                    # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                    # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                    image_dict = {
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy()
                        ),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].data.cpu().numpy()
                        ),
                        "all/mel": utils.plot_spectrogram_to_numpy(
                            mel[0].data.cpu().numpy()
                        ),
                    }

                    if net_g.module.use_automatic_f0_prediction:
                        image_dict.update(
                            {
                                "all/lf0": utils.plot_data_to_numpy(
                                    lf0[0, 0, :].cpu().numpy(),
                                    pred_lf0[0, 0, :].detach().cpu().numpy(),
                                ),
                                "all/norm_lf0": utils.plot_data_to_numpy(
                                    lf0[0, 0, :].cpu().numpy(),
                                    norm_lf0[0, 0, :].detach().cpu().numpy(),
                                ),
                            }
                        )

                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        images=image_dict,
                        scalars=scalar_dict,
                    )
                
            global_step += 1

    try:
        dist.barrier()
    except RuntimeError as e:
        logger.info("except RuntimeError as e: {}".format(e))

    if rank == 0:
        global start_time
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
        )
        utils.save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
        )
        keep_ckpts = getattr(hps.train, "keep_ckpts", 0)
        if keep_ckpts > 0:
            utils.clean_checkpoints(
                path_to_models=hps.model_dir,
                n_ckpts_to_keep=keep_ckpts,
                sort_by_time=True,
            )

        now = time.time()
        durtaion = format(now - start_time, ".2f")
        logger.info(f"====> Epoch: {epoch}, cost {durtaion} s")
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            if hps.train.train_type != 'finetune':
                if batch_idx > 0 and batch_idx % 10 == 0:  #  评估10条
                    break
            spec = items["spec"]
            c = items["ssl_feature"]
            lengths = items["ssl_length"]
            f0 = items["f0"]
            uv = items["uv"]
            volume = items["volume"]
            y = items["audio"]
            spk = items["spk_id"]
            g = spk[:1].cuda(0)
            spec, y = spec[:1].cuda(0), y[:1].cuda(0)
            c = c[:1].cuda(0)
            f0 = f0[:1].cuda(0)
            uv = uv[:1].cuda(0)
            if volume is not None:
                volume = volume[:1].cuda(0)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_hat, _ = generator.module.infer(c, f0, uv, g=g, vol=volume)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            audio_dict.update(
                {f"gen/audio_{batch_idx}": y_hat[0], f"gt/audio_{batch_idx}": y[0]}
            )
        image_dict.update(
            {
                "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
                "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
            }
        )
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    run()
