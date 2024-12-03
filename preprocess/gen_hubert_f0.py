import os
import sys
# 获取当前文件的绝对路径
cur_dir = os.path.dirname(os.path.abspath(__file__))
# 拿到父目录
parent_dir = os.path.dirname(cur_dir)
sys.path.append(parent_dir)
import argparse
import logging
import random
import librosa
import yaml
import numpy as np
import torch
import torch.multiprocessing as mp
import diffusion.utils.utils as du
import utils
import time
from diffusion.vocoder import Vocoder
from modules.mel_processing import spectrogram_torch
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def process_one(
    file_path, hmodel, f0_predictor, device
):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    wav, sr = librosa.load(file_path, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)
    soft_path = file_path.parent / (file_path.stem + "_ssl.pt")
    f0_path = file_path.parent / (file_path.stem + "_f0.npy")
    # spec_path = file_path.parent / (file_path.stem + "_spec.pt")
    strat_time = time.time()
    if not os.path.exists(soft_path) or not os.path.exists(f0_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)

    if not os.path.exists(soft_path):
        c = hmodel.encoder(wav16k)  # ssl
        torch.save(c.cpu(), soft_path)
    ssl_time = time.time() - strat_time
    # logger.info(f"Process {filename} 编码器耗时 {ssl_time:.2f}s")

    if not os.path.exists(f0_path):  # 预测f0
        f0, uv = f0_predictor.compute_f0_uv(wav16k.squeeze(0), 16000)
        np.save(f0_path, np.asanyarray((f0, uv), dtype=object))
    f0_time = time.time() - strat_time - ssl_time
    logger.info(f"Process {file_path.name} F0耗时 {f0_time:.2f}s")

    # if not spec_path.exists():
    #     # Process spectrogram
    #     # The following code can't be replaced by torch.FloatTensor(wav)
    #     # because load_wav_to_torch return a tensor that need to be normalized
    #     if sr != hps.data.sampling_rate:
    #         raise ValueError(
    #             "{} SR doesn't match target {} SR".format(sr, hps.data.sampling_rate)
    #         )
    #     # audio_norm = audio / hps.data.max_wav_value
    #     spec = spectrogram_torch(
    #         audio_norm,
    #         hps.data.filter_length,
    #         hps.data.sampling_rate,
    #         hps.data.hop_length,
    #         hps.data.win_length,
    #         center=False,
    #     )
    #     spec = torch.squeeze(spec, 0)
    #     torch.save(spec, spec_path)
    # spec_time = time.time() - strat_time - ssl_time - f0_time
    # # logger.info(f"Process {filename} 频谱耗时 {spec_time:.2f}s")

    # if hps.model.vol_embedding:
    #     volume_path = file_path.parent / (file_path.stem + "_vol.npy")
    #     volume_extractor = utils.Volume_Extractor(hop_length)
    #     if not volume_path.exists():
    #         volume = volume_extractor.extract(audio_norm)
    #         np.save(volume_path, volume.to("cpu").numpy())
    # vol_time = time.time() - strat_time - ssl_time - f0_time - spec_time
    # logger.info(f"Process {filename} 音量耗时 {vol_time:.2f}s")
    # if diff:
    #     mel_path = file_path.parent / (file_path.stem + "_mel.npy")
    #     if not mel_path.exists() and mel_extractor is not None:
    #         mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
    #         mel = mel_t.squeeze().to("cpu").numpy()
    #         np.save(mel_path, mel)
    #     aug_mel_path = file_path.parent / (file_path.stem + "_aug_mel.npy")
    #     aug_vol_path = file_path.parent / (file_path.stem + "_aug_vol.npy")
    #     max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
    #     max_shift = min(1, np.log10(1 / max_amp))
    #     log10_vol_shift = random.uniform(-1, max_shift)
    #     keyshift = random.uniform(-5, 5)
    #     if mel_extractor is not None:
    #         aug_mel_t = mel_extractor.extract(
    #             audio_norm * (10**log10_vol_shift), sampling_rate, keyshift=keyshift
    #         )
    #     aug_mel = aug_mel_t.squeeze().to("cpu").numpy()
    #     aug_vol = volume_extractor.extract(audio_norm * (10**log10_vol_shift))
    #     if not aug_mel_path.exists():
    #         np.save(aug_mel_path, np.asanyarray((aug_mel, keyshift), dtype=object))
    #     if not aug_vol_path.exists():
    #         np.save(aug_vol_path, aug_vol.to("cpu").numpy())
    # voco_time = time.time() - strat_time - ssl_time - f0_time - spec_time - vol_time
    # logger.info(f"Process {filename} voco耗时 {voco_time:.2f}s")


def process_batch(file_chunk, f0p, device="cpu"):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Rank {rank} uses device {device}")
    hmodel = utils.get_speech_encoder(speech_encoder, device=device)  # ssl model
    f0_predictor = utils.get_f0_predictor(
        f0p,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        device=device,
        threshold=0.05,
    )
    logger.info(f"Loaded speech encoder for rank {rank}")
    for filename in tqdm(file_chunk, position=rank):
        process_one(filename, hmodel, f0_predictor, device)


def parallel_process(data_dir, num_processes, f0p, device):

    subdirs = [x.name for x in data_dir.iterdir() if x.is_dir()]  # spk
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for spk_name in subdirs:
            filenames = list((data_dir / spk_name).glob("*.wav"))
            tasks = []
            for i in range(num_processes):
                start = int(i * len(filenames) / num_processes)
                end = int((i + 1) * len(filenames) / num_processes)
                file_chunk = filenames[start:end]
                tasks.append(
                    executor.submit(
                        process_batch,
                        file_chunk,
                        f0p,
                        device=device,
                    )
                )
            for task in tqdm(tasks, position=0):
                task.result()


hps = utils.get_hparams_from_file("configs/sovits_base_config.yaml")
dconfig = du.load_config("configs/diffusion_base_config.yaml")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps.model.speech_encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_dir", type=str, help="path to input dir")
    # parser.add_argument(
    #     "--use_diff", action="store_true", help="Whether to use the diffusion model"
    # )
    parser.add_argument(
        "--f0_predictor",
        type=str,
        default="rmvpe",
        help="Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe,fcpe|default: pm(note: crepe is original F0 using mean filter)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="You are advised to set the number of processes to the same as the number of CPU cores",
    )
    args = parser.parse_args()

    f0p = args.f0_predictor
    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Using device: " + str(device))
    logger.info("Using SpeechEncoder: " + speech_encoder)
    logger.info("Using extractor: " + f0p)
    # logger.info("Using diff Mode: " + str(args.use_diff))

    # if args.use_diff:  # diffusion
    #     print("use_diff")
    #     print("Loading Mel Extractor...")
    #     mel_extractor = Vocoder(
    #         dconfig.vocoder.type, dconfig.vocoder.ckpt, device=device
    #     )
    #     print("Loaded Mel Extractor.")
    # else:
    #     mel_extractor = None

    mp.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()

    data_dir = Path(args.data_dir)
    parallel_process(data_dir, num_processes, f0p, device)
