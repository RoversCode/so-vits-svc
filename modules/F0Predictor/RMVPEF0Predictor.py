from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from modules.F0Predictor.F0Predictor import F0Predictor

from .rmvpe import RMVPE


class RMVPEF0Predictor(F0Predictor):

    def __init__(self,
                 hop_length=512,
                 f0_min=50,
                 f0_max=1100,
                 dtype=torch.float32,
                 device=None,
                 sampling_rate=44100,
                 threshold=0.05):
        self.rmvpe = RMVPE(model_path="pretrain/rmvpe.pt",
                           dtype=dtype,
                           device=device)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.name = "rmvpe"

    def repeat_expand(self,
                      content: Union[torch.Tensor, np.ndarray],
                      target_len: int,
                      mode: str = "nearest"):
        ndim = content.ndim

        if content.ndim == 1:
            content = content[None, None]
        elif content.ndim == 2:
            content = content[None]

        assert content.ndim == 3

        is_np = isinstance(content, np.ndarray)
        if is_np:
            content = torch.from_numpy(content)

        results = torch.nn.functional.interpolate(content,
                                                  size=int(target_len),
                                                  mode=mode)

        if is_np:
            results = results.numpy()

        if ndim == 1:
            return results[0, 0]
        elif ndim == 2:
            return results[0]

    def post_process(self, x, sampling_rate, f0, pad_to): # self.post_process(x, self.sampling_rate, f0, p_len)
        # if isinstance(f0, np.ndarray):
        #     f0 = torch.from_numpy(f0).float().to(x.device)

        # if pad_to is None:
        #     return f0
        # 标记未发声片段(f0为0的部分)
        uv = f0 == 0
        
        # 对未发声片段进行插值处理
        if len(f0[~uv]) > 0:  # 如果存在有效的F0值
            # 使用线性插值填充未发声片段的F0值
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        
        # 创建原始时间轴(每10ms一个点)
        origin_time = 0.01 * np.arange(len(f0))
        # 创建目标时间轴(基于hop_size和采样率)
        target_time = self.hop_length / self.sampling_rate * np.arange(pad_to)
        
        # 将F0序列重采样到目标时间轴上
        f0 = np.interp(target_time, origin_time, f0)
        
        # 对未发声标记也进行重采样,并转换回布尔值
        # 如果插值后的未发声概率>0.5则认为是未发声片段
        uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
        
        # 将未发声片段的F0设为0
        f0[uv] = 0
        
        # 在序列开头填充start_frame个0
        # f0 = np.pad(f0, (start_frame, 0))
    
        # f0 = self.repeat_expand(f0, pad_to)
        # vuv_vector = torch.zeros_like(f0)
        # vuv_vector[f0 > 0.0] = 1.0
        # vuv_vector[f0 <= 0.0] = 0.0

        # # 去掉0频率, 并线性插值
        # nzindex = torch.nonzero(f0).squeeze()  # 找出所有非零F0值的索引位置
        # f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()  # 只保留非零F0值
        # time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()  # origin time
        # time_frame = np.arange(pad_to) * self.hop_length / sampling_rate  # frame time

        # vuv_vector = F.interpolate(vuv_vector[None, None, :],
        #                            size=pad_to)[0][0]
        # vuv_vector = (vuv_vector > 0.5).float()

        # if f0.shape[0] <= 0:
        #     return torch.zeros(
        #         pad_to, dtype=torch.float,
        #         device=x.device).cpu().numpy(), vuv_vector.cpu().numpy()
        # if f0.shape[0] == 1:
        #     return (torch.ones(pad_to, dtype=torch.float, device=x.device) *
        #             f0[0]).cpu().numpy(), vuv_vector.cpu().numpy()

        # f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        # # vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))
        return f0, uv

    def compute_f0_uv(self, wav, sr, p_len=None):
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        if wav.device != self.device:
            wav = wav.to(self.device)
        x = wav
        if p_len is None:
            dur = len(wav) / sr
            p_len = int((dur * self.sampling_rate) // self.hop_length)
        else:
            assert abs(p_len -
                       x.shape[0] // self.hop_length) < 4, "pad length error"
        f0 = self.rmvpe.infer_from_audio(x, sr, self.threshold)
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        f0 = f0.cpu().numpy()
        return self.post_process(x, self.sampling_rate, f0, p_len)
