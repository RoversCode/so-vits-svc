#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gradio_inference.py
@Time    :   2025/01/01 16:43:08
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
'''

# here put the import lib
import os
import glob
import gradio as gr
import soundfile as sf
from inference.infer_tool import Svc
import logging
import numpy as np
from pathlib import Path
import json
import yaml
from glob import glob
from typing import Dict, List, Tuple

logging.getLogger("numba").setLevel(logging.WARNING)


class ModelManager:
    def __init__(self):
        self.models: Dict[str, Dict] = {}
        self.scan_models()
    
    def scan_models(self):
        """扫描所有可用的模型"""
        for model_dir in glob("ckpts/*/"):
            dir_name = Path(model_dir).name
            model_info = {"dir_name": dir_name}
            
            # 检查sovits模型
            sovits_path = os.path.join(model_dir, "sovits")
            if os.path.exists(sovits_path):
                model_files = glob(os.path.join(sovits_path, "G_*.pth"))
                if model_files:
                    # 按数字排序所有模型
                    model_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
                    model_info["sovits"] = {
                        "models": model_files,  # 保存所有模型路径
                        "config": f"configs/{dir_name}/sovits.yaml"
                    }
            
            # 检查diffusion模型
            diffusion_path = os.path.join(model_dir, "diffusion")
            if os.path.exists(diffusion_path):
                model_files = glob(os.path.join(diffusion_path, "*.pt"))
                if model_files:
                    # 按数字排序所有模型
                    model_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
                    model_info["diffusion"] = {
                        "models": model_files,  # 保存所有模型路径
                        "config": f"configs/{dir_name}/diffusion.yaml"
                    }
            
            # 检查特征检索模型
            index_path = os.path.join(model_dir, "index")
            if os.path.exists(index_path):
                feature_file = os.path.join(index_path, "feature_and_index.pkl")
                if os.path.exists(feature_file):
                    model_info["feature_retrieval"] = {
                        "path": feature_file
                    }
            
            # 检查说话人信息
            spk_path = os.path.join(model_dir, "spk_info.json")
            if os.path.exists(spk_path):
                model_info["speakers"] = self._load_speakers(spk_path)
                
            if "sovits" in model_info or "diffusion" in model_info:
                self.models[dir_name] = model_info

    def _load_speakers(self, spk_path: str) -> List[str]:
        """加载说话人信息"""
        try:
            with open(spk_path, 'r', encoding='utf-8') as f:
                spk_info = json.load(f)
                return list(spk_info.keys())
        except Exception as e:
            print(f"Error loading speaker info from {spk_path}: {e}")
            return []
    
    def get_model_list(self) -> List[str]:
        """获取所有可用模型的列表"""
        return list(self.models.keys())
    
    def get_model_info(self, model_key: str) -> Dict:
        """获取指定模型的信息"""
        return self.models.get(model_key, {})


class SvcInterface:
    def __init__(self):
        self.current_model = None
        self.model_manager = ModelManager()
        
    def get_speakers(self, model_key: str) -> List[str]:
        """获取指定模型支持的说话人列表"""
        model_info = self.model_manager.get_model_info(model_key)
        return model_info.get("spk_info", [])
    
    def load_model(
        self,
        model_name: str,
        sovits_model_name: str = None,
        diffusion_model_name: str = None,
        use_sovits: bool = True,
        use_diffusion: bool = False,
        use_feature_retrieval: bool = False,
        enhance: bool = False,
    ) -> str:
        """加载选定的模型"""
        try:
            model_info = self.model_manager.models.get(model_name)
            if not model_info:
                return "未找到模型！"
            
            # 检查必要的模型是否存在
            if use_sovits:
                if "sovits" not in model_info:
                    return "未找到SoVITS模型！"
                if not sovits_model_name:
                    return "请选择一个SoVITS模型文件！"
                
            if use_diffusion:
                if "diffusion" not in model_info:
                    return "未找到扩散模型！"
                if not diffusion_model_name:
                    return "请选择一个扩散模型文件！"
            
            # 准备模型路径
            sovits_base = os.path.join("ckpts", model_name, "sovits")
            diffusion_base = os.path.join("ckpts", model_name, "diffusion")
            
            sovits_path = os.path.join(sovits_base, sovits_model_name) if use_sovits else ""
            sovits_config = f"configs/{model_name}/sovits.yaml" if use_sovits else ""
            diffusion_path = os.path.join(diffusion_base, diffusion_model_name) if use_diffusion else ""
            diffusion_config = f"configs/{model_name}/diffusion.yaml" if use_diffusion else ""
            
            # 特征检索模型路径
            cluster_path = ""
            if use_feature_retrieval:
                if "feature_retrieval" in model_info:
                    cluster_path = model_info["feature_retrieval"]["path"]
                else:
                    return "未找到特征检索模型！"
            
            self.current_model = Svc(
                sovits_path,
                sovits_config,
                cluster_path,
                enhance,
                diffusion_path,
                diffusion_config,
                use_diffusion,
                not use_sovits,
                False,
                use_feature_retrieval,
            )
            return f"成功加载模型：{model_name}"
        except Exception as e:
            return f"加载模型时出错：{str(e)}"

    def inference(
        self,
        audio_path,
        speaker,
        pitch_adjust,
        auto_predict_f0=False,
        cluster_ratio=0,
        noise_scale=0.4,
        pad_seconds=0.5,
        clip_seconds=0,
        lg_num=0,
        lgr_num=0.75,
        f0_predictor="rmvpe",
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=100,
        second_encoding=False,
        loudness_envelope_adjustment=0.75,
    ):
        """执行推理"""
        try:
            if self.current_model is None:
                return "Please load a model first!", None

            # 处理音频路径
            if isinstance(audio_path, tuple):  # gradio audio 组件返回 (sr, data) 元组
                sr, data = audio_path
                temp_path = "temp_audio_input.wav"
                sf.write(temp_path, data, sr)
                audio_path = temp_path
            elif hasattr(audio_path, 'name'):  # 处理上传的文件
                audio_path = audio_path.name

            # 执行推理
            audio = self.current_model.slice_inference(
                raw_audio_path=audio_path,
                spk=speaker,
                tran=pitch_adjust,
                slice_db=-40,
                cluster_infer_ratio=cluster_ratio,
                auto_predict_f0=auto_predict_f0,
                noice_scale=noise_scale,
                pad_seconds=pad_seconds,
                clip_seconds=clip_seconds,
                lg_num=lg_num,
                lgr_num=lgr_num,
                f0_predictor=f0_predictor,
                enhancer_adaptive_key=enhancer_adaptive_key,
                cr_threshold=cr_threshold,
                k_step=k_step,
                use_spk_mix=False,
                second_encoding=second_encoding,
                loudness_envelope_adjustment=loudness_envelope_adjustment,
            )

            # 保存结果
            output_path = f"results/output_{np.random.randint(10000)}.wav"
            sf.write(output_path, audio, self.current_model.target_sample)

            # 清理临时文件
            if audio_path == "temp_audio_input.wav":
                os.remove(audio_path)

            return "Inference completed!", output_path
        except Exception as e:
            return f"Error during inference: {str(e)}", None


def create_interface():
    svc = SvcInterface()

    with gr.Blocks() as demo:
        gr.Markdown("# So-VITS-SVC 4.1 推理界面 （支持SoVITS和Diffusion）By LJJ")
        
        with gr.Tab("模型选择"):
            with gr.Row():
                with gr.Column():
                    # 项目选择
                    model_dropdown = gr.Dropdown(
                        choices=svc.model_manager.get_model_list(),
                        label="选择项目",
                        value=svc.model_manager.get_model_list()[0] if svc.model_manager.get_model_list() else None
                    )
                    
                    # SoVITS模型选择
                    sovits_model = gr.Dropdown(
                        label="SoVITS模型",
                        choices=[],
                        value=None,
                        interactive=False
                    )
                    
                    # Diffusion模型选择
                    diffusion_model = gr.Dropdown(
                        label="扩散模型",
                        choices=[],
                        value=None,
                        interactive=False
                    )
                    
                    # 功能选择
                    use_sovits = gr.Checkbox(label="使用SoVITS", value=True)
                    use_diffusion = gr.Checkbox(label="使用扩散模型", value=False)
                    use_feature_retrieval = gr.Checkbox(label="使用特征检索", value=False)
                    enhance = gr.Checkbox(label="使用NSF_HIFIGAN增强", value=False)
                    
                    load_btn = gr.Button("加载选中的模型")
                    load_status = gr.Textbox(label="状态", interactive=False)
                
                with gr.Column():
                    model_info = gr.JSON(label="模型信息")

        with gr.Tab("推理设置"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(label="输入音频", type="filepath")
                    speaker = gr.Dropdown(
                        label="说话人",
                        choices=[],
                        value=None
                    )
                    pitch_adjust = gr.Slider(
                        label="音高调整（半音）",
                        minimum=-12,
                        maximum=12,
                        value=0,
                        step=1,
                    )
                
                with gr.Column():
                    auto_predict_f0 = gr.Checkbox(
                        label="自动预测音高（不建议用于歌声）",
                        value=False,
                    )
                    cluster_ratio = gr.Slider(
                        label="特征检索/聚类混合比例", 
                        minimum=0, 
                        maximum=1, 
                        value=0
                    )
                    noise_scale = gr.Slider(
                        label="噪声比例", 
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.4
                    )
                    f0_predictor = gr.Dropdown(
                        label="音高预测器",
                        choices=["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"],
                        value="rmvpe",
                    )
                    enhancer_adaptive_key = gr.Slider(
                        label="增强器自适应音高", 
                        minimum=0,
                        maximum=12,
                        value=0,
                        step=1
                    )
                    cr_threshold = gr.Slider(
                        label="F0过滤阈值",
                        minimum=0,
                        maximum=1,
                        value=0.05,
                        step=0.01
                    )

                with gr.Column():
                    pad_seconds = gr.Slider(
                        label="音频填充长度（秒）",
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.1
                    )
                    clip_seconds = gr.Slider(
                        label="强制切片长度（秒）",
                        minimum=0,
                        maximum=10,
                        value=0,
                        step=0.1
                    )
                    lg_num = gr.Slider(
                        label="交叉淡入长度（秒）",
                        minimum=0,
                        maximum=1,
                        value=0,
                        step=0.1
                    )
                    lgr_num = gr.Slider(
                        label="切片保留比例",
                        minimum=0,
                        maximum=1,
                        value=0.75,
                        step=0.01
                    )
                    k_step = gr.Slider(
                        label="扩散步数", 
                        minimum=1, 
                        maximum=1000, 
                        value=100
                    )
                    second_encoding = gr.Checkbox(
                        label="二次编码", 
                        value=False
                    )
                    loudness_adjustment = gr.Slider(
                        label="响度包络混合比例",
                        minimum=0,
                        maximum=1,
                        value=0.75,
                    )

            infer_btn = gr.Button("开始推理")
            with gr.Row():
                status_output = gr.Textbox(label="状态")
                audio_output = gr.Audio(label="输出音频")

        def update_model_info(model_name):
            """更新模型信息显示"""
            model_info = svc.model_manager.models.get(model_name, {})
            speakers = model_info.get("speakers", [])
            
            # 获取所有可用的模型文件
            sovits_models = [Path(p).name for p in model_info.get("sovits", {}).get("models", [])]
            diffusion_models = [Path(p).name for p in model_info.get("diffusion", {}).get("models", [])]
            
            has_sovits = bool(sovits_models)
            has_diffusion = bool(diffusion_models)
            has_feature = "feature_retrieval" in model_info
            
            return (
                model_info,
                gr.Dropdown(choices=speakers, value=speakers[0] if speakers else None),
                gr.Dropdown(choices=sovits_models, value=sovits_models[-1] if sovits_models else None, interactive=has_sovits), # 3. 更新 SoVITS 模型下拉框 
                gr.Dropdown(choices=diffusion_models, value=diffusion_models[-1] if diffusion_models else None, interactive=has_diffusion), # 4. 更新 Diffusion 模型下拉框
                gr.Checkbox(value=has_sovits, interactive=has_sovits),
                gr.Checkbox(value=False, interactive=has_diffusion),
                gr.Checkbox(value=False, interactive=has_feature)
            )

        model_dropdown.change(
            fn=update_model_info,
            inputs=[model_dropdown],
            outputs=[
                model_info,
                speaker,
                sovits_model,
                diffusion_model,
                use_sovits,
                use_diffusion,
                use_feature_retrieval
            ]
        )
        
        load_btn.click(
            fn=svc.load_model,
            inputs=[
                model_dropdown,
                sovits_model,
                diffusion_model,
                use_sovits,
                use_diffusion,
                use_feature_retrieval,
                enhance
            ],
            outputs=[load_status]
        )

        infer_btn.click(
            fn=lambda *args: svc.inference(*args),
            inputs=[
                audio_input,
                speaker,
                pitch_adjust,
                auto_predict_f0,
                cluster_ratio,
                noise_scale,
                pad_seconds,
                clip_seconds,
                lg_num,
                lgr_num,
                f0_predictor,
                enhancer_adaptive_key,
                cr_threshold,
                k_step,
                second_encoding,
                loudness_adjustment,
            ],
            outputs=[status_output, audio_output],
        )

    return demo


if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs("results", exist_ok=True)

    # 启动界面
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1")
