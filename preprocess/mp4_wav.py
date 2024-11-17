import os
import subprocess


def extract_audio(directory, suffix="mp4"):
    # 遍历指定目录及其子目录下的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 如果文件是一个 .mov 文件
            if file.endswith("." + suffix):
                # 获取文件的完整路径
                full_path = os.path.join(root, file)
                # 获取文件名（不包括扩展名）
                filename = os.path.splitext(file)[0]
                # 构造输出文件的路径
                output_path = os.path.join(root, f"{filename}.wav")
                # 使用 ffmpeg 提取音频
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        full_path,
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ac",
                        "1",
                        "-ar",
                        "44100",
                        output_path,
                    ]
                )
                
                
extract_audio("/mnt/e/Workspace/growth/audio/so-vits-svc/preprocess/raw_data/蜡笔小新")