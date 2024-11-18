import subprocess
import os

def download_bilibili_video(url, output_dir='.', filename=None):
    """
    Download a video from Bilibili.
    
    :param url: The URL of the Bilibili video.
    :param output_dir: The directory to save the video (default is current directory).
    :param filename: The filename for the downloaded video (optional).
    :return: True if download was successful, False otherwise.
    """
    try:
        # Ensure you-get is installed
        subprocess.run(["you-get", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("you-get is not installed. Please install it using 'pip install you-get'")
        return False
    except FileNotFoundError:
        print("you-get is not found. Please install it using 'pip install you-get'")
        return False

    # Prepare the command
    command = ["you-get", "-o", output_dir, "-l"]
    if filename:
        command.extend(["-O", filename])
    command.append(url)

    try:
        # Run the download command
        subprocess.run(command, check=True)
        print(f"Video downloaded successfully to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while downloading the video: {e}")
        return False


# Example usage
if __name__ == "__main__":
    video_url = "https://www.bilibili.com/video/BV1wf421Q7oe?p=2&vd_source=267a10f7991de338c115e6cce1439723"
    download_bilibili_video(video_url, output_dir="preprocess/downloads", filename="2")