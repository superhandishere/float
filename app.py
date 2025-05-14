# app.py

import argparse
import os
import uuid
import torch
import tempfile
import datetime
import subprocess
import gradio as gr
import torchvision
from generate import InferenceAgent, InferenceOptions

# ---- 手动 patch：覆盖 generate.py 中的 save_video 方法 ----
def patched_save_video(self, vid_target_recon: torch.Tensor, video_path: str, audio_path: str) -> str:
    # 1) 在临时目录生成唯一文件名（不打开文件句柄）
    tmp_mp4 = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.mp4")

    # 2) 写裸视频流
    vid = vid_target_recon.permute(0, 2, 3, 1).detach().clamp(-1, 1).cpu()
    vid = ((vid + 1) / 2 * 255).type(torch.uint8)
    torchvision.io.write_video(tmp_mp4, vid, fps=self.opt.fps)

    # 3) 如果有音频，用 ffmpeg 合成，再删掉临时文件
    if audio_path:
        cmd = f'ffmpeg -i "{tmp_mp4}" -i "{audio_path}" -c:v copy -c:a aac -y "{video_path}"'
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(video_path):
            os.remove(tmp_mp4)
    else:
        os.rename(tmp_mp4, video_path)

    return video_path

# Patch 掉原方法
InferenceAgent.save_video = patched_save_video
# --------------------------------------------------------

# 初始化 Agent
def get_agent():
    parser = argparse.ArgumentParser()
    opts = InferenceOptions()
    parser = opts.initialize(parser)
    args = parser.parse_args([])          # 全部用默认值
    args.rank      = 0
    args.ngpus     = 1
    args.ckpt_path = "./checkpoints/float.pth"
    args.res_dir   = "./results"
    os.makedirs(args.res_dir, exist_ok=True)
    return InferenceAgent(args), args

agent, opt = get_agent()

# Gradio 回调
def generate_video(ref_path, aud_path, a_cfg, e_cfg, seed, no_crop, emo):
    try:
        # 生成输出文件名
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out_name = f"{now}-float-out.mp4"
        target_path = os.path.join(opt.res_dir, out_name)

        # 调用推理
        result = agent.run_inference(
            res_video_path=target_path,
            ref_path=ref_path,
            audio_path=aud_path,
            a_cfg_scale=a_cfg,
            e_cfg_scale=e_cfg,
            seed=seed,
            no_crop=no_crop,
            emo=emo or "neutral",
            verbose=True
        )
        # 确保文件已经被释放
        if not os.path.exists(result):
            raise gr.Error("结果文件未生成")
        return result

    except Exception as e:
        # 在界面上显示错误
        raise gr.Error(f"推理失败：{e}")

# Gradio 界面
demo = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Image(label="参考图像", type="filepath"),
        gr.Audio(label="音频文件", type="filepath"),
        gr.Slider(0.1, 5.0, value=2.0, label="a_cfg_scale"),
        gr.Slider(0.1, 5.0, value=1.0, label="e_cfg_scale"),
        gr.Slider(0, 100, step=1, value=15, label="Seed"),
        gr.Checkbox(label="跳过裁剪（no_crop）"),
        gr.Dropdown(
            choices=["neutral","happy","sad","angry","disgust","fear","surprise"],
            value="neutral", label="情绪（emo）"
        ),
    ],
    outputs=gr.Video(label="输出视频"),
    title="FLOAT 推理演示",
    description="上传参考图像和音频，配置参数后生成口型同步视频"
)

if __name__ == "__main__":
    # 如需远程访问可加 share=True
    demo.launch()
