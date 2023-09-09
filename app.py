import gradio as gr
import random
import torch
import time
import cv2
import os
import numpy as np
import pytorch_lightning as pl
import moviepy.editor as mp
from pathlib import Path
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.config import parse_args
from scipy.spatial.transform import Rotation as RRR
import mGPT.render.matplot.plot_3d_global as plot_3d
from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from mGPT.render.pyrender.smpl_render import SMPLRender
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Load model
cfg = parse_args(phase="webui")  # parse config file
cfg.FOLDER = 'cache'
output_dir = Path(cfg.FOLDER)
output_dir.mkdir(parents=True, exist_ok=True)
pl.seed_everything(cfg.SEED_VALUE)
if cfg.ACCELERATOR == "gpu":
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
datamodule = build_data(cfg, phase="test")
model = build_model(cfg, datamodule)
state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict)
model.to(device)

audio_processor = WhisperProcessor.from_pretrained(cfg.model.whisper_path)
audio_model = WhisperForConditionalGeneration.from_pretrained(cfg.model.whisper_path).to(device)
forced_decoder_ids = audio_processor.get_decoder_prompt_ids(language="zh", task="translate")
forced_decoder_ids_zh = audio_processor.get_decoder_prompt_ids(language="zh", task="translate")
forced_decoder_ids_en = audio_processor.get_decoder_prompt_ids(language="en", task="translate")

# HTML Style
Video_Components = """
<div class="side-video" style="position: relative;">
    <video width="340" autoplay loop>
        <source src="file/{video_path}" type="video/mp4">
    </video>
    <a class="videodl-button" href="file/{video_path}" download="{video_fname}" title="Download Video">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-video"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg>
    </a>
    <a class="npydl-button" href="file/{motion_path}" download="{motion_fname}" title="Download Motion">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-person-standing"><circle cx="12" cy="5" r="1"/><path d="m9 20 3-6 3 6"/><path d="m6 8 6 2 6-2"/><path d="M12 10v4"/></svg>
    </a>
</div>
"""

Text_Components = """
<h3 class="side-content" >{msg}</h3>
"""


def motion_token_to_string(motion_token, lengths, codebook_size=512):
    motion_string = []
    for i in range(motion_token.shape[0]):
        motion_i = motion_token[i].cpu(
        ) if motion_token.device.type == 'cuda' else motion_token[i]
        motion_list = motion_i.tolist()[:lengths[i]]
        motion_string.append(
            (f'<motion_id_{codebook_size}>' +
             ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
             f'<motion_id_{codebook_size + 1}>'))
    return motion_string


def render_motion(data, feats, method='fast'):
    fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'
    feats_fname = fname + '.npy'
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    np.save(output_npy_path, feats)

    if method == 'slow':
        if len(data.shape) == 4:
            data = data[0]
        data = data - data[0, 0]
        pose_generator = HybrIKJointsToRotmat()
        pose = pose_generator(data)
        pose = np.concatenate([
            pose,
            np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
        ], 1)
        shape = [768, 768]
        render = SMPLRender(cfg.RENDER.SMPL_MODEL_PATH)

        if not os.environ.get("PYOPENGL_PLATFORM"):
            os.environ["DISPLAY"] = ":0.0"
            os.environ["PYOPENGL_PLATFORM"] = "egl"

        size = (shape[1], shape[0])
        fps = 20.0
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        videoWriter = cv2.VideoWriter(output_mp4_path, fourcc, fps, size)
        r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
        pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
        for i in range(data.shape[0]):
            img = np.zeros([shape[0], shape[1], 3])
            aroot = data[[i], 0] + np.array([[0.0, 0.0, 30.0]])
            aroot[:, 1] = -aroot[:, 1]
            params = dict(pred_shape=np.zeros([1, 10]),
                          pred_root=aroot,
                          pred_pose=pose[[i]])
            renderImg = render.render(img.copy(), params)
            renderImg = (renderImg * 255).astype(np.uint8)
            videoWriter.write(renderImg)
        videoWriter.release()
        output_video_h264_name = output_mp4_path[:-4] + '_h264.mp4'
        command = 'ffmpeg -y -i {} -vcodec h264 {}'.format(
            output_mp4_path, output_video_h264_name)
        os.system(command)
        output_mp4_path = output_video_h264_name
        video_fname = video_fname[:-4] + '_h264.mp4'
    elif method == 'fast':
        output_gif_path = output_mp4_path[:-4] + '.gif'
        if len(data.shape) == 3:
            data = data[None]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        pose_vis = plot_3d.draw_to_batch(data, [''], [output_gif_path])
        out_video = mp.VideoFileClip(output_gif_path)
        out_video.write_videofile(output_mp4_path)

    return output_mp4_path, video_fname, output_npy_path, feats_fname


def load_motion(motion_uploaded, method):
    file = motion_uploaded['file']

    feats = torch.tensor(np.load(file), device=model.device)
    if len(feats.shape) == 2:
        feats = feats[None]
    # feats = model.datamodule.normalize(feats)

    # Motion tokens
    motion_lengths = feats.shape[0]
    motion_token, _ = model.vae.encode(feats)

    motion_token_string = model.lm.motion_token_to_string(
        motion_token, [motion_token.shape[1]])[0]
    motion_token_length = motion_token.shape[1]

    # Motion rendered
    joints = model.datamodule.feats2joints(feats.cpu()).cpu().numpy()
    output_mp4_path, video_fname, output_npy_path, joints_fname = render_motion(
        joints,
        feats.to('cpu').numpy(), method)

    motion_uploaded.update({
        "feats": feats,
        "joints": joints,
        "motion_video": output_mp4_path,
        "motion_video_fname": video_fname,
        "motion_joints": output_npy_path,
        "motion_joints_fname": joints_fname,
        "motion_lengths": motion_lengths,
        "motion_token": motion_token,
        "motion_token_string": motion_token_string,
        "motion_token_length": motion_token_length,
    })

    return motion_uploaded


def add_text(history, text, motion_uploaded, data_stored, method):
    data_stored = data_stored + [{'user_input': text}]

    if 'file' in motion_uploaded.keys():
        text = Text_Components.format(msg=text)
        motion_uploaded = load_motion(motion_uploaded, method)
        output_mp4_path = motion_uploaded['motion_video']
        video_fname = motion_uploaded['motion_video_fname']
        output_npy_path = motion_uploaded['motion_joints']
        joints_fname = motion_uploaded['motion_joints_fname']

        text = text + Video_Components.format(video_path=output_mp4_path,
                                              video_fname=video_fname,
                                              motion_path=output_npy_path,
                                              motion_fname=joints_fname)
    else:
        text = f"""<h3>{text}</h3>"""
    history = history + [(text, None)]
    return history, gr.update(value="",
                              interactive=False), motion_uploaded, data_stored


def add_audio(history, audio_path, data_stored):
    audio, sampling_rate = librosa.load(audio_path, sr=16000)
    input_features = audio_processor(
        audio, sampling_rate, return_tensors="pt"
    ).input_features  # whisper training sampling rate, do not modify
    input_features = torch.Tensor(input_features).to(device)
    predicted_ids = audio_model.generate(input_features,
                                         forced_decoder_ids=forced_decoder_ids)
    text_input = audio_processor.batch_decode(predicted_ids,
                                              skip_special_tokens=True)
    text_input = str(text_input).strip('[]"')
    data_stored = data_stored + [{'user_input': text_input}]
    gr.update(value=data_stored, interactive=False)
    history = history + [(text_input, None)]

    return history, data_stored


def add_file(history, file, txt, motion_uploaded):

    motion_uploaded['file'] = file.name
    txt = txt.replace(" <Motion_Placeholder>", "") + " <Motion_Placeholder>"
    return history, gr.update(value=txt, interactive=True), motion_uploaded


def bot(history, motion_uploaded, data_stored, method):

    motion_length, motion_token_string = motion_uploaded[
        "motion_lengths"], motion_uploaded["motion_token_string"]

    input = data_stored[-1]['user_input']
    prompt = model.lm.placeholder_fulfill(input, motion_length,
                                          motion_token_string, "")
    data_stored[-1]['model_input'] = prompt
    batch = {
        "length": [motion_length],
        "text": [prompt],
    }

    outputs = model(batch, task="t2m")
    out_feats = outputs["feats"][0]
    out_lengths = outputs["length"][0]
    out_joints = outputs["joints"][:out_lengths].detach().cpu().numpy()
    out_texts = outputs["texts"][0]
    output_mp4_path, video_fname, output_npy_path, joints_fname = render_motion(
        out_joints,
        out_feats.to('cpu').numpy(), method)

    motion_uploaded = {
        "feats": None,
        "joints": None,
        "motion_video": None,
        "motion_lengths": 0,
        "motion_token": None,
        "motion_token_string": '',
        "motion_token_length": 0,
    }

    data_stored[-1]['model_output'] = {
        "feats": out_feats,
        "joints": out_joints,
        "length": out_lengths,
        "texts": out_texts,
        "motion_video": output_mp4_path,
        "motion_video_fname": video_fname,
        "motion_joints": output_npy_path,
        "motion_joints_fname": joints_fname,
    }

    if '<Motion_Placeholder>' == out_texts:
        response = [
            Video_Components.format(video_path=output_mp4_path,
                                    video_fname=video_fname,
                                    motion_path=output_npy_path,
                                    motion_fname=joints_fname)
        ]
    elif '<Motion_Placeholder>' in out_texts:
        response = [
            Text_Components.format(
                msg=out_texts.split("<Motion_Placeholder>")[0]),
            Video_Components.format(video_path=output_mp4_path,
                                    video_fname=video_fname,
                                    motion_path=output_npy_path,
                                    motion_fname=joints_fname),
            Text_Components.format(
                msg=out_texts.split("<Motion_Placeholder>")[1]),
        ]
    else:
        response = f"""<h3>{out_texts}</h3>"""

    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.02)
        yield history, motion_uploaded, data_stored


with open("assets/css/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS) as demo:

    # Variables
    motion_uploaded = gr.State({
        "feats": None,
        "joints": None,
        "motion_video": None,
        "motion_lengths": 0,
        "motion_token": None,
        "motion_token_string": '',
        "motion_token_length": 0,
    })
    data_stored = gr.State([])

    gr.Markdown(
        "# Welcome to MotionGPT! \n ## You can type or upload a numpy file contains motion joints."
    )

    chatbot = gr.Chatbot([], elem_id="mGPT", height=600, label="MotionGPT")

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or insert motion",
                container=False)
            with gr.Row():
                aud = gr.Audio(label='Speak', source="microphone", type='filepath')
                btn = gr.UploadButton("📁 Upload motion",
                                      elem_id="upload",
                                      file_types=["file"],
                                      variant='primary')
                regen = gr.Button("🔄 Regenerate", elem_id="regen")
                clear = gr.ClearButton([txt, chatbot, aud], value='🗑️ Clear')

        with gr.Column(scale=0.15, min_width=150):
            method = gr.Dropdown(["slow", "fast"],
                                label="Render method",
                                interactive=True,
                                elem_id="method",
                                value="fast")
            language = gr.Dropdown(["English", "中文"],
                                label="Speech language",
                                interactive=True,
                                elem_id="language",
                                value="English")

    txt_msg = txt.submit(
        add_text, [chatbot, txt, motion_uploaded, data_stored, method],
        [chatbot, txt, motion_uploaded, data_stored],
        queue=False).then(bot, [chatbot, motion_uploaded, data_stored, method],
                          [chatbot, motion_uploaded, data_stored])
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn, txt, motion_uploaded],
                          [chatbot, txt, motion_uploaded],
                          queue=False)
    aud_msg = aud.stop_recording(
        add_audio, [chatbot, aud, data_stored], [chatbot, data_stored],
        queue=False).then(bot, [chatbot, motion_uploaded, data_stored, method],
                          [chatbot, motion_uploaded, data_stored])
    regen_msg = regen.click(bot,
                            [chatbot, motion_uploaded, data_stored, method],
                            [chatbot, motion_uploaded, data_stored])

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8888, debug=True)
