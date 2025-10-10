import gc
import logging
from argparse import ArgumentParser
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import gradio as gr
import torch
import torchaudio

from mmaudiosep.eval_utils import (ModelConfig, VideoInfo, all_model_cfg, separate, 
                                load_video, make_video, setup_eval_logging)
from mmaudiosep.model.networks import MMAudioSep, get_my_mmaudiosep
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils
import subprocess

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    log.warning('CUDA/MPS are not available, running on CPU')
dtype = torch.bfloat16

model: ModelConfig = all_model_cfg['large_44k']
model.download_if_needed()
output_dir = Path('./output/gradio')

setup_eval_logging()


def get_model() -> tuple[MMAudioSep, FeaturesUtils, SequenceConfig]:
    seq_cfg = model.seq_cfg

    net: MMAudioSep = get_my_mmaudiosep(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=True)
    feature_utils = feature_utils.to(device, dtype).eval()

    return net, feature_utils, seq_cfg


net, feature_utils, seq_cfg = get_model()

def extract_raw_audio(mp4_path, start_sec, duration_sec, target_sr=44100):
    temp_wav = "temp.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", mp4_path,
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-ar", str(target_sr),
        "-ac", "2", 
        "-vn",
        temp_wav
    ]
    subprocess.run(command, check=True)

    waveform = load_audio(temp_wav, duration_sec)

    return waveform

def load_audio(audio_path: str, duration: float) :
    sr=44100
    audio_mixture, fs=torchaudio.load(audio_path)
    if sr != fs :
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=sr)
        audio_mixture = resampler(audio_mixture)

    samples=int(duration*sr)
    if samples % 1024 != 0 :
        samples += 1024
    samples=(samples//1024)*1024
    
    current_length = audio_mixture.shape[1]
    if current_length < samples:
        pad_length = samples - current_length
        padding = torch.zeros((audio_mixture.shape[0], pad_length), dtype=audio_mixture.dtype, device=audio_mixture.device)
        audio_mixture = torch.cat([audio_mixture, padding], dim=1)
    else:
        audio_mixture = audio_mixture[:, :samples]

    audio_mixture=audio_mixture.to(dtype)

    return audio_mixture

@torch.inference_mode()
def video_query_separation(video: gr.Video,
                   prompt: str, negative_prompt: str, seed: int, num_steps: int,
                   cfg_strength: float, duration: float):

    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)


    video_info = load_video(video, duration)
    clip_frames = video_info.clip_frames
    sync_frames = video_info.sync_frames
    duration = video_info.duration_sec
    clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audio_mixture = extract_raw_audio(video, start_sec=0, duration_sec=duration)
    if audio_mixture.shape[0] != 1 : # treat as stereo signal
        clip_frames = torch.cat([clip_frames, clip_frames], dim=0)
        sync_frames = torch.cat([sync_frames, sync_frames], dim=0)
    audios = separate(clip_frames,
                      sync_frames, [prompt]*audio_mixture.shape[0],
                      negative_text=[negative_prompt]*audio_mixture.shape[0],
                      audio_mixture=audio_mixture,
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu().squeeze(1) # audios : [bs, 1, length]

    current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(exist_ok=True, parents=True)
    # debug usage
    audio_save_path = output_dir / f'{current_time_string}.flac'
    torchaudio.save(audio_save_path, audio, seq_cfg.sampling_rate)
    
    video_save_path = output_dir / f'{current_time_string}.mp4'
    make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)

    gc.collect()
    return video_save_path

@torch.inference_mode()
def text_query_separation(mixture: gr.Audio, prompt: str, negative_prompt: str, seed: int, num_steps: int, cfg_strength: float,
                  duration: float):

    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    clip_frames = sync_frames = None
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audio_mixture=load_audio(mixture, duration)
    audios = separate(clip_frames,
                      sync_frames, [prompt]*audio_mixture.shape[0],
                      negative_text=[negative_prompt]*audio_mixture.shape[0],
                      audio_mixture=audio_mixture,
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu().squeeze(1) # audios : [bs, 1, length]

    current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(exist_ok=True, parents=True)
    audio_save_path = output_dir / f'{current_time_string}.flac'
    torchaudio.save(audio_save_path, audio, seq_cfg.sampling_rate)
    gc.collect()
    return audio_save_path


video_query_separation_tab = gr.Interface(
    fn=video_query_separation,
    description="""
    Code: <a href="https://github.com/sony/mmaudiosep">https://github.com/sony/mmaudiosep</a><br>

    NOTE: It takes longer to process high-resolution videos (>384 px on the shorter side). 
    Doing so does not improve results.
    """,
    inputs=[
        gr.Video(),
        #gr.Audio(type='filepath'),
        gr.Text(label='Prompt'),
        gr.Text(label='Negative prompt'),
        gr.Number(label='Seed (-1: random)', value=42, precision=0, minimum=-1),
        gr.Number(label='Num steps', value=25, precision=0, minimum=1),
        gr.Number(label='Guidance Strength', value=4.5, minimum=1),
        gr.Number(label='Duration (sec)', value=8, minimum=1),
    ],
    outputs='playable_video',
    cache_examples=False,
    title='MMAudioSep — Video Query Separation',
    )

text_query_separation_tab = gr.Interface(
    fn=text_query_separation,
    description="""
    Code: <a href="https://github.com/sony/mmaudiosep">https://github.com/sony/mmaudiosep</a><br>
    """,
    inputs=[
        gr.Audio(type='filepath'),
        gr.Text(label='Prompt'),
        gr.Text(label='Negative prompt'),
        gr.Number(label='Seed (-1: random)', value=42, precision=0, minimum=-1),
        gr.Number(label='Num steps', value=25, precision=0, minimum=1),
        gr.Number(label='Guidance Strength', value=4.5, minimum=1),
        gr.Number(label='Duration (sec)', value=8, minimum=1),
    ],
    outputs='audio',
    cache_examples=False,
    title='MMAudioSep — Text-Query Separation',
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()

    gr.TabbedInterface([video_query_separation_tab, text_query_separation_tab],
                       ['Video-Query', 'Text-Qeury']).launch(
                       server_name="0.0.0.0", server_port=args.port, allowed_paths=[output_dir])
