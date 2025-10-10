import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from mmaudiosep.eval_utils import (ModelConfig, all_model_cfg, generate, separate, load_video, make_video, make_video_from_video_and_audio,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudiosep.model.networks import MMAudioSep, get_my_mmaudiosep
from mmaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--variant',
                        type=str,
                        default='large_44k',
                        help='small_16k, small_44k, medium_44k, large_44k')
    parser.add_argument('--video', type=Path, help='Path to the video file')
    parser.add_argument('--audio', type=Path, help='Path to the audio file')
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)

    parser.add_argument('--mask_away_clip', action='store_true')

    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--skip_video_composite', action='store_true')
    parser.add_argument('--full_precision', action='store_true')

    args = parser.parse_args()

    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model: ModelConfig = all_model_cfg[args.variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    if args.video:
        video_path: Path = Path(args.video).expanduser()
    else:
        video_path = None
    if args.audio:
        audio_path: Path = Path(args.audio).expanduser()
    else:
        audio_path = None
    prompt: str = args.prompt
    negative_prompt: str = args.negative_prompt
    output_dir: str = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength
    skip_video_composite: bool = args.skip_video_composite
    mask_away_clip: bool = args.mask_away_clip
    load_all_frames = False # if you want to use less memroy. [ToDo] option

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    net: MMAudioSep = get_my_mmaudiosep(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=True)

    feature_utils = feature_utils.to(device, dtype).eval()

    if video_path is not None:
        log.info(f'Using video {video_path}')
        video_info = load_video(video_path, duration, load_all_frames=load_all_frames) # added load_all_frames flags to reduce CPU RAM
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        duration = video_info.duration_sec
        if mask_away_clip:
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)
    else:
        log.info('No video provided -- text-to-audio mode')
        clip_frames = sync_frames = None

    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    if args.variant == "small_16k" :
        sr=16000
    else :
        sr=44100
    if audio_path is not None :
        audio_mixture, fs=torchaudio.load(audio_path)
        if sr != fs :
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=sr)
            audio_mixture = resampler(audio_mixture)

        if audio_mixture.shape[0] != 1 :
            audio_mixture = audio_mixture.mean(dim=0).unsqueeze(0)
        samples=int(duration*sr)
        if sr == 16000 :
            if samples % 512 != 0 :
                samples += 512
            samples=(samples//512)*512
        else :
            if samples % 1024 != 0 :
                samples += 1024
            samples=(samples//1024)*1024
        audio_mixture=audio_mixture[:,0:samples]
        audio_mixture=audio_mixture.to(dtype)
    else :
        audio_mixture=None

    log.info(f'Prompt: {prompt}')
    log.info(f'Negative prompt: {negative_prompt}')
    if audio_path is not None : # [Note] separation mode
        audios = separate(clip_frames,
                          sync_frames, [prompt],
                          negative_text=[negative_prompt],
                          audio_mixture=audio_mixture,
                          feature_utils=feature_utils,
                          net=net,
                          fm=fm,
                          rng=rng,
                          cfg_strength=cfg_strength)
    else : # [Note] generation mode
        audios = generate(clip_frames,
                          sync_frames, [prompt],
                          negative_text=[negative_prompt],
                          feature_utils=feature_utils,
                          net=net,
                          fm=fm,
                          rng=rng,
                          cfg_strength=cfg_strength)

    audio = audios.float().cpu()[0]
    if video_path is not None:
        save_path = output_dir / f'{video_path.stem}.flac'
    else:
        safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
        save_path = output_dir / f'{safe_filename}.flac'
    torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

    log.info(f'Audio saved to {save_path}')
    if video_path is not None and not skip_video_composite:
        video_save_path = output_dir / f'{video_path.stem}.mp4'
        if load_all_frames == True : # ReEnc from pre-load images
            make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
        else : # Remux
            make_video_from_video_and_audio(video_path, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
        log.info(f'Video saved to {output_dir / video_save_path}')

    log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))


if __name__ == '__main__':
    main()
