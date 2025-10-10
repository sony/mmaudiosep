import dataclasses
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from colorlog import ColoredFormatter
from PIL import Image
from torchvision.transforms import v2

from mmaudiosep.data.av_utils import ImageInfo, VideoInfo, read_frames, read_frames_with_transform, reencode_with_audio, remux_from_video_and_audio
from mmaudiosep.model.networks import MMAudioSep

from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.sequence_config import CONFIG_16K, CONFIG_44K, SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.utils.download_utils import download_model_if_needed

### for debug memory usage
#import psutil

log = logging.getLogger()


@dataclasses.dataclass
class ModelConfig:
    model_name: str
    model_path: Path
    vae_path: Path
    bigvgan_16k_path: Optional[Path]
    mode: str
    synchformer_ckpt: Path = Path('./ext_weights/synchformer_state_dict.pth')

    @property
    def seq_cfg(self) -> SequenceConfig:
        if self.mode == '16k':
            return CONFIG_16K
        elif self.mode == '44k':
            return CONFIG_44K

    def download_if_needed(self):
        #download_model_if_needed(self.model_path)
        download_model_if_needed(self.vae_path)
        if self.bigvgan_16k_path is not None:
            download_model_if_needed(self.bigvgan_16k_path)
        download_model_if_needed(self.synchformer_ckpt)


small_16k = ModelConfig(model_name='small_16k',
                        model_path=Path('./weights/mmaudiosep_small_16k.pth'),
                        vae_path=Path('./ext_weights/v1-16.pth'),
                        bigvgan_16k_path=Path('./ext_weights/best_netG.pt'),
                        mode='16k')
small_44k = ModelConfig(model_name='small_44k',
                        model_path=Path('./weights/mmaudiosep_small_44k.pth'),
                        vae_path=Path('./ext_weights/v1-44.pth'),
                        bigvgan_16k_path=None,
                        mode='44k')
medium_44k = ModelConfig(model_name='medium_44k',
                         model_path=Path('./weights/mmaudiosep_medium_44k.pth'),
                         vae_path=Path('./ext_weights/v1-44.pth'),
                         bigvgan_16k_path=None,
                         mode='44k')
large_44k = ModelConfig(model_name='large_44k',
                        model_path=Path('./weights/mmaudiosep_large_44k.pth'),
                        vae_path=Path('./ext_weights/v1-44.pth'),
                        bigvgan_16k_path=None,
                        mode='44k')

all_model_cfg: dict[str, ModelConfig] = {
    'small_16k': small_16k,
    'small_44k': small_44k,
    'medium_44k': medium_44k,
    'large_44k': large_44k,
}


def generate(
    clip_video: Optional[torch.Tensor],
    sync_video: Optional[torch.Tensor],
    text: Optional[list[str]],
    *,
    negative_text: Optional[list[str]] = None,
    feature_utils: FeaturesUtils,
    net: MMAudioSep,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    image_input: bool = False,
) -> torch.Tensor:
    device = feature_utils.device
    dtype = feature_utils.dtype

    bs = len(text)
    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(clip_video,
                                                             batch_size=bs *
                                                             clip_batch_size_multiplier)
        if image_input:
            clip_features = clip_features.expand(-1, net.clip_seq_len, -1)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None and not image_input:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(sync_video,
                                                             batch_size=bs *
                                                             sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    if text is not None:
        text_features = feature_utils.encode_text(text)
    else:
        text_features = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    if net.latent_dim == 40 :
        audio_len= net.latent_seq_len * 1024
    else : 
        audio_len= net.latent_seq_len * 512
    audio_mixture = torch.randn(bs, audio_len).to(dtype) * 0.9 # [TODO] using white noise as mixture. 0.9 is temporary value.

    dist = feature_utils.encode_audio(audio_mixture)
    a_mean = dist.mean.detach().transpose(1, 2)
    a_std = dist.std.detach().transpose(1, 2)
    a_randn = torch.empty_like(a_mean).normal_(generator=rng)
    x_cond = a_mean + a_std * a_randn

    x0 = torch.randn(bs,
                     net.latent_seq_len,
                     net.latent_dim,
                     device=device,
                     dtype=dtype,
                     generator=rng)
    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_features, x_cond)
    empty_conditions = net.get_empty_conditions(
        bs, x_cond=x_cond, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(t, x, preprocessed_conditions, empty_conditions,
                                                   cfg_strength)
    x1 = fm.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio

def separate(
    clip_video: Optional[torch.Tensor],
    sync_video: Optional[torch.Tensor],
    text: Optional[list[str]],
    audio_mixture: torch.Tensor,
    *,
    negative_text: Optional[list[str]] = None,
    feature_utils: FeaturesUtils,
    net: MMAudioSep,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    image_input: bool = False,
) -> torch.Tensor:
    device = feature_utils.device
    dtype = feature_utils.dtype

    bs = len(text)
    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(clip_video,
                                                             batch_size=bs *
                                                             clip_batch_size_multiplier)
        if image_input:
            clip_features = clip_features.expand(-1, net.clip_seq_len, -1)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None and not image_input:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(sync_video,
                                                             batch_size=bs *
                                                             sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    if text is not None:
        text_features = feature_utils.encode_text(text)
    else:
        text_features = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)
    dist = feature_utils.encode_audio(audio_mixture)

    a_mean = dist.mean.detach().transpose(1, 2)
    a_std = dist.std.detach().transpose(1, 2)
    a_randn = torch.empty_like(a_mean).normal_(generator=rng)
    x_cond = a_mean + a_std * a_randn

    x0 = torch.randn(bs,
                     net.latent_seq_len,
                     net.latent_dim,
                     device=device,
                     dtype=dtype,
                     generator=rng)
    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_features, x_cond)
    empty_conditions = net.get_empty_conditions(
        bs, x_cond=x_cond, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(t, x, preprocessed_conditions, empty_conditions,
                                                   cfg_strength)
    x1 = fm.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio

LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)


_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


def load_video(video_path: Path, duration_sec: float, load_all_frames: bool = True) -> VideoInfo:

    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    """
    # original code
    output_frames, all_frames, orig_fps = read_frames(video_path,
                                                      list_of_fps=[_CLIP_FPS, _SYNC_FPS],
                                                      start_sec=0,
                                                      end_sec=duration_sec,
                                                      need_all_frames=load_all_frames)

    clip_chunk, sync_chunk = output_frames
    clip_chunk = torch.from_numpy(clip_chunk).permute(0, 3, 1, 2)
    sync_chunk = torch.from_numpy(sync_chunk).permute(0, 3, 1, 2)
    clip_frames = clip_transform(clip_chunk)
    sync_frames = sync_transform(sync_chunk)
    print(clip_frames.shape, sync_frames.shape)
    clip_length_sec = clip_frames.shape[0] / _CLIP_FPS
    sync_length_sec = sync_frames.shape[0] / _SYNC_FPS
    """

    # new code for less CPU memory
    # -------------------------------------------------------------------
    output_frames, all_frames, orig_fps = read_frames_with_transform(video_path,
                                                      list_of_fps_and_transform=[(_CLIP_FPS,clip_transform), (_SYNC_FPS,sync_transform)],
                                                      start_sec=0,
                                                      end_sec=duration_sec,
                                                      need_all_frames=load_all_frames)
    clip_frames, sync_frames=output_frames
    clip_length_sec = clip_frames.shape[0] / _CLIP_FPS
    sync_length_sec = sync_frames.shape[0] / _SYNC_FPS
    # -------------------------------------------------------------------

    if clip_length_sec < duration_sec:
        log.warning(f'Clip video is too short: {clip_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Truncating to {clip_length_sec:.2f} sec')
        duration_sec = clip_length_sec

    if sync_length_sec < duration_sec:
        log.warning(f'Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Truncating to {sync_length_sec:.2f} sec')
        duration_sec = sync_length_sec

    clip_frames = clip_frames[:int(_CLIP_FPS * duration_sec)]
    sync_frames = sync_frames[:int(_SYNC_FPS * duration_sec)]

    video_info = VideoInfo(
        duration_sec=duration_sec,
        fps=orig_fps,
        clip_frames=clip_frames,
        sync_frames=sync_frames,
        all_frames=all_frames if load_all_frames else None,
    )
    return video_info


def load_image(image_path: Path) -> VideoInfo:
    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img=Image.open(image_path)
    frame=np.array(img)

    # fixed even number
    width, height = img.size
    even_width = width if width % 2 == 0 else width + 1
    even_height = height if height % 2 == 0 else height + 1

    # resize if needed
    if (width != even_width) or (height != even_height):
        img = img.resize((even_width, even_height), Image.BICUBIC)
        frame = np.array(img)
    
    clip_chunk = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2)
    sync_chunk = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2)

    clip_frames = clip_transform(clip_chunk)
    sync_frames = sync_transform(sync_chunk)

    video_info = ImageInfo(
        clip_frames=clip_frames,
        sync_frames=sync_frames,
        original_frame=frame,
    )
    return video_info


def make_video(video_info: VideoInfo, output_path: Path, audio: torch.Tensor, sampling_rate: int):
    reencode_with_audio(video_info, output_path, audio, sampling_rate)

def make_video_from_video_and_audio(video_path: Path, output_path: Path, audio: torch.Tensor, sampling_rate: int):
    remux_from_video_and_audio(video_path, audio, output_path, sampling_rate)
