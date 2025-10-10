import json
import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
import torchaudio

from mmaudio.utils.dist_utils import local_rank

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class VideoDataset(Dataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        audio_path: Union[str, Path],
        *,
        duration_sec: float = 8.0,
    ):
        self.video_root = Path(video_root)
        self.audio_path = Path(audio_path)

        self.duration_sec = duration_sec

        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self.clip_transform = v2.Compose([
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # to be implemented by subclasses
        self.captions = {}
        self.videos = sorted(list(self.captions.keys()))

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        caption = self.captions[video_id]
        negative = self.negatives[video_id]

        audio_path = f'{self.audio_path}/{video_id}.flac'
        audio, fs = torchaudio.load(audio_path)
        if audio.size()[1] < self.audio_len :
            audio= F.pad(audio, (0, self.audio_len-audio.size()[1]))
        mixture = audio[:, :self.audio_len].squeeze()

        reader = StreamingMediaDecoder(self.video_root / (video_id + '.mp4'))
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
            frame_rate=_CLIP_FPS,
            format='rgb24',
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = data_chunk[0]
        sync_chunk = data_chunk[1]
        if clip_chunk is None:
            raise RuntimeError(f'CLIP video returned None {video_id}')
        if clip_chunk.shape[0] < self.clip_expected_length:
            raise RuntimeError(
                f'CLIP video too short {video_id}, expected {self.clip_expected_length}, got {clip_chunk.shape[0]}'
            )

        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_id}')
        if sync_chunk.shape[0] < self.sync_expected_length:
            raise RuntimeError(
                f'Sync video too short {video_id}, expected {self.sync_expected_length}, got {sync_chunk.shape[0]}'
            )

        # truncate the video
        clip_chunk = clip_chunk[:self.clip_expected_length]
        if clip_chunk.shape[0] != self.clip_expected_length:
            raise RuntimeError(f'CLIP video wrong length {video_id}, '
                               f'expected {self.clip_expected_length}, '
                               f'got {clip_chunk.shape[0]}')
        clip_chunk = self.clip_transform(clip_chunk)

        sync_chunk = sync_chunk[:self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(f'Sync video wrong length {video_id}, '
                               f'expected {self.sync_expected_length}, '
                               f'got {sync_chunk.shape[0]}')
        sync_chunk = self.sync_transform(sync_chunk)

        data = {
            'name': video_id,
            'caption': caption,
            'no_caption': '',
            'negative': negative,
            'mixture': mixture,
            'clip_video': clip_chunk,
            'sync_video': sync_chunk,
        }

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f'Error loading video {self.videos[idx]}: {e}')
            return None

    def __len__(self):
        return len(self.captions)

class Mixed(VideoDataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        tsv_path: Union[str, Path],
        audio_path: Union[str, Path],
        duration_sec: float = 10.0,
        mode: str = '44k'
    ):
        super().__init__(video_root, audio_path, duration_sec=duration_sec)
        self.video_root = Path(video_root)
        self.tsv_path = Path(tsv_path)
        self.audio_path = Path(audio_path)
        self.mode = mode

        if self.mode == '44k':
            self.audio_len= duration_sec * 44100 + (1024 - duration_sec*44100 % 1024) % 1024
        elif self.mode == '16k':
            self.audio_len= duration_sec * 16000 + (512 - duration_sec*16000 % 512) % 512
        else :
            self.audio_len= duration_sec * 16000 + (512 - duration_sec*16000 % 512) % 512

        videos = sorted(os.listdir(self.video_root))
        audios = sorted(os.listdir(self.audio_path))
        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {video_root}')
        self.captions = {}
        self.negatives = {}

        df = pd.read_csv(tsv_path, sep='\t').to_dict('records')

        videos_no_found = []
        audios_no_found = []
        for row in df:
            video_id = str(row['id'])
            # this is how our videos are named
            video_name = f'{video_id}'
            if video_name + '.mp4' not in videos:
                videos_no_found.append(video_name)
                continue
            audio_name = f'{video_id}'
            if audio_name + '.flac' not in audios:
                audios_no_found.append(audio_name)
                continue

            self.captions[video_name] = row['label']
            self.negatives[video_name] = row['negative_label']

        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {video_root}')
            log.info(f'{len(self.captions)} useable videos found')
            if videos_no_found:
                log.info(f'{len(videos_no_found)} found in {tsv_path} but not in {video_root}')
                log.info(
                    'A small amount is expected, as not all videos are still available on YouTube')

        self.videos = sorted(list(self.captions.keys()))
        self.audios = sorted(list(self.captions.keys()))

