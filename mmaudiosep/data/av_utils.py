from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

import av
import numpy as np
import torch
from av import AudioFrame


@dataclass
class VideoInfo:
    duration_sec: float
    fps: Fraction
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    all_frames: Optional[list[np.ndarray]]

    @property
    def height(self):
        return self.all_frames[0].shape[0]

    @property
    def width(self):
        return self.all_frames[0].shape[1]

    @classmethod
    def from_image_info(cls, image_info: 'ImageInfo', duration_sec: float,
                        fps: Fraction) -> 'VideoInfo':
        num_frames = int(duration_sec * fps)
        all_frames = [image_info.original_frame] * num_frames
        return cls(duration_sec=duration_sec,
                   fps=fps,
                   clip_frames=image_info.clip_frames,
                   sync_frames=image_info.sync_frames,
                   all_frames=all_frames)


@dataclass
class ImageInfo:
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    original_frame: Optional[np.ndarray]

    @property
    def height(self):
        return self.original_frame.shape[0]

    @property
    def width(self):
        return self.original_frame.shape[1]

# modified
def read_frames_with_transform(video_path: Path, list_of_fps_and_transform: list[tuple[float,callable]], 
                               start_sec: float, end_sec: float,
                               need_all_frames: bool)-> tuple[list[np.ndarray], list[np.ndarray], Fraction]:
    output_frames = [[] for _ in list_of_fps_and_transform]
    next_frame_time_for_each_fps = [0.0 for _ in list_of_fps_and_transform]
    time_delta_for_each_fps = [1 / fps for fps, _ in list_of_fps_and_transform]
    all_frames = []
    # container = av.open(video_path)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = stream.guessed_rate
        stream.thread_type = 'AUTO'
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_time = frame.time
                if frame_time < start_sec:
                    continue
                if frame_time > end_sec:
                    break

                frame_np = None
                if need_all_frames:
                    frame_np = frame.to_ndarray(format='rgb24')
                    all_frames.append(frame_np)
                # transform image for CLIP/SyncFormer to reduce CPU RAM usage
                for i, (fps_, transform_) in enumerate(list_of_fps_and_transform):
                    this_time = frame_time
                    while this_time >= next_frame_time_for_each_fps[i]:
                        if frame_np is None:
                            frame_np = frame.to_ndarray(format='rgb24')
                        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)
                        transformed_frame = transform_(frame_tensor)
                        output_frames[i].append(transformed_frame) # [changed] torch tensor
                        next_frame_time_for_each_fps[i] += time_delta_for_each_fps[i]

    output_frames = [torch.stack(frames) for frames in output_frames] # [changed] torch tensor
    return output_frames, all_frames, fps

# original
def read_frames(video_path: Path, list_of_fps: list[float], start_sec: float, end_sec: float,
                need_all_frames: bool) -> tuple[list[np.ndarray], list[np.ndarray], Fraction]:
    output_frames = [[] for _ in list_of_fps]
    next_frame_time_for_each_fps = [0.0 for _ in list_of_fps]
    time_delta_for_each_fps = [1 / fps for fps in list_of_fps]
    all_frames = []

    # container = av.open(video_path)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = stream.guessed_rate
        stream.thread_type = 'AUTO'
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_time = frame.time
                if frame_time < start_sec:
                    continue
                if frame_time > end_sec:
                    break

                frame_np = None
                if need_all_frames:
                    frame_np = frame.to_ndarray(format='rgb24')
                    all_frames.append(frame_np)

                for i, _ in enumerate(list_of_fps):
                    this_time = frame_time
                    while this_time >= next_frame_time_for_each_fps[i]:
                        if frame_np is None:
                            frame_np = frame.to_ndarray(format='rgb24')

                        output_frames[i].append(frame_np)
                        next_frame_time_for_each_fps[i] += time_delta_for_each_fps[i]

    output_frames = [np.stack(frames) for frames in output_frames]
    return output_frames, all_frames, fps


def reencode_with_audio(video_info: VideoInfo, output_path: Path, audio: torch.Tensor,
                        sampling_rate: int):
    container = av.open(output_path, 'w')
    output_video_stream = container.add_stream('h264', video_info.fps)
    output_video_stream.codec_context.bit_rate = 10 * 1e6  # 10 Mbps
    output_video_stream.width = video_info.width
    output_video_stream.height = video_info.height
    output_video_stream.pix_fmt = 'yuv420p'

    output_audio_stream = container.add_stream('aac', sampling_rate)

    # encode video
    for image in video_info.all_frames:
        image = av.VideoFrame.from_ndarray(image)
        packet = output_video_stream.encode(image)
        container.mux(packet)

    for packet in output_video_stream.encode():
        container.mux(packet)

    # convert float tensor audio to numpy array
    audio_np = audio.numpy().astype(np.float32)
    if audio_np.shape[0] == 2 : # stereo
        audio_frame = AudioFrame.from_ndarray(audio_np, format='fltp', layout='stereo')
    else :
        audio_frame = AudioFrame.from_ndarray(audio_np, format='fltp', layout='mono')
    audio_frame.sample_rate = sampling_rate

    for packet in output_audio_stream.encode(audio_frame):
        container.mux(packet)

    for packet in output_audio_stream.encode():
        container.mux(packet)

    container.close()

# [Note by PON] this API didn't work with pyav v14.1.0
def remux_with_audio(video_path: Path, audio: torch.Tensor, output_path: Path, sampling_rate: int):
    """
    NOTE: I don't think we can get the exact video duration right without re-encoding
    so we are not using this but keeping it here for reference
    """
    video = av.open(video_path)
    output = av.open(output_path, 'w')
    input_video_stream = video.streams.video[0]
    output_video_stream = output.add_stream(template=input_video_stream)
    output_audio_stream = output.add_stream('aac', sampling_rate)

    duration_sec = audio.shape[-1] / sampling_rate

    for packet in video.demux(input_video_stream):
        # We need to skip the "flushing" packets that `demux` generates.
        if packet.dts is None:
            continue
        # We need to assign the packet to the new stream.
        packet.stream = output_video_stream
        output.mux(packet)

    # convert float tensor audio to numpy array
    audio_np = audio.numpy().astype(np.float32)
    audio_frame = av.AudioFrame.from_ndarray(audio_np, format='flt', layout='mono')
    audio_frame.sample_rate = sampling_rate

    for packet in output_audio_stream.encode(audio_frame):
        output.mux(packet)

    for packet in output_audio_stream.encode():
        output.mux(packet)

    video.close()
    output.close()

def remux_from_video_and_audio(video_path: Path, audio: torch.Tensor, output_path: Path, sampling_rate: int):
    input_container = av.open(video_path)
    input_vstream=input_container.streams.video[0]
    codec_name = input_vstream.codec_context.name

    output_video = av.open(output_path, 'w')

    output_video_stream = output_video.add_stream_from_template(input_vstream)

    output_audio_stream = output_video.add_stream('aac', rate=sampling_rate)
    duration_sec = audio.shape[-1] / sampling_rate

    if codec_name=="vp9":
        for packet in input_container.demux(input_vstream):
            if packet.dts is None:
                continue
            # check duration
            elapsed_time = packet.time_base * packet.dts
            if elapsed_time > duration_sec:
                break
            output_video.mux(packet)
    else : # h264
        for packet in input_container.demux(input_vstream):
            if packet.dts is None:
                continue
            # check duration
            elapsed_time = packet.time_base * packet.dts
            if elapsed_time >= duration_sec:
                break

            for frame in packet.decode():
                img = frame.to_ndarray(format="rgb24")
                # for debug
                if False :
                    from PIL import Image, ImageDraw
                    img_height, img_width, _ = img.shape
                    
                    rect_width = img_height
                    rect_height = img_height
                    
                    left = (img_width - rect_width) // 2
                    top = (img_height - rect_height) // 2
                    
                    right = left + rect_width
                    bottom = top + rect_height
                    
                    pil_img = Image.fromarray(img)
                    draw = ImageDraw.Draw(pil_img)
                    draw.rectangle([left, top, right, bottom], outline="red", width=10)
                    
                    img_with_rect = np.array(pil_img)
                    
                    image = av.VideoFrame.from_ndarray(img_with_rect, format="rgb24")
                else :
                    image = av.VideoFrame.from_ndarray(img)
                p = output_video_stream.encode(image)
                output_video.mux(p)

    # convert float tensor audio to numpy array
    audio_np = audio.numpy().astype(np.float32)
    if audio_np.shape[0] == 2 : # stereo
        audio_frame = AudioFrame.from_ndarray(audio_np, format='fltp', layout='stereo')
    else :
        audio_frame = AudioFrame.from_ndarray(audio_np, format='fltp', layout='mono')
    audio_frame.sample_rate = sampling_rate

    for packet in output_audio_stream.encode(audio_frame):
        output_video.mux(packet)

    for packet in output_video_stream.encode():
        output_video.mux(packet)

    for packet in output_audio_stream.encode():
        output_video.mux(packet)

    output_video.close()
    input_container.close()

