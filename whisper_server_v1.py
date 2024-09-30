import tornado.ioloop
import tornado.web
import tornado.gen
import base64
from faster_whisper import WhisperModel
import io
import numpy as np
import torch
from pydub import AudioSegment
import os

cuda_available = torch.cuda.is_available()

if cuda_available:
    try:
        import ctypes
        ctypes.CDLL("libcudnn_ops_infer.so.8")
        device = "cuda"
        compute_type = "float16"
    except OSError:
        print("CUDA libraries not found. falling back to CPU.")
        device = "cpu"
        compute_type = "int8"
else:
    device = "cpu"
    compute_type = "int8"

model = WhisperModel("large-v2", device=device, compute_type=compute_type)

def split_and_tag_channels(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path, format="wav")
    left_channel = audio.split_to_mono()[0]
    right_channel = audio.split_to_mono()[1]
    left_channel = (left_channel, "Agent")
    right_channel = (right_channel, "Member")
    return left_channel, right_channel

def transcribe_audio(audio_segment):
    audio_array = np.array(audio_segment.get_array_of_samples())
    audio_array = audio_array.astype(np.float32) / 32768.0
    result = model.transcribe(audio_array)
    return result

def combine_transcripts(left_transcript, right_transcript):
    all_segments = []
    for segment in left_transcript:
        segment_dict = segment._asdict()
        segment_dict['speaker_role'] = 'Agent'
        all_segments.append(segment_dict)
    for segment in right_transcript:
        segment_dict = segment._asdict()
        segment_dict['speaker_role'] = 'Member'
        all_segments.append(segment_dict)

    all_segments.sort(key=lambda x: x['end'])
    return all_segments

def transcribe(audio_file_path):
    left_channel, right_channel = split_and_tag_channels(audio_file_path)

    left_transcript, _ = transcribe_audio(left_channel[0])
    right_transcript, _ = transcribe_audio(right_channel[0])

    combined_transcript = combine_transcripts(left_transcript, right_transcript)
    transcription = ' '.join([segment['text'] for segment in combined_transcript])
    return transcription

transcription_result = transcribe("audi_trimmed.wav")
print(transcription_result)
