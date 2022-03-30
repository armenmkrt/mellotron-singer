import argparse
import os
import struct
from functools import partial
from multiprocessing import Pool
from typing import List

import librosa
import numpy as np
import webrtcvad
from omegaconf import OmegaConf
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.ndimage import binary_dilation

from data.text import _clean_text


def prepare_dataset_for_alignment(wavs_folder: str, metadata_path: str,
                                  sampling_rate: int, max_wav_value: float,
                                  text_cleaners: List[str], output_dir: str,
                                  num_processes: int = None,
                                  normalise: bool = False,
                                  target_dbfs: float = -23,
                                  run_vad: bool = False,
                                  vad_sampling_rate: int = 16000,
                                  start_end_trim: bool = False):
    """Prepare dataset format for MFA

    Args:
        wavs_folder: path of wav files
        metadata_path: path to metadata.csv
        sampling_rate: audio processing sampling rate
        max_wav_value: maximum value of audio
        text_cleaners: names of text cleaners
        output_dir: output directory for wavs and labs
        num_processes: number of processes
        normalise: normalise wav volume
        vad_sampling_rate:
        target_dbfs:
        run_vad: Remove long silences if True
        start_end_trim: Remove silences only from start and trail

    Returns:

    """
    os.makedirs(output_dir, exist_ok=True)
    if num_processes is None:
        num_processes = os.cpu_count() // 2

    map_function = partial(process_line, wavs_folder=wavs_folder,
                           output_dir=output_dir, sampling_rate=sampling_rate,
                           max_wav_value=max_wav_value, cleaners=text_cleaners,
                           target_dbfs=target_dbfs, normalise=normalise,
                           run_vad=run_vad, vad_sampling_rate=vad_sampling_rate,
                           start_end_trim=start_end_trim)

    with open(metadata_path, encoding="utf-8") as f:
        with Pool(processes=num_processes) as pool:
            pool.map(map_function, f)


def process_line(line: str, wavs_folder: str, output_dir: str,
                 sampling_rate: int, max_wav_value: float,
                 cleaners: List[str], normalise: bool,
                 target_dbfs: float = -23, run_vad: bool = False,
                 vad_sampling_rate: int = 16000, start_end_trim: bool = False):
    """Process single line of metadata file

    Args:
        line: part name and text separated with |
        wavs_folder: path of wav files
        output_dir: output directory for wavs and labs
        sampling_rate: audio processing sampling rate
        max_wav_value: maximum value of audio
        cleaners: names of text cleaners
        normalise: normalise wav volume
        target_dbfs:
        run_vad: Remove long silences if True
        vad_sampling_rate:
        start_end_trim:

    Returns:

    """
    parts = line.strip().split("|")
    base_name = parts[0]
    text = parts[-1]
    text = _clean_text(text, cleaners)
    speaker = base_name[:2]

    wav_path = os.path.join(wavs_folder, f"{base_name}.wav")
    speaker_folder = os.path.join(output_dir, speaker)
    os.makedirs(speaker_folder, exist_ok=True)
    out_wav_path = os.path.join(speaker_folder, f"{base_name}.wav")
    out_lab_path = os.path.join(speaker_folder, f"{base_name}.lab")
    if os.path.exists(out_wav_path) and os.path.exists(out_lab_path):
        return
    if os.path.exists(wav_path):
        wav, _ = librosa.load(wav_path, sr=sampling_rate)
        wav = wav / max(abs(wav)) * max_wav_value
        if run_vad:
            if sampling_rate != vad_sampling_rate:
                wav = librosa.resample(wav, orig_sr=sampling_rate, target_sr=vad_sampling_rate)
            wav = trim_long_silences(wav=wav, start_end_trim=start_end_trim)
        wavfile.write(out_wav_path, sampling_rate, wav.astype(np.int16))
        with open(out_lab_path, "w") as outfile:
            outfile.write(text)
        if normalise:
            sound = AudioSegment.from_file(wav_path)
            normalized_sound = match_target_amplitude(sound, target_dbfs)
            normalized_sound.export(out_wav_path, format="wav")


def match_target_amplitude(sound, target_dBFS: float = -23):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def trim_long_silences(wav, vad_window_length: int = 30, vad_sample_rate: int = 16000,
                       vad_max_silence_length: int = 6, vad_moving_average_width: int = 8,
                       start_end_trim: bool = False):
    int16_max = (2 ** 15) - 1
    samples_per_window = (vad_window_length * vad_sample_rate) // 1000
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=vad_sample_rate))
    voice_flags = np.array(voice_flags)

    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)
    audio_mask[:] = binary_dilation(audio_mask[:], np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    if start_end_trim:
        speech_start = 0
        for i, flag in enumerate(audio_mask):
            if audio_mask[i] and i != 0:
                if not audio_mask[i - 1]:
                    speech_start = i
                    break

        reversed_audio_mask = audio_mask[::-1]
        speech_end = 0
        for i, flag in enumerate(reversed_audio_mask):
            if reversed_audio_mask[i] and i != 0:
                if not reversed_audio_mask[i - 1]:
                    speech_end = i
                    break

        speech_end = audio_mask.shape[0] - speech_end
        return wav[speech_start:speech_end]
    else:
        return wav[audio_mask]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config .yaml file")
    parser.add_argument("-j", "--num_processes", default=8, type=int)
    parser.add_argument('--normalise', default=False, action="store_true")
    parser.add_argument('--run_vad', default=False, action="store_true")
    parser.add_argument('--start_end_trim', default=False, action='store_true')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    wavs_folder_ = os.path.join(config["path"]["dataset_path"], 'wavs')
    prepare_dataset_for_alignment(wavs_folder=wavs_folder_,
                                  output_dir=config["path"]["mfa_folder"],
                                  metadata_path=os.path.join(config["path"]["dataset_path"], 'metadata.csv'),
                                  sampling_rate=config["preprocessing"]["audio"]["sampling_rate"],
                                  max_wav_value=config["preprocessing"]["audio"]["max_wav_value"],
                                  text_cleaners=config["preprocessing"]["text"]["text_cleaners"],
                                  vad_sampling_rate=16000,
                                  num_processes=args.num_processes,
                                  normalise=args.normalise,
                                  target_dbfs=-23,
                                  run_vad=args.run_vad,
                                  start_end_trim=args.start_end_trim)
