import numpy as np
import torch

from utils.helper_funcs import load_wav_to_torch_with_librosa


def get_mel(filename, params, stft, mel_save_path):
    # Calculating mel-spectrogram
    audio, sampling_rate = load_wav_to_torch_with_librosa(filename, params.sampling_rate)

    audio_norm = audio  # / self.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0).cpu().detach().numpy()

    mel_name = filename.split("/")[-1][:-4]
    mel_path = mel_save_path + mel_name + ".npy"
    np.save(mel_path, melspec)
