import json
import os.path
import random
import re

import numpy as np
import torch
import torch.utils.data

import utils.helper_funcs as helpers
from data.audio.stft import TacotronSTFT
from data.text import phoneme_duration_to_sequence


class TextMelDurLoader(torch.utils.data.Dataset):
    """
        1) loads audio,embedding and text duration pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
        4) computes durations in frames and per phoneme durations count
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.dataset_path = hparams.dataset_path
        self.duration_folder_path = os.path.join(self.dataset_path, "duration")
        self.duration_in_sec_folder_path = os.path.join(self.dataset_path, "duration_in_sec")
        self.mel_folder_path = os.path.join(self.dataset_path, "mel")
        self.f0_folder_path = os.path.join(self.dataset_path, "f0s")

        self.audiopaths_and_text = helpers.load_file_paths_dur_and_phonemes(audiopaths_and_text)

        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.hop_length = hparams.hop_length
        self.mel_token_value = hparams.mel_token_value
        with open(os.path.join(self.dataset_path, "speakers.json"), "r") as f:
            self.speaker_mapping = json.load(f)

        pretrained_embeddings_np = np.zeros((len(self.speaker_mapping), hparams.speaker_embedding_dim),
                                            dtype=np.float32)

        for speaker, index in self.speaker_mapping.items():
            pretrained_embeddings_np[index] = self.get_embedding(embed_path=os.path.join(self.dataset_path,
                                                                                         'averaged_embeddings',
                                                                                         f'{speaker}.npy'))

        self.pretrained_embeddings = torch.from_numpy(pretrained_embeddings_np)
        self.durations_length_mean, self.durations_length_std = self.get_duration_stats()

        self.stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)

    def get_duration_stats(self):
        with open(os.path.join(self.dataset_path, 'duration_stats.json'), 'r') as f:
            duration_stats = json.load(f)
        return duration_stats["durations_mean"], duration_stats["durations_std"]

    def get_mel_text_pair(self, audiopath_and_text):
        # Separate filenames and texts, embeddings, durations
        audio_name, phonemes = audiopath_and_text[0], audiopath_and_text[1]

        speaker_id = audio_name.split("_")[0]
        mel_path = os.path.join(self.mel_folder_path, audio_name + ".npy")
        f0_path = os.path.join(self.f0_folder_path, audio_name + '.npy')
        duration_path = os.path.join(self.duration_folder_path, audio_name + ".npy")
        duration_in_sec_path = os.path.join(self.duration_in_sec_folder_path, audio_name + ".npy")

        # Calculating mel-spectrogram and loading embedding
        mel = self.get_mel(mel_path)
        f0 = self.get_f0(f0_path)
        duration_in_frames = self.load_durations_from_numpy(duration_path, in_seconds=False)
        duration_in_sec = self.load_durations_from_numpy(duration_in_sec_path, in_seconds=True)
        embedding = self.pretrained_embeddings[self.speaker_mapping[speaker_id]]

        # Converting phonemes to IDs
        phoneme_ids = self.get_phoneme_ids(phonemes)

        # Getting positional embedding IDs
        unpacked_durations = self.get_positional_embedding_ids(duration_in_frames)

        # Normalizing durations in seconds
        duration_in_sec = (duration_in_sec - self.durations_length_mean) / self.durations_length_std

        return phoneme_ids, duration_in_sec, unpacked_durations, duration_in_frames, mel, embedding, f0

    def get_mel(self, filename):
        # Calculating mel-spectrogram
        pad_matrices = np.ones((80, 2)) * self.mel_token_value  # Padding silence frames for sos and eos tokens

        mel_matrix = np.load(filename).T
        mel_matrix = np.concatenate((pad_matrices, mel_matrix, pad_matrices), axis=1)
        melspec = torch.from_numpy(mel_matrix)

        assert melspec.size(0) == self.stft.n_mel_channels, (
            'Mel dimension mismatch: given {}, expected {}'.format(
                melspec.size(0), self.stft.n_mel_channels))

        return melspec

    @staticmethod
    def get_f0(filename):
        f0 = np.load(filename)[0]

        # Padding zeros in place of sos and sos tokens
        f0 = np.insert(f0, 0, [0, 0])
        f0 = np.append(f0, [0, 0])[None, :]

        f0 = torch.from_numpy(f0)
        return f0

    @staticmethod
    def load_durations_from_numpy(path, in_seconds=True):
        durations = np.load(path)
        if in_seconds:
            durations = np.insert(durations, 0, [0.02])
            durations = np.append(durations, [0.02])
        else:
            durations = np.insert(durations, 0, [2])
            durations = np.append(durations, [2])
        return torch.from_numpy(durations)

    @staticmethod
    def get_positional_embedding_ids(durations_in_frames):
        unpacked_durations = []

        # Getting expanded durations in frames
        for duration in durations_in_frames:
            unpacked_durations.append(torch.arange(duration) + 1)

        unpacked_durations = torch.cat(unpacked_durations, dim=-1)

        return unpacked_durations

    @staticmethod
    def get_phoneme_ids(duration_phoneme_pair):
        phoneme_sequence = phoneme_duration_to_sequence(duration_phoneme_pair)
        phoneme_sequence = torch.IntTensor(phoneme_sequence)
        return phoneme_sequence

    @staticmethod
    def get_embedding(embed_path):
        return np.load(embed_path, allow_pickle=True)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelDurCollate:
    """
        Zero-pads model inputs and targets based on number of frames per step
    """

    def __init__(self, n_frames_per_step, dur_length_mean, dur_length_std):
        self.n_frames_per_step = n_frames_per_step

        self.scaled_zero = (0 - dur_length_mean) / dur_length_std
        self.embed_size = 256

    def __call__(self, batch):
        """
        Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [phonemes, durations, durations_unpacked, durations_in_frames, mel, embed, f0]
        """
        batch_size = len(batch)
        # Get longest unpacked duration for right zero padding
        _, unpacked_dur_sorted_decreasing = torch.sort(
            torch.LongTensor([x[2].size(0) for x in batch]),
            dim=0, descending=True)

        # Right zero-pad all one-hot phoneme sequences to max input length
        input_lengths, ids_of_sorted_phonemes = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        phonemes_padded = torch.LongTensor(batch_size, max_input_len)
        phonemes_padded.zero_()
        for i in range(len(ids_of_sorted_phonemes)):
            phonemes = batch[ids_of_sorted_phonemes[i]][0]
            phonemes_padded[i, :phonemes.size(0)] = phonemes

        # Right zero-pad all duration sequences to max input length
        duration_padded = torch.FloatTensor(batch_size, max_input_len)
        duration_padded.fill_(self.scaled_zero)
        for i in range(len(ids_of_sorted_phonemes)):
            durations = batch[ids_of_sorted_phonemes[i]][1]
            duration_padded[i, :durations.size(0)] = durations

        # Right zero-pad all unpacked durations
        get_longest_unpacked_duration = batch[unpacked_dur_sorted_decreasing[0]][2].size(0)
        unpacked_durations_padded = torch.LongTensor(batch_size, get_longest_unpacked_duration)
        unpacked_durations_padded.zero_()
        for i in range(len(ids_of_sorted_phonemes)):
            unpacked_durations = batch[ids_of_sorted_phonemes[i]][2]
            unpacked_durations_padded[i, :unpacked_durations.size(0)] = unpacked_durations

        # Right zero-pad all durations in frames
        durations_in_frames_padded = torch.LongTensor(batch_size, max_input_len)
        durations_in_frames_padded.zero_()
        for i in range(len(ids_of_sorted_phonemes)):
            durations_in_frames = batch[ids_of_sorted_phonemes[i]][3]
            durations_in_frames_padded[i, :durations_in_frames.size(0)] = durations_in_frames

        # Right zero-pad mel-spec
        num_mels = batch[0][4].size(0)
        max_target_len = max([x[4].size(1) for x in batch])

        mel_padded = torch.FloatTensor(batch_size, num_mels, max_target_len)
        mel_padded.zero_()

        f0_padded = torch.FloatTensor(batch_size, 1, max_target_len)
        f0_padded.zero_()

        output_lengths = torch.LongTensor(batch_size)

        embeddings = torch.FloatTensor(batch_size, self.embed_size)
        for i in range(len(ids_of_sorted_phonemes)):
            mel = batch[ids_of_sorted_phonemes[i]][4]
            f0 = batch[ids_of_sorted_phonemes[i]][6]

            mel_padded[i, :, :mel.size(1)] = mel
            f0_padded[i, :, :f0.size(1)] = f0

            output_lengths[i] = mel.size(1)

            embeddings[i] = batch[ids_of_sorted_phonemes[i]][5]

        return [phonemes_padded, duration_padded, unpacked_durations_padded, durations_in_frames_padded,
                input_lengths, mel_padded, output_lengths, embeddings, f0_padded]


def batch_to_gpu(batch, device):
    (text_padded, input_lengths, mel_padded, gate_padded,
     output_lengths, len_x, audio) = batch
    text_padded = helpers.to_gpu(text_padded).long() if device == 'cuda' else text_padded.long()
    input_lengths = helpers.to_gpu(input_lengths).long() if device == 'cuda' else input_lengths.long()
    max_len = torch.max(input_lengths.data).item() if device == 'cuda' else input_lengths.data.long()
    mel_padded = helpers.to_gpu(mel_padded).float() if device == 'cuda' else mel_padded.float()
    gate_padded = helpers.to_gpu(gate_padded).float() if device == 'cuda' else gate_padded.float()
    output_lengths = helpers.to_gpu(output_lengths).long() if device == 'cuda' else output_lengths.long()
    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
    y = (mel_padded, gate_padded)
    len_x = torch.sum(output_lengths)
    return x, y, len_x, audio
