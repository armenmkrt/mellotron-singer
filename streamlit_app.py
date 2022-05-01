import os
import io

import json
import torch
import numpy as np
import streamlit as st
from scipy.io import wavfile
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from data.text import phoneme_duration_to_sequence
from train import load_model


from HiFi_GAN.env import AttrDict
from HiFi_GAN.models import Generator


config_path = "/home/podcastle/Documents/mellotron-singer/configs/train.yaml"
synthesizer_checkpoint_path = "/home/podcastle/Documents/mellotron-singer/models/base_model_lr_scheduling/checkpoint_35000"
vocoder_checkpoint_path = "/home/podcastle/workspace/vc-training-pipeline/models/vocoders/hifi_gan/universal_pretrained/g_02500000"

hparams_ = OmegaConf.load(config_path)
device = torch.device('cpu')


@st.cache
def load_synthesizer():
    synthesizer = load_model(hparams_)
    synthesizer.eval()
    synthesizer.to(device)
    print("Loading checkpoint '{}'".format(synthesizer_checkpoint_path))
    synthesizer_checkpoint_dict = torch.load(synthesizer_checkpoint_path)
    synthesizer.load_state_dict(synthesizer_checkpoint_dict['state_dict'])
    return synthesizer


synthesizer = load_synthesizer()


def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

    return virtualfile


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


config_file = os.path.join("/home/podcastle/Documents/mellotron-singer/HiFi_GAN/UNIVERSAL_V1/config.json")
with open(config_file) as f:
    data = f.read()

global h
json_config = json.loads(data)
h = AttrDict(json_config)


@st.cache
def load_vocoder():
    vocoder = Generator(h).to(device)

    vocoder_checkpoint_dict = load_checkpoint(vocoder_checkpoint_path, device)
    vocoder.load_state_dict(vocoder_checkpoint_dict['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder

vocoder = load_vocoder()

def get_f0_(filename):
    f0 = np.load(filename)[0]

    # Padding zeros in place of sos and sos tokens
    f0 = np.insert(f0, 0, [0, 0])
    f0 = np.append(f0, [0, 0])[None, :]

    f0 = torch.from_numpy(f0)
    return f0


def load_durations_from_numpy(path, in_seconds=False):
    durations = np.load(path)
    if in_seconds:
        durations = np.insert(durations, 0, [0.02])
        durations = np.append(durations, [0.02])
    else:
        durations = np.insert(durations, 0, [2])
        durations = np.append(durations, [2])
    return torch.from_numpy(durations)


def get_embedding(embed_path):
    return torch.from_numpy(np.load(embed_path, allow_pickle=True))


st.write(""" # Melodic speech generation """)


audio_id = "GW8"

durations_in_frames = load_durations_from_numpy(
    f"/home/podcastle/Documents/GW_preprocessed/duration/{audio_id}.npy").unsqueeze(0).long()
f0s = get_f0_(f"/home/podcastle/Documents/GW_preprocessed/f0s/{audio_id}.npy").unsqueeze(1).float()

with open("/home/podcastle/Documents/GW_preprocessed/train.txt", 'r') as f:
    phonemes = [line.strip().split("|")[1] for line in f.readlines() if audio_id in line][0]

with open("/home/podcastle/Documents/GW_preprocessed/train.txt", 'r') as f:
    text = [line.strip().split("|")[2] for line in f.readlines() if audio_id in line][0]


phonemes_sequence = phonemes.split()
durations_list = list(durations_in_frames.squeeze().numpy())

phoneme_duration_pairs = []
for i in range(1, len(durations_list) - 1):
    phoneme_duration_pairs.append(" ".join([phonemes_sequence[i - 1],
                                            str(durations_list[i])]))
phoneme_duration_input = "|".join(phoneme_duration_pairs)


st.write(text)
st.write("Input phoneme sequence")
st.write(phonemes)

st.header("Modify phoneme sequence if needed")
phoneme_duration_input = st.text_area("Phoneme and duration pairs", phoneme_duration_input)

phonemes = list()
for pair in phoneme_duration_input.split("|"):
    phonemes.append(pair.split()[0])
phonemes = " ".join(phonemes)

phoneme_ids = phoneme_duration_to_sequence(phonemes)
phoneme_ids = torch.IntTensor(phoneme_ids).unsqueeze(0)

durations_sum = torch.sum(durations_in_frames)
st.write("Durations sum before editing", durations_sum)


def duations_to_alignment(durations):
    T = torch.sum(durations)
    N = durations.size(1)

    start_index = 0
    alignment = torch.zeros(N, T)

    for i in range(N):
        duration = durations[0][i]
        alignment[i, start_index: start_index + duration] = 1
        start_index += duration

    return alignment


def adjust_durations(phoneme_duration_input):
    edited_durations = [2]
    phoneme_duration_pairs = phoneme_duration_input.split("|")
    for i in range(len(phoneme_duration_pairs)):
        edited_duration = st.sidebar.slider(phoneme_duration_pairs[i].split()[0],
                                            min_value=1,
                                            max_value=60,
                                            value=int(phoneme_duration_pairs[i].split()[1]),
                                            key=i)
        edited_durations.append(edited_duration)
    edited_durations.append(2)
    return torch.from_numpy(np.array(edited_durations)[None, :])


durations_in_frames = adjust_durations(phoneme_duration_input)
st.write("Durations sum after editing", torch.sum(durations_in_frames))

hard_alignment = duations_to_alignment(durations_in_frames)

generate_flag = st.button("Generate")

if generate_flag:
    with torch.no_grad():
        _, mel, _, alignment = synthesizer.semi_inference(phoneme_ids, durations_in_frames, f0s, hard_alignment)

    alignment = alignment.squeeze(-1)

    mel_np = mel.float().data.cpu().numpy()[0]
    alignment_np = alignment.float().data.cpu().numpy()[0].T

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(mel_np, aspect="auto", origin="lower",
                   interpolation='none')

    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")


    st.pyplot(fig)

    with torch.no_grad():
        audio = vocoder(mel)
        audio = audio.squeeze()

    st.audio(create_audio_player(audio.numpy(), sample_rate=22050), format='audio/wav', start_time=0)
    st.write("---")
