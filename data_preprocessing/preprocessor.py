import argparse
import json
import os

import torch
import librosa
import numpy as np
import pyworld as pw
import tgt
from omegaconf import OmegaConf
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data import audio as Audio


class Preprocessor:
    def __init__(self, config, compute_pitch: bool, compute_energy: bool):
        self.config = config
        self.corpus_dir = config["path"]["dataset_path"]
        self.in_dir = config["path"]["mfa_folder"]
        self.out_dir = config["path"]["output_path"]
        self.val_ratio = config["preprocessing"]["val_ratio"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in ["phoneme_level", "frame_level"]
        assert config["preprocessing"]["energy"]["feature"] in ["phoneme_level", "frame_level"]
        self.pitch_phoneme_averaging = (config["preprocessing"]["pitch"]["feature"] == "phoneme_level")
        self.energy_phoneme_averaging = (config["preprocessing"]["energy"]["feature"] == "phoneme_level")

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        text_config = config['preprocessing']['text']
        self.ipa = True if (('ipa' in text_config) and text_config['ipa']) else False

        self.STFT = Audio.stft.TacotronSTFT(
            filter_length=config["preprocessing"]["stft"]["filter_length"],
            hop_length=config["preprocessing"]["stft"]["hop_length"],
            win_length=config["preprocessing"]["stft"]["win_length"],
            n_mel_channels=config["preprocessing"]["mel"]["n_mel_channels"],
            sampling_rate=config["preprocessing"]["audio"]["sampling_rate"],
            mel_fmin=config["preprocessing"]["mel"]["mel_fmin"],
            mel_fmax=config["preprocessing"]["mel"]["mel_fmax"],
        )
        self.error_count = 0
        self.error_names = []
        self.compute_pitch = compute_pitch
        self.compute_energy = compute_energy

    def to_file(self, lines, filename):
        with open(os.path.join(self.out_dir, filename), "w", encoding="utf-8") as f:
            for m in lines:
                f.write(m + "\n")

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        if self.compute_pitch:
            os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        if self.compute_energy:
            os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration_in_sec")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "f0s")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = dict()
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir), position=0)):
            speakers[speaker] = i
            for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker)), position=1, leave=False):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".wav")[0]
                tg_dir = os.path.join(self.out_dir, "TextGrid", speaker)

                tg_path = os.path.join(tg_dir, f"{basename}.TextGrid")
                if not os.path.isfile(tg_path):
                    tg_path = os.path.join(tg_dir, f"{basename.replace('_', '-')}.TextGrid")
                    if not os.path.isfile(tg_path):
                        tg_path = os.path.join(tg_dir, f"{speaker}-{basename}.TextGrid")
                        if not os.path.isfile(tg_path):
                            tg_path = os.path.join(tg_dir, f"{speaker}-{basename.replace('_', '-')}.TextGrid")

                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker=speaker, basename=basename, tg_path=tg_path)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)
                else:
                    print(f'{tg_path} missing')
                    continue

                if pitch is not None and self.compute_pitch:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if energy is not None and self.compute_energy:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print(f'{self.error_count} errors encountered')
        print(self.error_names)

        print("Computing statistic quantities ...")
        stats = dict()
        # Perform normalization if necessary
        if self.compute_pitch:
            if self.pitch_normalization:
                pitch_mean = pitch_scaler.mean_[0]
                pitch_std = pitch_scaler.scale_[0]
            else:
                # A numerical trick to avoid normalization...
                pitch_mean = 0
                pitch_std = 1
            pitch_min, pitch_max = self.normalize(os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std)
            stats["pitch"] = [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)]

        if self.compute_energy:
            if self.energy_normalization:
                energy_mean = energy_scaler.mean_[0]
                energy_std = energy_scaler.scale_[0]
            else:
                energy_mean = 0
                energy_std = 1
            energy_min, energy_max = self.normalize(os.path.join(self.out_dir, "energy"), energy_mean, energy_std)
            stats['energy'] = [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)]

        if stats:
            with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
                json.dump(stats, f)

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump(speakers, f)

        print(f"Total time: {n_frames * self.hop_length / self.sampling_rate / 3600} hours")

        out = [row for row in out if row is not None]
        train_names_path = os.path.join(self.corpus_dir, 'train.txt')
        val_names_path = os.path.join(self.corpus_dir, 'val.txt')
        train, val, test = [], [], []
        if os.path.isfile(val_names_path) and os.path.isfile(train_names_path):  # if split is provided
            print('Split is provided!')
            with open(train_names_path) as infile:
                train_names = set(infile.read().split('\n'))
            with open(val_names_path) as infile:
                val_names = set(infile.read().split('\n'))
            for row in out:
                if row.split('|')[0] in val_names:
                    val.append(row)
                elif row.split('|')[0] in train_names:
                    train.append(row)
                else:  # if test set is separated
                    test.append(row)
        else:
            print('Random split!')
            np.random.seed(17 if 'seed' not in self.config else self.config['seed'])
            indices = np.arange(len(out))
            val_indices = set(list(np.random.choice(indices, size=int(len(indices) * self.val_ratio), replace=False)))
            train = [out[index] for index in indices if index not in val_indices]
            val = [out[index] for index in val_indices]

        # Write metadata
        self.to_file(train, "train.txt")
        self.to_file(val, "val.txt")
        if test:
            self.to_file(test, "test.txt")

        # Calculating mean and std for train
        duration_statistics_dict = self.calculate_duration_stats(train)
        with open(os.path.join(self.out_dir, "duration_stats.json"), 'w') as f:
            json.dump(duration_statistics_dict, f)

        return out

    def calculate_duration_stats(self, metadata):
        durations_list = list()
        for metadata_line in metadata:
            name = metadata_line.split("|")[0]
            duration_in_secs_path = os.path.join(self.out_dir, 'duration_in_sec', name + '.npy')
            durations_list.append(np.load(duration_in_secs_path))
        durations = np.concatenate(durations_list)

        stat_dict = dict()
        stat_dict["durations_mean"] = durations.mean()
        stat_dict["durations_std"] = durations.std()
        return stat_dict

    def process_utterance(self, speaker, basename, tg_path):
        try:
            wav_path = os.path.join(self.in_dir, speaker, f"{basename}.wav")
            text_path = os.path.join(self.in_dir, speaker, f"{basename}.lab")

            # Get alignments
            textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
            phone, duration, duration_sec, start, end = self.get_alignment(textgrid.get_tier_by_name("phones"))
            text = " ".join(phone)
            if start >= end:
                return None

            # Read and trim wav files
            wav, _ = librosa.load(wav_path)
            wav = wav[int(self.sampling_rate * start): int(self.sampling_rate * end)].astype(np.float32)

            # Read raw text
            with open(text_path, "r") as f:
                raw_text = f.readline().strip("\n")

            if self.compute_pitch:
                pitch = self.calculate_pitch(wav=wav, duration=duration)
                np.save(os.path.join(self.out_dir, "pitch", f"{basename}.npy"), pitch)
            else:
                pitch = None

            # Compute mel-scale spectrogram and energy
            mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
            mel_spectrogram = mel_spectrogram[:, :sum(duration)]
            energy = energy[:sum(duration)]

            if self.energy_phoneme_averaging and self.compute_energy:
                # Phoneme-level average
                pos = 0
                for i, d in enumerate(duration):
                    if d > 0:
                        energy[i] = np.mean(energy[pos:pos + d])
                    else:
                        energy[i] = 0
                    pos += d
                energy = energy[:len(duration)]

            # Compute f0s
            f0 = self.get_f0(audio=wav, sampling_rate=self.sampling_rate,
                             frame_length=self.config["preprocessing"]["stft"]["filter_length"],
                             hop_length=self.hop_length, f0_min=100,
                             f0_max=500, harm_thresh=0.1)
            f0 = torch.from_numpy(f0)[None]
            f0 = f0[:, :mel_spectrogram.shape[1]]

            # Save files
            np.save(os.path.join(self.out_dir, "duration", f"{basename}.npy"), duration)
            np.save(os.path.join(self.out_dir, "duration_in_sec", f"{basename}.npy"), duration_sec)
            np.save(os.path.join(self.out_dir, "mel", f"{basename}.npy"), mel_spectrogram.T)
            np.save(os.path.join(self.out_dir, "f0s", f"{basename}.npy"), f0)

            if self.compute_energy:
                np.save(os.path.join(self.out_dir, "energy", f"{basename}.npy"), energy)

            return "|".join([basename, text, raw_text]), pitch, energy, mel_spectrogram.shape[1]
        except BaseException as e:
            print(f'Exception {e}')
            self.error_count += 1
            self.error_names.append(basename)
            return None

    def calculate_pitch(self, wav, duration):
        # Compute fundamental frequency
        pitch, t = pw.dio(wav.astype(np.float64), self.sampling_rate,
                          frame_period=self.hop_length / self.sampling_rate * 1000)
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[:sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(nonzero_ids, pitch[nonzero_ids],
                                 fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]), bounds_error=False, )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos:pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[:len(duration)]
        return pitch

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations, durations_sec = [], []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            if p == '':
                p = 'sp'
                # if e - s > 0.5:
                #     p = 'sil'
                # else:
                #     p = 'sp'

            # Trim leading silences
            if not phones:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(int(np.round(e * self.sampling_rate / self.hop_length)
                                 - np.round(s * self.sampling_rate / self.hop_length)))
            durations_sec.append(e - s)

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        durations_sec = durations_sec[:end_idx]

        return phones, durations, durations_sec, start_time, end_time

    @staticmethod
    def normalize(in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

    @staticmethod
    def get_f0(audio, sampling_rate=22050, frame_length=1024,
               hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
        f0, harmonic_rates, argmins, times = compute_yin(
            audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
            harm_thresh)
        pad = int((frame_length / hop_length) / 2)
        f0 = [0.0] * pad + f0 + [0.0] * pad

        f0 = np.array(f0, dtype=np.float32)
        return f0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config .yaml file")
    parser.add_argument("--compute_pitch", default=False, action="store_true")
    parser.add_argument("--compute_energy", default=False, action="store_true")

    args = parser.parse_args()
    preprocessor = Preprocessor(config=OmegaConf.load(args.config),
                                compute_pitch=args.compute_pitch,
                                compute_energy=args.compute_energy)
    preprocessor.build_from_path()
