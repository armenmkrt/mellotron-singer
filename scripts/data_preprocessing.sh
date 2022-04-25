#!/bin/bash

# Install `shyaml`
pip install shyaml

CONFIG_PATH=configs/preprocessor.yaml
TRAIN_CONFIG=configs/train.yaml
DATASET_PATH=$(cat $CONFIG_PATH | shyaml get-value path.output_path) # path.output_path in preprocessor.yaml

#echo "Prepare dataset for alignment"
#python -m data_preprocessing.prepare_for_alignment --config $CONFIG_PATH --normalise --run_vad --start_end_trim
#printf '=%.0s' {1..100}

#echo "Download g2p model"
#mfa models download g2p english_us_arpa # or `mfa download g2p english_g2p`
#mfa models download acoustic english_us_arpa # or `mfa download acoustic english`
#printf '=%.0s' {1..100}

echo "Create dictionary for dataset"
mfa g2p english_g2p "${DATASET_PATH}"/mfa_folder "${DATASET_PATH}"/lexicon_.txt -j 16 --clean --overwrite
printf '=%.0s' {1..100}

echo "Align dataset"
mfa align "${DATASET_PATH}"/mfa_folder "${DATASET_PATH}"/lexicon_.txt english "${DATASET_PATH}"/TextGrid --clean -v 4 --overwrite -j 16
printf '=%.0s' {1..100}

echo "Run data preprocessor to extract mels, duration, (pitch, energy optional)"
python -m data_preprocessing.preprocessor --config $CONFIG_PATH
printf '=%.0s' {1..100}

echo "Compute speaker embeddings"
python -m data_preprocessing.speaker_embedding.inference --wav_folder "${DATASET_PATH}"/mfa_folder --output_path "$DATASET_PATH" --multi_speaker --model_path models/embedding.pt
printf '=%.0s' {1..100}

echo "Synthesizer training"
python -m train --hparams="$TRAIN_CONFIG"
printf '=%.0s' {1..100}