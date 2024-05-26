#!/bin/bash

python3 create_indexes.py create_indexes --waveforms_hdf5_path="./dataset/audioset/indexes/balanced_train.h5"

# Unbalanced training indexes
for IDX in {00..40}; do
    echo $IDX
    python3 create_indexes.py create_indexes --waveforms_hdf5_path="./dataset/audioset/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5" --indexes_hdf5_path="./dataset/audioset/indexes/unbalanced_train/unbalanced_train_part$IDX.h5"
done

# Combine balanced and unbalanced training indexes to a full training indexes hdf5
python3 create_indexes.py combine_full_indexes --indexes_hdf5s_dir="./dataset/audioset/indexes" --full_indexes_hdf5_path="./dataset/audioset/indexes/full_train.h5"
