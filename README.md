# Text to music dataset prepare
The implementation for the audio captioning pipeline used in [MuseControlLite](https://github.com/fundwotsai2001/MuseControlLite), our ICML2025 publication. We use [Jamendo](https://github.com/MTG/mtg-jamendo-dataset) and [FMA](https://github.com/mdeff/fma) as exmaple, but the pipeline should be suitable for other datasets. The key feature is that we use PANNs to filter audio containing vocals and employ Qwen2-Audio-7B-Instruct for captioning.


Follow the instructions below:
1. Clone the github repo, and install the required packages:
```
git clone https://github.com/fundwotsai2001/Text-to-music-dataset-preparation.git
cd text-to-music-dataset-preparation/
conda create -n data_prep_mcl python=3.11
conda activate data_prep_mcl
pip install -r requirements.txt
```

2. Download the full-length version of Jamendo or FMA via:
```
# For Jamendo
mkdir mtg-full
python3 ./mtg-jamendo-dataset/scripts/download/download.py --dataset raw_30s --type audio ./mtg-full --unpack --remove
# For FMA
wget https://os.unil.cloud.switch.ch/fma/fma_full.zip
```
`./mtg-full` is the output directory.

3. Resample the audio to 44100 hz, and slice it to the required shape. For 47s -> (2, 2097152), for 30s -> (2, 1323000). This is optional if you are not using MuseControlLite. You can modify the audio shape with your preference.
```
python slice_2_47s.py --input_folder ./mtg-full --output_folder ./mtg-full-47s
python slice_2_30s.py --input_folder /home/kidrm2/workspace/braintwin/data/spotify_sleep_dataset/sleep_only --output_folder /home/kidrm2/workspace/braintwin/data/spotify_sleep_dataset/sleep_only_30s
```

4. Use sound event detection to filter out audio that contains vocal.
```
python ./panns_inference/filter_vocal.py --audio_folder ./mtg-full-47s --json_path ./filtered_vocal_all.json
python ./panns_inference/filter_vocal.py --audio_folder /home/kidrm2/workspace/braintwin/data/spotify_sleep_dataset/sleep_only_30s --json_path ./filtered_vocal_all.json
```
`--json_path` is the output json file that contains audio paths that do not contain vocal.

5. Use `Qwen/Qwen2-Audio-7B-Instruct` to obtain all captions for the audio.
```
python Qwen2audio_captioning.py --input_json ./filtered_vocal_all.json --output_json ./filtered_vocal_all_caption.json
```
6. Optional, if you are evaluating with the [Song Describer Dataset](https://github.com/mulab-mir/song-describer-dataset), or benchmarking with [MuseControlLite](https://github.com/fundwotsai2001/MuseControlLite), you should filter the no-singing segment of [Song Describer Dataset](https://github.com/mulab-mir/song-describer-dataset) from the the Jamendo dataset.
```
python ./filter_SDD_from_jamendo.py
```

