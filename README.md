# Wav2Vec2-Classification


### Installation
```bash
conda create -n py38 python=3.8.8
conda activate py38
pip install -r requirements.txt
```

### Data

- Data Tree (each folder name inside sounds is a class): 
```
|___data
    |___sounds
        |___1 folder
        |___ ...
        |___n folder
    |___train.csv
    |___test.csv
```
- To generate `train.csv` and  `test.csv`:
```bash
python3 dataset.py
```

### Training
- To train:
```bash
python3 train.py \
--model_name "facebook/wav2vec2-base-100k-voxpopuli" \
--num_epochs 50 \
--batch_size 32 \
```


### Inference
- To infer:

```bash
python3 infer.py \
--model_path "checkpoints/checkpoint_name
--audio_filepath path/to/audiofile
```