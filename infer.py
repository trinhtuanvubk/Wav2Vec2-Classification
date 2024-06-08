import torch
import torchaudio
import numpy as np
import transformers
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from train import Wav2Vec2ClasificationTrainer
from model import Wav2Vec2ForSpeechClassification
import librosa
import argparse

def infer(model_path = "/content/Wav2Vec2-Classification/checkpoints/checkpoint-40",
          audio_filepath = "data/sounds/burger/burger_1_01.wav"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = AutoConfig.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_path).to(device)

    # id2label = Wav2Vec2ClasificationTrainer("data/train.csv", "data/train.csv").config.id2label
    id2label = config.id2label
    print(id2label)

    speech_array, sampling_rate = torchaudio.load(audio_filepath)
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, feature_extractor.sampling_rate)

    features = feature_extractor(speech_array, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits
    # print(logits)
    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    # print(pred_ids)
    print(id2label[pred_ids[0]])


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Wav2Vec2 Classification Trainer')
    # Add arguments
    parser.add_argument('--model_path', type=str, default="/content/Wav2Vec2-Classification/checkpoints/checkpoint-40")
    parser.add_argument('--audio_filepath', type=str, default="data/sounds/burger/burger_1_01.wav")
    # Parse the arguments
    args = parser.parse_args()
    
    infer(args.model_path, args.audio_filepath)