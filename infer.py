

import transformers
from transformers import AutoConfig, Wav2Vec2FeatureExtractor

from model import Wav2Vec2ForSpeechClassification


model_name_or_path = "m3hrdadfi/wav2vec2-base-100k-eating-sound-collection"
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)




speech_array, sampling_rate = torchaudio.load("data/sounds/burger/burger_1_01.wav")
speech_array = speech_array.squeeze().numpy()
speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, feature_extractor.sampling_rate)

features = feature_extractor(speech_array, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)

input_values = features.input_values.to(device)

with torch.no_grad():
    logits = model(input_values).logits

pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
print(pred_ids)
