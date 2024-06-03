
import torchaudio


def speech_file_to_array_fn(path, target_sampling_rate):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples, feature_extractor, target_sampling_rate, label_list):
    speech_list = [speech_file_to_array_fn(path, target_sampling_rate) for path in examples["path"]]
    target_list = [label_to_id(label, label_list) for label in examples["label"]]

    result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)

    return result