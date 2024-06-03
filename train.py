import os

from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor

from dataloader import dataloader, DataCollatorCTCWithPadding
from utils import speech_file_to_array_fn, label_to_id, preprocess_function
from metric import compute_metrics
from model import Wav2Vec2ClassificationHead, Wav2Vec2ForSpeechClassification
from transformers import TrainingArguments, Trainer

class Wav2Vec2ClasificationTrainer:
    def __init__(self,
                 train_csv_path,
                 eval_csv_path,
                 model_name="facebook/wav2vec2-base-100k-voxpopuli",
                 pooling_mode="mean",
                 is_regression=False
                 ):
        # csv dataset
        train_dataset, eval_dataset = dataloader(train_csv_path, eval_csv_path)
        
        # option for testing
        max_samples = 4
        train_dataset = train_dataset.select(range(max_samples))
        eval_dataset = eval_dataset.select(range(max_samples))
        
        # label
        self.label_list = train_dataset.unique("label")
        self.label_list.sort()
        
        print(f"num labels: {len(self.label_list)}")
        
        # feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        target_sampling_rate = self.feature_extractor.sampling_rate
        print(f"The target sampling rate: {target_sampling_rate}")
        
        # dataset mapping
        self.train_dataset = train_dataset.map(
            lambda x: preprocess_function(x, self.feature_extractor, target_sampling_rate, self.label_list),
            batch_size=16,
            batched=True,
            num_proc=4
        )
        self.eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(x, self.feature_extractor, target_sampling_rate, self.label_list),
            batch_size=16,
            batched=True,
            num_proc=4
        )
        
        
        
        # data collator
        self.data_collator = DataCollatorCTCWithPadding(feature_extractor=self.feature_extractor, padding=True)
        
        
        # model config
        self.config = AutoConfig.from_pretrained(
                        model_name,
                        num_labels=len(self.label_list),
                        label2id={label: i for i, label in enumerate(self.label_list)},
                        id2label={i: label for i, label in enumerate(self.label_list)},
                        finetuning_task="wav2vec2_clf",
                    )
        setattr(self.config, 'pooling_mode', pooling_mode)
        
        
        # model
        self.model = Wav2Vec2ForSpeechClassification.from_pretrained(
                        model_name,
                        config=self.config,
                    )
        
        self.model.freeze_feature_extractor()
        
        os.makedirs("checkpoints", exist_ok=True)
        self.training_args = TrainingArguments(
                output_dir="checkpoints/",
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=2,
                num_train_epochs=1,
                fp16=True,
                save_steps=10,
                eval_steps=10,
                logging_steps=10,
                learning_rate=1e-4,
                save_total_limit=2,
            )
        
        self.trainer = Trainer(
                model=self.model,
                data_collator=self.data_collator,
                args=self.training_args,
                compute_metrics=compute_metrics,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.feature_extractor,
            )
                                        
    def train(self):
        self.trainer.train()
    
if __name__=="__main__":
    train_csv_path = "data/train.csv"
    eval_csv_path = "data/test.csv"
    trainer = Wav2Vec2ClasificationTrainer(train_csv_path, eval_csv_path)
    trainer.train()
    