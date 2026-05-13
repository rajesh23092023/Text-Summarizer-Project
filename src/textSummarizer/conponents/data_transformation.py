import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        # Tokenize the input dialogue
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
        
        # Modern way: Tokenize the summary using the text_target parameter
        # This replaces the old 'with self.tokenizer.as_target_tokenizer()' block
        target_encodings = self.tokenizer(text_target=example_batch['summary'], max_length=128, truncation=True)
            
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    
    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        
        # Mapping the conversion function across the dataset
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        
        # Saving the processed dataset to the artifacts folder
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))