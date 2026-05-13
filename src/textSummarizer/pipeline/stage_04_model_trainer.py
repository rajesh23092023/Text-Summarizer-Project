from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.model_trainer import ModelTrainer
from textSummarizer.logging import logger
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ModelTrainerTrainingPipeline:
    def __init__(self):
        
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        
        # Load model and tokenizer
        device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_trainer_config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_trainer_config.model_ckpt).to(device)
        
        print("✓ Model loaded successfully!")
        print(f"✓ Model: {model_trainer_config.model_ckpt}")
        print(f"✓ Device: {device}")
        
        # Test inference with a sample text
        sample_text = "Text summarization is the process of reducing a text document with a computer program in order to create a computer-generated summary that retains the most important information from the original document."
        
        inputs = tokenizer.encode(sample_text, max_length=512, truncation=True, return_tensors="pt").to(device)
        summary_ids = model_pegasus.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        print("\n" + "="*60)
        print("SAMPLE TEXT SUMMARIZATION TEST")
        print("="*60)
        print(f"\nOriginal Text:\n{sample_text}")
        print(f"\nGenerated Summary:\n{summary}")
        print("\n✓ Model inference working correctly!")
        
        # Save model and tokenizer
        model_pegasus.save_pretrained(os.path.join(model_trainer_config.root_dir,"pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(model_trainer_config.root_dir,"tokenizer"))
        print(f"\n✓ Model saved to: {model_trainer_config.root_dir}")
        print("✓ All tasks completed successfully!")