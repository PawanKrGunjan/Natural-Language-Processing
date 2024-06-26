import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self, save_directory):
        self.save_directory = save_directory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def generate_translation(self, tokenizer, model, input_text):
        input_tokens = tokenizer(input_text, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(self.device)
        input_ids = input_tokens['input_ids'].to(self.device)
        attention_mask = input_tokens['attention_mask'].to(self.device)
        
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128, num_beams=4, early_stopping=True)
        
        translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return translation
    
    def translate(self, input_language, text_to_translate):
        model_dir = os.path.join(self.save_directory, input_language)
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
        
        return self.generate_translation(tokenizer, model, text_to_translate)

