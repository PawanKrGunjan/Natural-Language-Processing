import transformers
import torch
import gradio as gr

# Define model and tokenizer loading and translation function as before
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-en-hi"
save_directory = 'D:/Natural-Language-Processing/Machine Translation/machine-translation-app/saved_model/'

def generate_translation(tokenizer, model, input_text, device):
    input_tokens = tokenizer(input_text, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(device)
    input_ids = input_tokens['input_ids'].to(device).to(device)
    attention_mask = input_tokens['attention_mask'].to(device).to(device)
    
    generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128, num_beams=4, early_stopping=True)
    
    translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return translation

def translate(input_language, text_to_translate):
    if input_language == 'En':
        model_dir = save_directory + 'opus-mt-en-hi'
    else:
        model_dir = save_directory + 'opus-mt-hi-en'
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return generate_translation(tokenizer, model, text_to_translate, device)

# Define Gradio interface with language selection
inputs = [
    gr.Textbox(lines=3, label="Enter text to translate"),
    gr.Radio(['English to Hindi', 'Hindi to English'], label="Translation Direction")
]

outputs = gr.Textbox(label="Translated text")

def translate_with_direction(text_to_translate, direction):
    if direction == 'English to Hindi':
        translated_text = translate('En', text_to_translate)
    else:  # 'Hindi to English'
        translated_text = translate('Hi', text_to_translate)
    return translated_text

gr.Interface(fn=translate_with_direction, inputs=inputs, outputs=outputs, title="Language Translator", description="Translate text between English and Hindi").launch()
