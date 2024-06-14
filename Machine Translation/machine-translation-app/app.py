import transformers
import torch
import gradio as gr
import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base directory
if os.getenv("DOCKER_ENV") == "true":
    # Path within the Docker container
    save_directory = '/app/saved_model/'
else:
    # Local development path
    save_directory = 'D:/Natural-Language-Processing/Machine Translation/machine-translation-app/saved_model/'

logging.debug(f"Model save directory: {save_directory}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models and tokenizers
logging.debug("Loading English model and tokenizer...")
model_dir = os.path.join(save_directory, 'opus-mt-en-hi')
english_tokenizer = AutoTokenizer.from_pretrained(model_dir)
english_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
logging.debug("English model and tokenizer loaded successfully.")

logging.debug("Loading Hindi model and tokenizer...")
hindi_tokenizer = AutoTokenizer.from_pretrained(os.path.join(save_directory, 'opus-mt-hi-en'))
hindi_model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(save_directory, 'opus-mt-hi-en')).to(device)
logging.debug("Hindi model and tokenizer loaded successfully.")

def generate_translation(tokenizer, model, input_text, device):
    logging.debug(f"Generating translation for input: {input_text}")
    input_tokens = tokenizer(input_text, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(device)
    input_ids = input_tokens['input_ids'].to(device).to(device)
    attention_mask = input_tokens['attention_mask'].to(device).to(device)

    logging.debug("Generating model output...")
    generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128, num_beams=4, early_stopping=True)

    translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    logging.debug(f"Generated translation: {translation}")
    return translation

def translate(input_language, text_to_translate):
    logging.debug(f"Translating text: {text_to_translate} from {input_language}")
    if input_language == 'En':
        return generate_translation(english_tokenizer, english_model, text_to_translate, device)
    else:
        return generate_translation(hindi_tokenizer, hindi_model, text_to_translate, device)

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

def main():
    interface = gr.Interface(fn=translate_with_direction, 
                            inputs=inputs, 
                            outputs=outputs, 
                            title="Language Translator",
                            description="Translate text between English and Hindi")

    print("Starting Gradio interface...")
    # Use this config when running on Docker
    interface.launch(server_name="0.0.0.0", server_port=7000)
    print("Gradio interface started.")

if __name__ == "__main__":
    main()


