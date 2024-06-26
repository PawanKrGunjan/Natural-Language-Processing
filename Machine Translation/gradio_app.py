import gradio as gr
import os
from Hindi_English_Translations import Translator

class GradioInterface:
    def __init__(self, translator):
        self.translator = translator
    
    def setup_interface(self):
        def translate_with_direction(text_to_translate, direction):
            input_language = 'English-Hindi' if direction == 'English to Hindi' else 'Hindi-English'
            return self.translator.translate(input_language, text_to_translate)

        # Define Gradio interface
        inputs = [
            gr.Textbox(lines=3, label="Enter text to translate"),
            gr.Radio(['English to Hindi', 'Hindi to English'], label="Translation Direction")
        ]

        outputs = gr.Textbox(label="Translated text")

        interface = gr.Interface(fn=translate_with_direction, 
                                inputs=inputs, 
                                outputs=outputs, 
                                title="Hindi-To-English & English-To-Hindi Translator", 
                                description="Translate text between English and Hindi")
        return interface
    
    def run_interface(self):
        interface =self.setup_interface()
        print("Starting Gradio interface...")
        interface.launch(debug = True, server_port=7000)
        print("Gradio interface started.")

def main():
    save_directory = os.path.join(os.getcwd(), 'saved_model/')
    translator = Translator(save_directory)
    gradio_interface = GradioInterface(translator)
    
    print("Startinng Gradio interface")
    gradio_interface.run_interface()

if __name__ == "__main__":
    main()