from flask import Flask, render_template, request, jsonify
from Hindi_English_Translations import Translator
import os

app = Flask(__name__)

# Initialize the translator
save_directory = os.path.join(os.getcwd(), 'saved_model/')
translator = Translator(save_directory)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text_to_translate = data['text']
    direction = data['direction']
    input_language = 'English-Hindi' if direction == 'English to Hindi' else 'Hindi-English'
    translated_text = translator.translate(input_language, text_to_translate)
    return jsonify({'translated_text': translated_text})

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True, port=5000)
