from flask import Flask, render_template, request, jsonify, send_file
import Voice_assistent 
import os

app = Flask(__name__)
# Initialize your VoiceChatbot instance with appropriate credentials
chatbot = Voice_assistent.VoiceChatbot('secrets.json')  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files['audio_data']
    print(audio_file)
    if audio_file:
        audio_path = os.path.join('uploads', 'user_audio.wav')
        audio_file.save(audio_path)
        # Process the audio file
        text = chatbot.convert_voice_to_text(audio_path)
        print(f"User said: {text}")

        return jsonify({'text': text})

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    text = data.get('text', '')
    response_text = chatbot.get_response(text)
    print(f"Chatbot response: {response_text}")

    return jsonify({
        'response_text': response_text,
    })

@app.route('/get_audio/<filename>', methods=['GET'])
def get_audio(filename):
    audio_response_path = chatbot.speak_text(filename, Play=False)
    print(f'AUDIO RESPONSE PATH :{audio_response_path}')
    
    if os.path.exists(audio_response_path):
        return send_file(audio_response_path, mimetype='audio/wav')
    else:
        return jsonify({'error': 'Audio file not found'}), 404

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=False)
