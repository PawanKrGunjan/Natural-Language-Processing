import os
import json
import speech_recognition as sr
from gtts import gTTS
import re
from playsound import playsound
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from IPython.display import Audio, display
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from transformers import pipeline
import torch
import time
from pydub import AudioSegment
from pydub.playback import play

class VoiceChatbot:
    def __init__(self, secrets_file):
        """
        Initialize the VoiceChatbot with necessary configurations and models.
        """
        self.keys = self.load_secret_keys(secrets_file)
        self.setup_environment()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.classifier = pipeline("audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=self.device)
        self.llm = ChatGroq(model="llama3-8b-8192")
        self.prompt_template = self.create_prompt_template()
        self.chat_history = ChatMessageHistory()
        self.chatbot_chain = self.create_chatbot_chain()

    def load_secret_keys(self, file_path):
        """
        Load secret keys from a JSON file.
        """
        with open(file_path, 'r') as file:
            secret_keys = json.load(file)
        return secret_keys

    def setup_environment(self):
        """
        Set up the environment variables required for Langchain and Groq.
        """
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = self.keys['Langchain Smith']
        os.environ["GROQ_API_KEY"] = self.keys['Groq']

    def create_prompt_template(self):
        """
        Create the prompt template for the chatbot.
        """
        template = """Your are a Voice Assistant bot based on a large language model. 
                      Your name is Gunjan.
                      You are designed to assist with a wide range of tasks, from answering simple questions to providing accurate explanations on various topics. 
                      If you don't know the answer, respond with "I don't know, I will update it soon."
                      You have to behave like the human whose name is Gunjan.
                      allowing it to engage in natural-sounding conversations and provide coherent and relevant responses.
                      The assistant is aware that human input is transcribed from audio, which may contain errors. It will attempt to account for similar-sounding words or phrases. 
                      Responses must be accurate, concise, and no more than 5 sentences, considering human attention spans are limited over audio as listening takes time.
                    """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                MessagesPlaceholder(variable_name="messages"),
            ])
        return prompt

    def create_chatbot_chain(self):
        # Update the history with the current question and get the response
        question_answer_chain = self.create_prompt_template() | self.llm        
        return question_answer_chain

    def convert_to_wav(self, file):
        audio = AudioSegment.from_file(file)
        converted_file = f"{file.rsplit('.', 1)[0]}_converted.wav"
        audio.export(converted_file, format='wav')
        return converted_file    

    def convert_voice_to_text(self, file, chunk_length_s=5.0):
        converted_file = self.convert_to_wav(file)
        with sr.AudioFile(converted_file) as source:
            audio = self.recognizer.record(source)
        try:
            voice_text = self.recognizer.recognize_google(audio, language="en-IN")
            print(f"You said: {voice_text}")
        except sr.UnknownValueError:
            voice_text = None
            print("Sorry, I did not get that. Can you please ask again?")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")
        return voice_text

    def get_voice_input(self, file_path='recorded_audio.wav'):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening ...")
            audio = self.recognizer.listen(source)
        try:
            with open(file_path, 'wb') as f:
                f.write(audio.get_wav_data())
            print(f"Audio saved as {file_path}")
            return file_path
        except sr.UnknownValueError:
            print("Sorry, I did not get that. Can you please ask again?")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")    
            return None

    def speak_text(self, text, speed=1.0, Play=True):
        tts = gTTS(text=text, lang='en')
        filename = "templates/result.mp3"
        audio_response_filepath = os.path.join(os.getcwd(), filename)
        tts.save(audio_response_filepath)
        display(Audio(audio_response_filepath, rate=16000))
        if Play:
            audio = AudioSegment.from_file(audio_response_filepath)
            def change_playback_speed(sound, speed=1.0):
                sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
                    "frame_rate": int(sound.frame_rate * speed)
                })
                return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)
            speed_up_audio = change_playback_speed(audio, speed)
            play(speed_up_audio)
        return audio_response_filepath

    def excite_fn(self, excite_word, debug=False):
        prob_threshold = 0.5
        sampling_rate = self.classifier.feature_extractor.sampling_rate
        time.sleep(3)
        mic = ffmpeg_microphone_live(sampling_rate=sampling_rate, chunk_length_s=0.2, stream_chunk_s=0.25)
        self.speak_text("Speak 'STOP' to stop the conversation, just after it.", speed=1.3)
        count = 0
        for prediction in self.classifier(mic):
            prediction = prediction[0]
            if debug:
                print(prediction)
            elif prediction["label"] == excite_word and prediction["score"] > prob_threshold:
                return 'stop'
            count += 1
            if count == 50:
                return
                
    def get_response(self, user_input):
        if user_input is not None:
            self.chat_history.add_user_message(user_input)
            response=self.chatbot_chain.invoke({"messages": self.chat_history.messages})
            self.chat_history.add_ai_message(response.content)
            return response.content
        else:
            return 'No User Input'

    def run(self):
        self.speak_text('Hi, I am your intelligent voice assistant. How can I help you?', speed=1.1)
        i = 0
        while True and i < 3:
            path = self.get_voice_input()
            user_input = self.convert_voice_to_text(path)
            if user_input is None:
                self.speak_text("Sorry, I did not get that. Can you please ask again?")
                i += 1
            else:
                print('Querying ...')
                response = self.get_response(user_input)
                print(f"Response: {response}")
                clean_text = re.sub(r'[!*#]', '', response)
                self.speak_text(clean_text)
                if self.excite_fn(excite_word='stop') == "stop":
                    self.speak_text("Thanks, it was great assisting you.")
                    return "Thanks, it was great assisting you!"
                i = 0

if __name__ == "__main__":
    chatbot = VoiceChatbot('secrets.json')
    chatbot.run()