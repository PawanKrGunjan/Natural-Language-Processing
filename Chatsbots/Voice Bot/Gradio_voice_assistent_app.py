import os
import json
import re
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from langchain import ConversationChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import gradio as gr

class VoiceChatbot:
    def __init__(self, secrets_file):
        self.keys = self.load_secret_keys(secrets_file)
        self.setup_environment()
        self.recognizer = sr.Recognizer()
        self.llm = ChatGroq(model="llama3-8b-8192")
        self.prompt_template = self.create_prompt_template()
        self.chatbot_chain = self.create_chatbot_chain()

    def load_secret_keys(self, file_path):
        with open(file_path, 'r') as file:
            secret_keys = json.load(file)
        return secret_keys

    def setup_environment(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = self.keys['Langchain Smith']
        os.environ["GROQ_API_KEY"] = self.keys['Groq']

    def create_prompt_template(self):
        template = """Voice Assistant is a based on large language model.

                    It is designed to be able to assist with a wide range of tasks, from answering simple questions to providing accurate explanations on a wide range of topics. 
                    In the case, if you don't know, answer that currently I don't know I will update it soon.
                    
                    As a language model, it should be able to generate human-like text based on the input it receives, 
                    allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

                    Assistant is aware that human input is being transcribed from audio and as such there may be some errors in the transcription. It will attempt to account for some words being swapped with similar-sounding words or phrases. 
                    Assistant must be accurate, concise and not more that 5 sentences, because human attention spans are more limited over the audio channel since it takes time to listen to a response.

                    {history}
                    Human: {input}
                    AI:
                    """

        return PromptTemplate(input_variables=["history", "human_input"], template=template)

    def create_chatbot_chain(self):
        chatbot_chain = ConversationChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=False,
            memory=ConversationBufferWindowMemory(k=1),
        )
        return chatbot_chain

    def speak_text(self, text):
        tts = gTTS(text=text, lang='en')
        file_path="response.mp3"
        tts.save(file_path)
        return file_path

    def process_audio(self, audio):
        audio_data = sr.AudioFile(audio)
        with audio_data as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio_data = self.recognizer.record(source)
        user_input = self.recognizer.recognize_google(audio_data, language="en-IN")
        if user_input is None:
            user_input="Sorry, I did not get that. Can you please ask again?"
            path = self.speak_text(user_input)
            assistant_response=None
            return user_input, path, assistant_response

        else:
            print('Querying ...')
            assistant_response = self.chatbot_chain.predict(input=user_input)
            clean_text = re.sub(r'[!*#]', '', assistant_response)
            audio_response_filepath = self.speak_text(clean_text)
            return user_input, audio_response_filepath, assistant_response


def run_gradio():
    chatbot = VoiceChatbot('secrets.json')
    interface = gr.Interface(
        fn=chatbot.process_audio,
        inputs=gr.Audio(sources=["microphone"], type="filepath", label="Speak Here"),
        outputs=[
            gr.Textbox(label="You Said:", placeholder="Recording..."),
            gr.Audio(type='filepath', label="Generated Audio", autoplay=True),
            gr.Textbox(label="Assistant's Response:", placeholder="Waiting for response..."),
        ],
        live=True,
        title="Voice Assistant",
        description="Speak into the microphone and interact with the Voice Assistant.",
        theme="compact",
    )
    interface.launch(inbrowser=True)

if __name__ == "__main__":
    run_gradio()