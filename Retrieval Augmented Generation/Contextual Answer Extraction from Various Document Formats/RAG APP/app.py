from flask import Flask, render_template, request, jsonify
import os
from RetrievalAugmentedGeneration import RetrievalAugmentedGeneration

app = Flask(__name__)

# Example usage
prompt_template = (
    "You are an assistant for finding the exact or similar descriptions to questions using the provided context. "
    "If the question is not related to the provided context, please specify. "
    "Please provide further details or related references for your question. "
    "Keep the answer concise."
    "\n\n"
    "{context}"
)

# Initialize the retrieval-augmented generation instance
rag = RetrievalAugmentedGeneration(secrets_file='secrets.json', prompt_template=prompt_template)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_text', methods=['POST'])
def extract_text():
    files = request.files.getlist("files[]")
    
    file_paths = []
    messages = []
    for file in files:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        file_paths.append(file_path)
        # Load documents using the RAG instance
        status = rag.load_documents([file_path])
        messages.append(f"Text extraction from {file.filename} completed.")
    
    return jsonify({'messages': messages})

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    print(data)
    if data:
        question = data.get('query', '')
        print(question)
        response = rag.question_answering(question)
        
        context_data = []
        for ctx in response['context']:
            context_data.append({
                'Type': ctx.type,
                'Source': ctx.metadata['source'],
                'Page': ctx.metadata['page'],
                'Content': ctx.page_content
            })

        return jsonify({
            'ANSWER': response['answer'],
            'contexts': context_data
        })
    else:
        return jsonify({'error': 'No JSON data received'}), 400


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
