document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const statusMessages = document.getElementById('status-messages');
    const queryInput = document.getElementById('query-input');
    const submitBtn = document.getElementById('submit-btn');
    const answerContent = document.getElementById('answer-content');
    const contextContent = document.getElementById('context-content');

    // Upload button click event
    uploadBtn.addEventListener('click', () => {
        const files = fileInput.files;
        const formData = new FormData();

        for (let i = 0; i < files.length; i++) {
            formData.append('files[]', files[i]);
        }

        fetch('/extract_text', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            statusMessages.textContent = ''; // Clear previous status messages
            data.messages.forEach(message => {
                const p = document.createElement('p');
                p.textContent = message;
                statusMessages.appendChild(p);
            });
        })
        .catch(error => {
            console.error('Error:', error);
            const p = document.createElement('p');
            p.textContent = 'An error occurred during file upload.';
            statusMessages.appendChild(p);
        });
    });

    // Submit button click event
    submitBtn.addEventListener('click', () => {
        const question = queryInput.value;
        getResponse(question);
    });

    // Function to handle question answering
    const getResponse = (question) => {
        fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: question })
        })
        .then(response => response.json())
        .then(data => {
            answerContent.textContent = data.ANSWER;

            // Clear previous context content
            contextContent.innerHTML = '';

            // Display context information
            data.contexts.forEach(context => {
                const contextItem = document.createElement('div');
                contextItem.className = 'context-item';

                contextItem.innerHTML = `
                    <p><strong>Type:</strong> ${context.Type}</p>
                    <p><strong>Source:</strong> ${context.Source}</p>
                    <p><strong>Page:</strong> ${context.Page}</p>
                    <p><strong>Content:</strong></p>
                    <p>${context.Content}</p>
                `;

                contextContent.appendChild(contextItem);
            });
        })
        .catch(error => {
            console.error('Error:', error);
            const p = document.createElement('p');
            p.textContent = 'An error occurred while fetching the response.';
            answerContent.appendChild(p);
        });
    };
});
