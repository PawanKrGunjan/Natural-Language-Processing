let microphone = document.getElementById('microphone');
let statusDiv = document.getElementById('status');
let responseAudio = document.getElementById('response-audio');
let userText = document.getElementById('userText');
let responseText = document.getElementById('responseText');

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

microphone.addEventListener('click', () => {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
    isRecording = !isRecording;
});

function startRecording() {
    statusDiv.textContent = "Recording...";
    audioChunks = [];

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            mediaRecorder.addEventListener('dataavailable', event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener('stop', () => {
                let audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                let formData = new FormData();
                formData.append('audio_data', audioBlob);

                fetch('/process_audio', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    let text = data.text;
                    userText.textContent = text;
                    statusDiv.textContent = "Generating response...";

                    return fetch('/get_response', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text: text })
                    });
                })
                .then(response => response.json())
                .then(data => {
                    let responseTextContent = data.response_text;
                    responseText.textContent = responseTextContent;
                    statusDiv.textContent = "Response received.";

                    return fetch(`/get_audio/${encodeURIComponent(responseTextContent)}`);
                })
                .then(response => {
                    if (response.ok) {
                        return response.blob();
                    } else {
                        throw new Error('Network response was not ok');
                    }
                })
                .then(blob => {
                    let audioUrl = URL.createObjectURL(blob);
                    responseAudio.src = audioUrl;
                    responseAudio.style.display = 'block';
                    responseAudio.play();
                })
                .catch(error => {
                    statusDiv.textContent = "Error occurred.";
                });
            });

            statusDiv.textContent = "Recording...";
        })
        .catch(error => {
            statusDiv.textContent = "Error accessing microphone.";
        });
}

function stopRecording() {
    statusDiv.textContent = "Recording stopped.";
    mediaRecorder.stop();
}
