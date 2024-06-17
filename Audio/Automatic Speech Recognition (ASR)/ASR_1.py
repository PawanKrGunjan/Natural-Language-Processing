import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from glob import glob
import random

# Define the Transformer Input Layer
class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = nn.Embedding(num_vocab, num_hid)
        self.pos_emb = nn.Embedding(maxlen, num_hid)

    def forward(self, x):
        maxlen = x.size(1)
        x = self.emb(x)
        positions = torch.arange(0, maxlen, device=x.device).unsqueeze(0).expand_as(x[:, :, 0])
        positions = self.pos_emb(positions)
        return x + positions

class SpeechFeatureEmbedding(nn.Module):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = nn.Conv1d(1, num_hid, 11, stride=2, padding=5)
        self.conv2 = nn.Conv1d(num_hid, num_hid, 11, stride=2, padding=5)
        self.conv3 = nn.Conv1d(num_hid, num_hid, 11, stride=2, padding=5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x.transpose(1, 2)

# Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs, training=False):
        attn_output, _ = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

# Transformer Decoder Layer
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.self_att = nn.MultiheadAttention(embed_dim, num_heads)
        self.enc_att = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_dropout = nn.Dropout(0.5)
        self.enc_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(feed_forward_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim)
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        mask = torch.tril(torch.ones((n_dest, n_src), dtype=dtype))
        return mask.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, enc_out, target):
        batch_size, seq_len = target.size(0), target.size(1)
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, dtype=torch.bool).to(target.device)
        target_att, _ = self.self_att(target, target, target, attn_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out, _ = self.enc_att(target_norm, enc_out, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm

# Complete the Transformer model
class Transformer(nn.Module):
    def __init__(self, num_hid=64, num_head=2, num_feed_forward=128, source_maxlen=100, target_maxlen=100,
                 num_layers_enc=4, num_layers_dec=1, num_classes=10):
        super().__init__()
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid)

        self.encoder = nn.ModuleList(
            [self.enc_input] + [TransformerEncoder(num_hid, num_head, num_feed_forward) for _ in range(num_layers_enc)]
        )

        self.decoder = nn.ModuleList(
            [TransformerDecoder(num_hid, num_head, num_feed_forward) for _ in range(num_layers_dec)]
        )

        self.classifier = nn.Linear(num_hid, num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = self.decoder[i](enc_out, y)
        return y

    def forward(self, source, target):
        x = source.transpose(1, 2)
        for i in range(self.num_layers_enc):
            x = self.encoder[i](x)
        y = self.decode(x, target)
        return self.classifier(y)

# Download the dataset
import requests
import tarfile

url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
r = requests.get(url, allow_redirects=True)
open('data.tar.bz2', 'wb').write(r.content)
tar = tarfile.open('data.tar.bz2', "r:bz2")
tar.extractall()
tar.close()

saveto = "./LJSpeech-1.1"
wavs = glob("{}/**/*.wav".format(saveto), recursive=True)

id_to_text = {}
with open(os.path.join(saveto, "metadata.csv"), encoding="utf-8") as f:
    for line in f:
        id = line.strip().split("|")[0]
        text = line.strip().split("|")[2]
        id_to_text[id] = text

def get_data(wavs, id_to_text, maxlen=50):
    """returns mapping of audio paths and transcription texts"""
    data = []
    for w in wavs:
        id = os.path.basename(w).split(".")[0]
        if len(id_to_text[id]) < maxlen:
            data.append({"audio": w, "text": id_to_text[id]})
    return data

# Preprocess the dataset
class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab

max_target_len = 200  # all transcripts in our data are < 200 characters
data = get_data(wavs, id_to_text, max_target_len)
vectorizer = VectorizeChar(max_target_len)
print("vocab size", len(vectorizer.get_vocabulary()))

class TextDataset(Dataset):
    def __init__(self, data, vectorizer):
        self.data = data
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        vectorized_text = self.vectorizer(text)
        return torch.tensor(vectorized_text)

class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data[idx]["audio"]
        waveform, sample_rate = torchaudio.load(audio_path)
        spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
        return spectrogram.squeeze(0)

def create_datasets(data, vectorizer, bs=4, split_ratio=0.8):
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_audio_ds = AudioDataset(train_data)
    train_text_ds = TextDataset(train_data, vectorizer)
    val_audio_ds = AudioDataset(val_data)
    val_text_ds = TextDataset(val_data, vectorizer)

    train_audio_dl = DataLoader(train_audio_ds, batch_size=bs, shuffle=True, collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True))
    train_text_dl = DataLoader(train_text_ds, batch_size=bs, shuffle=True, collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True))
    val_audio_dl = DataLoader(val_audio_ds, batch_size=bs, shuffle=False, collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True))
    val_text_dl = DataLoader(val_text_ds, batch_size=bs, shuffle=False, collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True))

    return train_audio_dl, train_text_dl, val_audio_dl, val_text_dl

# Create datasets and dataloaders
batch_size = 4
train_audio_dl, train_text_dl, val_audio_dl, val_text_dl = create_datasets(data, vectorizer, bs=batch_size)

# Define training loop
def train_model(model, train_audio_dl, train_text_dl, val_audio_dl, val_text_dl, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (audio, text) in zip(train_audio_dl, train_text_dl):
            audio = audio.to(device)
            text = text.to(device)

            optimizer.zero_grad()
            outputs = model(audio, text[:, :-1])
            loss = criterion(outputs.transpose(1, 2), text[:, 1:])
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_audio_dl)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (audio, text) in zip(val_audio_dl, val_text_dl):
                audio = audio.to(device)
                text = text.to(device)

                outputs = model(audio, text[:, :-1])
                loss = criterion(outputs.transpose(1, 2), text[:, 1:])
                val_loss += loss.item()

        val_loss /= len(val_audio_dl)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Set device and initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = Transformer().to(device)

# Train the model
train_model(model, train_audio_dl, train_text_dl, val_audio_dl, val_text_dl, epochs=10, lr=1e-4)

