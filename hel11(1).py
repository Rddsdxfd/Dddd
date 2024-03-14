from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CTCLoss
from torchaudio.transforms import MelSpectrogram
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextTransform:
    def __init__(self):
        self.char_map = {'а': 0, 'б': 1, 'в': 2, 'г': 3, 'д': 4, 'е': 5, 'ё': 6, 'ж': 7, 'з': 8, 'и': 9,
                         'й': 10, 'к': 11, 'л': 12, 'м': 13, 'н': 14, 'о': 15, 'п': 16, 'р': 17, 'с': 18,
                         'т': 19, 'у': 20, 'ф': 21, 'х': 22, 'ц': 23, 'ч': 24, 'ш': 25, 'щ': 26, 'ъ': 27,
                         'ы': 28, 'ь': 29, 'э': 30, 'ю': 31, 'я': 32, ' ': 33}

    def text_to_int(self, text):
        return [self.char_map[char] for char in text]

    def int_to_text(self, labels):
        return ''.join([char for label in labels for char, idx in self.char_map.items() if idx == label])

def calculate_seq_len(input_lengths):
    return (input_lengths - 3) // 2 + 1

BATCH_SIZE = 64
SAMPLE_RATE = 16000
HOP_LENGTH = 160
NUM_MELS = 160
EMBEDDING_SIZE = 512
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

DATASET_PATH = '/content/drive/MyDrive/cc/c'
MODEL_SAVE_PATH = '/content/drive/MyDrive/m.pt'

text_transform = TextTransform()

class SpeechDataset(Dataset):
    MAX_INPUT_LENGTH = 160

    def __init__(self, file_paths, transcriptions, text_transform):
        self.file_paths = [path for path in file_paths if os.path.exists(path)]
        self.transcriptions = [transcriptions[i] for i, path in enumerate(file_paths) if os.path.exists(path)]
        self.text_transform = text_transform
        self.mel_spectrogram_transform = MelSpectrogram(SAMPLE_RATE, n_fft=800, hop_length=160, n_mels=160)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        try:
            waveform, _ = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return None

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        feature = self.mel_spectrogram_transform(waveform)

        if feature.shape[-1] > self.MAX_INPUT_LENGTH:
            feature = feature[..., :self.MAX_INPUT_LENGTH]
        elif feature.shape[-1] < self.MAX_INPUT_LENGTH:
            padding = torch.zeros((1, feature.shape[1], self.MAX_INPUT_LENGTH - feature.shape[-1]))
            feature = torch.cat((feature, padding), dim=-1)

        label = torch.Tensor(self.text_transform.text_to_int(self.transcriptions[idx].lower()))
        return feature, label, torch.tensor(len(label))

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    label_lengths = torch.tensor([len(item[1]) for item in batch])

    features = pad_sequence(features, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    input_lengths = torch.full((len(features),), SpeechDataset.MAX_INPUT_LENGTH, dtype=torch.long)

    return features, labels, input_lengths, label_lengths

class SpeechToTextRCNN(nn.Module):
    def __init__(self, input_size=NUM_MELS, hidden_size=EMBEDDING_SIZE, output_size=len(text_transform.char_map) + 1):
        super(SpeechToTextRCNN, self).__init__()

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)  # Adding dropout with probability 0.5

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)  # Applying dropout
        x, _ = self.lstm(x.transpose(1, 2))
        return F.log_softmax(self.fc(x), dim=2)

def prepare_datasets_helper(dataset_path, batch_size=BATCH_SIZE):
    with open(os.path.join(dataset_path, 'transcriptions.txt'), 'r') as f:
        lines = f.readlines()

    file_paths = []
    transcriptions = []
    for line in lines:
        filename, transcription = line.strip().split(maxsplit=1)
        file_paths.append(os.path.join(dataset_path, filename))
        transcriptions.append(transcription)

    print("First train file text: ", transcriptions[0])
    print("First validation file text: ", transcriptions[len(transcriptions)//2])

    train_files, val_files, train_transcriptions, val_transcriptions = train_test_split(
        file_paths, transcriptions, test_size=0.2, random_state=42
    )

    train_dataset = SpeechDataset(train_files, train_transcriptions, text_transform)
    val_dataset = SpeechDataset(val_files, val_transcriptions, text_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, len(text_transform.char_map)

def prepare_datasets(dataset_path, batch_size=BATCH_SIZE):
    train_dataloader, val_dataloader, output_size = prepare_datasets_helper(dataset_path, batch_size)
    return train_dataloader, val_dataloader, output_size

def train_model(model, train_dataloader, val_dataloader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (audio, targets, input_lengths, target_lengths) in enumerate(train_dataloader):
            audio, targets = audio.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(audio)
            outputs = outputs.transpose(0, 1)

            input_lengths = calculate_seq_len(input_lengths).long()

            loss = criterion(outputs, targets.int(), input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (audio, targets, input_lengths, target_lengths) in enumerate(val_dataloader):
                audio, targets = audio.to(device), targets.to(device)
                outputs = model(audio)
                outputs = outputs.transpose(0, 1)

                input_lengths = calculate_seq_len(input_lengths).long()

                loss = criterion(outputs, targets.int(), input_lengths, target_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f'Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            print('Validation Loss Decreased({:.6f}--->{:.6f}) \t Saving The Model'.format(best_val_loss,avg_val_loss))
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_val_loss = avg_val_loss

        scheduler.step(avg_val_loss)

        plt.figure()
        plt.plot(np.arange(1, epoch + 2), train_losses, label='Train Loss')
        plt.plot(np.arange(1, epoch + 2), val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(MODEL_SAVE_PATH.replace('.pt', f'_epoch_{epoch+1}.png'))
        plt.close()

    return model

def infer(model, dataloader, text_transform):
    model = model.eval()
    with torch.no_grad():
        for i, (audio, targets, input_lengths, target_lengths) in enumerate(dataloader):
            audio = audio.to(device)
            outputs = model(audio)
            outputs = outputs.transpose(0, 1)
            predicted_texts = decode_predictions(outputs, text_transform)

            for idx, _ in enumerate(audio):
                print(f"Input Audio File: {dataloader.dataset.file_paths[i * dataloader.batch_size + idx]}")
                print(f"Predicted Text: {predicted_texts[idx]}")

def decode_predictions(outputs, text_transform):
    _, preds = outputs.max(2)
    preds = preds.transpose(0, 1).cpu().numpy()
    decoded_preds = []
    for pred in preds:
        decoded_preds.append(text_transform.int_to_text(pred))
    return decoded_preds

train_dataloader, val_dataloader, output_size = prepare_datasets(DATASET_PATH)
model = SpeechToTextRCNN().to(device)
trained_model = train_model(model, train_dataloader, val_dataloader, num_epochs=NUM_EPOCHS)

print(f"Number of batches in training DataLoader: {len(train_dataloader)}")
print(f"Number of batches in validation DataLoader: {len(val_dataloader)}")

print("Inspecting model outputs during inference:")
infer(trained_model, val_dataloader, text_transform)
                         
