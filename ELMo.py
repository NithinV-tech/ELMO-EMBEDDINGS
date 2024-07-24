import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from nltk.tokenize import word_tokenize
import string
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
from torch.optim import Adam
from tqdm import tqdm 


class NewsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        return torch.tensor(seq, dtype=torch.long), label

class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_embeddings=None, dropout=0.5):
        super(ELMo, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_embeddings, dtype=torch.float32), freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding_projection = nn.Linear(embedding_dim, hidden_dim)
        self.forward_lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.forward_lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.backward_lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.backward_lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_forward = nn.Linear(hidden_dim, vocab_size)
        self.output_backward = nn.Linear(hidden_dim, vocab_size)
        self.lambdas = nn.Parameter(torch.rand(3), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.combined_output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def freeze_parameters_except_lambdas(self):
        for name, param in self.named_parameters():
            if 'lambdas' not in name:
                param.requires_grad = False

    def forward(self, x):
        embeddings = self.embedding(x)
        embeddings = self.embedding_projection(self.dropout(embeddings))
        forward_out1, _ = self.forward_lstm1(embeddings)
        forward_out2, _ = self.forward_lstm2(forward_out1)
        
        backward_x = torch.flip(x, [1])
        backward_embeddings = self.embedding(backward_x)
        backward_embeddings = self.embedding_projection(self.dropout(backward_embeddings))
        backward_out1, _ = self.backward_lstm1(backward_embeddings)
        backward_out2, _ = self.backward_lstm2(backward_out1)
        backward_out1 = torch.flip(backward_out1, [1])
        backward_out2 = torch.flip(backward_out2, [1])
        
        forward_predictions = self.output_forward(forward_out2)
        backward_predictions = self.output_backward(backward_out2)
        
        combined_layer1 = torch.cat((forward_out1, backward_out1), dim=-1)
        combined_layer2 = torch.cat((forward_out2, backward_out2), dim=-1)
        combined_layer1 = self.combined_output_projection(combined_layer1)
        combined_layer2 = self.combined_output_projection(combined_layer2)

        combined_embeddings = self.gamma * (
            self.lambdas[0] * embeddings +
            self.lambdas[1] * combined_layer1 +
            self.lambdas[2] * combined_layer2
        )

        return forward_predictions, backward_predictions, combined_embeddings



    
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]   
    return words

def load_glove_embeddings(path, vocab, embedding_dim):
    embeddings = np.random.randn(len(vocab), embedding_dim) * 0.01
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[vocab[word]] = vector
    return embeddings


def build_vocab(texts):
    all_tokens = [token for text in texts for token in text]
    token_freqs = Counter(all_tokens)
    sorted_tokens = sorted(token_freqs.items(), key=lambda x: (-x[1], x[0]))
    vocab = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}
    return vocab

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences_padded, labels

def text_to_sequence(text, vocab):
    return [vocab[token] for token in text if token in vocab]


#################################DRIVER CODE################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dataset_path = '/kaggle/input/datasets-custom/train.csv'
df = pd.read_csv(dataset_path)
df['processed_description'] = df['Description'].apply(preprocess_text)
vocab = build_vocab(df['processed_description'])
label_encoder = LabelEncoder()
df['Class Index'] = label_encoder.fit_transform(df['Class Index'])
df['sequence'] = df['processed_description'].apply(lambda x: text_to_sequence(x, vocab))
dataset = NewsDataset(df['sequence'].tolist(), df['Class Index'].tolist())
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
glove_embeddings = load_glove_embeddings('/kaggle/input/gloveem/glove.6B.100d.txt', vocab, embedding_dim=100)


#####################################PARAMETERS#############################################################3
vocab_size = len(vocab)  
embedding_dim = 100
hidden_dim = 256 
#############################################################################################################


################################## PRETRAINING AND SAVING THE MODEL########################################################
model = ELMo(vocab_size, embedding_dim, hidden_dim, pretrained_embeddings=glove_embeddings,dropout=0.5)
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    for sequences, labels in progress_bar:
        sequences = sequences.to(device)
        optimizer.zero_grad()      
        forward_pred, backward_pred, _ = model(sequences)
        forward_target = sequences[:, 1:].to(device) 
        backward_target = torch.flip(sequences, [1])[:, 1:].to(device)
        forward_loss = criterion(forward_pred[:, :-1].reshape(-1, vocab_size), forward_target.reshape(-1))
        backward_loss = criterion(backward_pred[:, :-1].reshape(-1, vocab_size), backward_target.reshape(-1))       
        loss = forward_loss + backward_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': total_loss / len(progress_bar)})

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")
    torch.save(model.state_dict(), f'elmo_pretrained_epoch{epoch + 1}.pth')

print("Training complete! Model saved.")
