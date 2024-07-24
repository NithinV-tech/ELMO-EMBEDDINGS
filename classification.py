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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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



###########################################DOWN STREAM CLASS#######################################################
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim , output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        logits = self.fc(last_output)
        return logits
##########################################################################################################################


##############################TRAINING CLASSIFIER WITH DATASET################################################################
def train_classification(model, classifier, data_loader, epochs=5):
    model.train() 
    classifier.train() 
    model.freeze_parameters_except_lambdas()  #--->freezed
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)  #->classifier weights changed
    elmo_lambda_optimizer = torch.optim.Adam([model.lambdas], lr=0.001)         #->elmo only lambda is changing
    criterion = nn.CrossEntropyLoss()   
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for sequences, labels in progress_bar:
            sequences, labels = sequences.to(device), labels.to(device)          
            classifier_optimizer.zero_grad()
            elmo_lambda_optimizer.zero_grad()
            _,_,embeddings = model(sequences)
            predictions = classifier(embeddings)
            loss = criterion(predictions, labels)
            loss.backward()
            classifier_optimizer.step()
            elmo_lambda_optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': total_loss / len(progress_bar)})
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")
    torch.save(classifier.state_dict(), 'lstm_classifier.pth')


    ################################ TRAIN CLASSIFIER WITH FROZEN ELMO############################################################

def train_classification_with_frozen_elmo(model, classifier, data_loader, epochs=5):

        model.eval() 
        classifier.train() 
        for param in model.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
            for sequences, labels in progress_bar:
                sequences, labels = sequences.to(device), labels.to(device)
                optimizer.zero_grad()
                _, _, embeddings = model(sequences)
                predictions = classifier(embeddings)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({'Loss': total_loss / len(progress_bar)})
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")
        torch.save(classifier.state_dict(), 'frozen_lstm_classifier.pth')

################################################EVALUTION PART##############################################################################
def evaluate(model, classifier, data_loader, is_test=False):
    model.eval()
    classifier.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            _, _, embeddings = model(sequences)
            predictions = classifier(embeddings)
            _, predicted = torch.max(predictions, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_predictions))
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    if is_test:
        plt.title("Confusion Matrix for Test Data")
    else:
        plt.title("Confusion Matrix for Train Data")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2", "Class 3"], yticklabels=["Class 0", "Class 1", "Class 2", "Class 3"])
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()

##############################################EVALUATE FROZEN LAMBDA##############################################3

def evaluate2(model, classifier, data_loader, is_test=False):
    model.eval()
    classifier.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            _, _, embeddings = model(sequences)
            predictions = classifier(embeddings)
            _, predicted = torch.max(predictions, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_predictions))
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    if is_test:
        plt.title("Confusion Matrix for Test Data(FROZEN ELMO)")
    else:
        plt.title("Confusion Matrix for Train Data(FROZEN ELMO)")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2", "Class 3"], yticklabels=["Class 0", "Class 1", "Class 2", "Class 3"])
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()

################################################## LEARNABLE ELMO - FUNCTION######################################################## 
class LearnableELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_embeddings=None, dropout=0.5):
        super(LearnableELMo, self).__init__()
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

        self.learnable_function = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True) 
        )

        self.combined_output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def freeze_parameters_except_learnable_function(self):    
        for param in self.parameters():
            param.requires_grad = False
        for param in self.learnable_function.parameters():
            param.requires_grad = True
    

    def forward(self, x, backward_x=None):
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
        forward_predictions = self.output_forward(forward_out2[:, -1, :])  
        backward_predictions = self.output_backward(backward_out2[:, -1, :]) 
        combined_layer1 = torch.cat((forward_out1, backward_out1), dim=-1)
        combined_layer2 = torch.cat((forward_out2, backward_out2), dim=-1)
        combined_layer1 = self.combined_output_projection(combined_layer1)
        combined_layer2 = self.combined_output_projection(combined_layer2)
        combined_embeddings = self.learnable_function(torch.cat((embeddings, combined_layer1, combined_layer2), dim=-1))
        return forward_predictions, backward_predictions, combined_embeddings


##########################################LEARNABLE FUNCTION CLASSIFIER####################################################################################
def train_classification_learnable(model, classifier, data_loader, epochs=5):
    model.train()  
    classifier.train()
    model.freeze_parameters_except_learnable_function()
    optimizer = torch.optim.Adam(model.learnable_function.parameters(), lr=0.001)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for sequences, labels in progress_bar:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            _, _, embeddings = model(sequences)
            predictions = classifier(embeddings)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            classifier_optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': total_loss / len(progress_bar)})
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")
    torch.save(classifier.state_dict(), 'lstm_classifier_lear.pth')


#################################DRIVER CODE#############################################################################3

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
test_df = pd.read_csv('/kaggle/input/datasets-custom/test.csv') 
test_df['processed_description'] = test_df['Description'].apply(preprocess_text)
test_df['Class Index'] = label_encoder.transform(test_df['Class Index'])
test_df['sequence'] = test_df['processed_description'].apply(lambda x: text_to_sequence(x, vocab))
test_dataset = NewsDataset(test_df['sequence'].tolist(), test_df['Class Index'].tolist())
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

#####################################DRIVER PART - 2#############################################################3

output_dim = 4 
hidden_dim = 256  
embedding_dim = 100
vocab_size = len(vocab)  
classifier = LSTMClassifier(input_dim=hidden_dim, hidden_dim=256, output_dim=output_dim).to(device)
saved_model_path = '/kaggle/working/elmo_pretrained_epoch5.pth'
model_t = ELMo(vocab_size, embedding_dim, hidden_dim, pretrained_embeddings=glove_embeddings, dropout=0.5)
model_t.load_state_dict(torch.load(saved_model_path))
model_t.to(device)
model_t.eval()
train_classification(model_t, classifier, data_loader, epochs=5)
loaded_classifier = LSTMClassifier(input_dim=hidden_dim, hidden_dim=256, output_dim=output_dim).to(device)
loaded_classifier.load_state_dict(torch.load('lstm_classifier.pth'))
evaluate(model_t, loaded_classifier, test_data_loader, is_test=True)
evaluate(model_t, loaded_classifier, data_loader, is_test=False)

######################################FROZEN DRIVER#######################################################################
saved_model_path = '/kaggle/working/elmo_pretrained_epoch5.pth'
model_t2 = ELMo(vocab_size, embedding_dim, hidden_dim, pretrained_embeddings=glove_embeddings, dropout=0.5)
model_t2.load_state_dict(torch.load(saved_model_path))
model_t2.to(device)
model_t2.eval()
frozen_classifier = LSTMClassifier(input_dim=hidden_dim, hidden_dim=256, output_dim=output_dim).to(device)
frozen_classifier.load_state_dict(torch.load('frozen_lstm_classifier.pth'))
evaluate2(model_t2, frozen_classifier, test_data_loader, is_test=True)
evaluate2(model_t2, frozen_classifier, data_loader, is_test=False)
########################################LEARNABLE ELMO DRIVER######################################################################################

model_learnable = LearnableELMo(vocab_size, embedding_dim, hidden_dim, pretrained_embeddings=glove_embeddings,dropout=0.5)
model_learnable.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
num_epochs = 5
optimizer = optim.Adam(model_learnable.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model_learnable.train()
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    for sequences, labels in progress_bar:
        sequences = sequences.to(device)
        optimizer.zero_grad()       
        forward_pred, backward_pred, _ = model_learnable(sequences, torch.flip(sequences, [1]))
        target = sequences[:, -1].to(device)  
        backward_target = torch.flip(sequences, [1])[:, -1].to(device)
        forward_pred = forward_pred.reshape(-1, vocab_size) 
        backward_pred = backward_pred.reshape(-1, vocab_size) 
        target = target.reshape(-1) 
        backward_target = backward_target.reshape(-1) 
        forward_loss = criterion(forward_pred, target)
        backward_loss = criterion(backward_pred, backward_target)
        loss = forward_loss + backward_loss
        loss.backward()
        optimizer.step()        
        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': total_loss / len(progress_bar)})

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")
    torch.save(model_learnable.state_dict(), f'elmo_pretrained_learnable_latest_epoch{epoch + 1}.pth')

print("Training complete! Model saved.")

saved_model_path_learnable = '/kaggle/working/elmo_pretrained_learnable_latest_epoch5.pth'
model_t3 = LearnableELMo(vocab_size, embedding_dim, hidden_dim, pretrained_embeddings=glove_embeddings, dropout=0.5)
model_t3.load_state_dict(torch.load(saved_model_path_learnable))
model_t3.to(device)
classifier_learnable = LSTMClassifier(input_dim=hidden_dim, hidden_dim=256, output_dim=output_dim).to(device)
train_classification_learnable(model_t3, classifier_learnable, data_loader, epochs=5)
classifier_learnable.load_state_dict(torch.load('lstm_classifier_lear.pth'))
evaluate(model_t3, classifier_learnable, test_data_loader, is_test=True)
evaluate(model_t3, classifier_learnable, data_loader, is_test=False)

########################################################################################################################