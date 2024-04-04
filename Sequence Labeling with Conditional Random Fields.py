# I'm starting off by importing all the necessary libraries and modules. I need PyTorch for the neural network,
# Adam optimizer for training, CRF from TorchCRF for sequence tagging, and some utilities for data handling.
import torch
import torch.nn as nn
from torch.optim import Adam
from TorchCRF import CRF
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Here, I define some constants for the dimensions of my embeddings, the number of unique tags, and the size of my vocabulary.
EMBEDDING_DIM = 5
NUM_TAGS = 3
VOCAB_SIZE = 14

# I'm creating a custom dataset class for Named Entity Recognition (NER). This class will handle the data I'm working with,
# converting sentences and their tags into tensors for the model.
class NERDataset(Dataset):
    def __init__(self, sentences, tags):
        self.sentences = [torch.LongTensor(s) for s in sentences]
        self.tags = [torch.LongTensor(t) for t in tags]

    # This method returns the size of my dataset.
    def __len__(self):
        return len(self.sentences)

    # This method retrieves a single data point from my dataset.
    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]

# Now, I define the architecture of my neural network model. It's a simple model with embedding, LSTM, and a linear layer,
# followed by a CRF layer for sequence tagging.
class Net(nn.Module):
    def __init__(self, embedding_dim, num_tags):
        super().__init__()
        # I initialize an embedding layer, an LSTM for sequence modeling, a CRF layer for sequence tagging,
        # and a fully connected layer to map LSTM outputs to tag space.
        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, bidirectional=True)
        self.crf = CRF(num_tags + 2)
        self.fc = nn.Linear(embedding_dim * 2, num_tags + 2)

    # This is where the forward pass of my model happens.
    def forward(self, sentences, tags=None):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.fc(lstm_out)
        # I use a mask for the CRF layer to handle variable-length sequences.
        mask = torch.ones_like(sentences, dtype=torch.bool)
        if tags is not None:
            # If tags are provided, calculate the loss.
            loss = self.crf(lstm_feats, tags, mask=mask)
            return loss.mean()
        else:
            # Otherwise, return the model's predictions.
            output = self.crf.decode(lstm_feats, mask=mask)
            return output

# This function will help in padding the sequences to have the same length for batch processing.
def pad_collate(batch):
    (sentences, tags) = zip(*batch)
    sentences_pad = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_pad = pad_sequence(tags, batch_first=True, padding_value=0)
    return sentences_pad, tags_pad

# Here's how I train the model. I loop over the dataset and update the model's weights using backpropagation.
def train_model(model, train_loader, optimizer):
    for epoch in range(10):  # I'll train for 10 epochs.
        for sentences, tags in train_loader:
            optimizer.zero_grad()  # Reset gradients.
            loss = model(sentences, tags)  # Compute loss.
            loss.backward()  # Backpropagation.
            optimizer.step()  # Update model parameters.

# And this function is for testing the model. It prints out predictions for each sentence in the dataset.
def test_model(model, train_loader):
    tag_dict = {0: "MIDDLE NAME", 1: "FIRST NAME", 2: "LAST NAME", 3: "NICK NAME"}
    word_dict = {1: "John", 2: "Doe", 3: "Sara", 4: "Smith", 5: "Ali", 6: "Khan"}

    with torch.no_grad():  # No need to track gradients here.
        for sentences, _ in train_loader:
            predictions = model(sentences)
            # I'm printing out the predictions along with their corresponding words.
            for sentence, prediction in zip(sentences, predictions):
                print(" ".join([f"{word_dict[int(word)]}:{tag_dict[int(tag)]}" for word, tag in zip(sentence, prediction)]))

# Finally, this is the main function where everything comes together.
def main():
    sentences = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 2]]
    tags = [[0, 1, 1, 2, 1], [0, 1, 1, 3, 3], [2, 1, 1, 1, 1]]
    train_dataset = NERDataset(sentences, tags)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    model = Net(EMBEDDING_DIM, NUM_TAGS)
    optimizer = Adam(model.parameters())
    train_model(model, train_loader, optimizer)
    test_model(model, train_loader)

# I kick everything off here.
if __name__ == "__main__":
    main()
