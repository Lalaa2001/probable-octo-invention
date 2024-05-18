import torch
import torch.nn as nn
import numpy as np
import random

# Define the text corpus
corpus = "Hello world. This is a simple example of text generation using a character-level RNN."

# Create a set of all unique characters in the corpus
chars = sorted(list(set(corpus)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# Hyperparameters
input_size = len(chars)
hidden_size = 128
output_size = len(chars)
num_layers = 1
seq_length = 25
learning_rate = 0.003
num_epochs = 500

# RNN Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# Instantiate the model
model = RNN(input_size, hidden_size, output_size, num_layers)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Prepare input and target data
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = char_to_idx[string[c]]
    return tensor

# Training the model
for epoch in range(num_epochs):
    # Select a random start point for the sequence
    start_idx = random.randint(0, len(corpus) - seq_length - 1)
    input_seq = corpus[start_idx:start_idx + seq_length]
    target_seq = corpus[start_idx + 1:start_idx + seq_length + 1]

    input_tensor = char_tensor(input_seq).unsqueeze(0)
    target_tensor = char_tensor(target_seq)

    # One-hot encoding
    input_tensor = nn.functional.one_hot(input_tensor, num_classes=input_size).float()

    # Initialize hidden state
    hidden = model.init_hidden(1)

    # Forward pass
    output, hidden = model(input_tensor, hidden)
    loss = criterion(output, target_tensor.view(-1))

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'char_rnn_model.pth')

# Function to generate text
def generate_text(model, start_str, length):
    model.eval()
    input_seq = char_tensor(start_str).unsqueeze(0)
    input_tensor = nn.functional.one_hot(input_seq, num_classes=input_size).float()
    hidden = model.init_hidden(1)
    generated_text = start_str

    for _ in range(length):
        output, hidden = model(input_tensor, hidden)
        output_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        char = idx_to_char[top_i.item()]
        generated_text += char

        input_tensor = nn.functional.one_hot(char_tensor(char).unsqueeze(0), num_classes=input_size).float()

    return generated_text

# Generate new text
start_str = "Hello"
generated_text = generate_text(model, start_str, 100)
print(f'\nGenerated Text:\n{generated_text}')
