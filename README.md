# VAEGP

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class ConvVAE(nn.Module):
    def __init__(self, input_channels, hidden_dim1, hidden_dim2, latent_dim, input_length):
        super(ConvVAE, self).__init__()
        self.encoder_conv1 = nn.Conv1d(input_channels, hidden_dim1, kernel_size=4, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv1d(hidden_dim1, hidden_dim2, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(hidden_dim2 * (input_length // 4), latent_dim)
        self.fc2 = nn.Linear(hidden_dim2 * (input_length // 4), latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim2 * (input_length // 4))
        self.decoder_conv1 = nn.ConvTranspose1d(hidden_dim2, hidden_dim1, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.decoder_conv2 = nn.ConvTranspose1d(hidden_dim1, input_channels, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.domain_classifier = nn.Linear(latent_dim, 1)

    def encode(self, x):
        h1 = torch.relu(self.encoder_conv1(x))
        h2 = torch.relu(self.encoder_conv2(h1))
        h2_flat = h2.view(h2.size(0), -1)
        z_mean = self.fc1(h2_flat)
        z_log_var = self.fc2(h2_flat)
        return z_mean, z_log_var

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        h3_reshaped = h3.view(h3.size(0), hidden_dim2, input_length // 4)
        h4 = torch.relu(self.decoder_conv1(h3_reshaped))
        x_reconstructed = torch.sigmoid(self.decoder_conv2(h4))
        return x_reconstructed

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x, alpha):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decode(z)
        reverse_features = GradientReversalLayer.apply(z, alpha)
        domain_prediction = torch.sigmoid(self.domain_classifier(reverse_features))
        return x_reconstructed, z_mean, z_log_var, domain_prediction

def loss_function(x, x_reconstructed, z_mean, z_log_var, domain_labels, domain_prediction):
    BCE = nn.functional.mse_loss(x_reconstructed, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    domain_loss = nn.functional.binary_cross_entropy(domain_prediction, domain_labels)
    return BCE + KLD + domain_loss

def train_conv_vae_with_domain_classifier(model, source_loader, target_loader, epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (source_data, _), (target_data, _) in zip(source_loader, target_loader):
            source_data = source_data.to(device)
            target_data = target_data.to(device)
            domain_labels = torch.cat([torch.ones(source_data.size(0)), torch.zeros(target_data.size(0))], dim=0).to(device)

            optimizer.zero_grad()
            x_reconstructed, z_mean, z_log_var, domain_prediction = model(torch.cat([source_data, target_data], dim=0), alpha)
            loss = loss_function(torch.cat([source_data, target_data], dim=0), x_reconstructed, z_mean, z_log_var, domain_labels, domain_prediction)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / (len(source_loader.dataset) + len(target_loader.dataset))
        train_losses.append(average_loss)
        print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")

    return model

# Example usage
input_channels = 1
input_length = X_comb_np.shape[1]
hidden_dim1 = 32
hidden_dim2 = 64
latent_dim = 10
learning_rate = 0.001
batch_size = 64
epochs = 100

# Prepare data loaders
X_comb = torch.tensor(X_comb_np, dtype=torch.float32).unsqueeze(1)
y_comb = torch.tensor(y_comb_np, dtype=torch.float32)
dataset = TensorDataset(X_comb, y_comb)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize and train the model
model = ConvVAE(input_channels, hidden_dim1, hidden_dim2, latent_dim, input_length)
trained_model = train_conv_vae_with_domain_classifier(model, dataloader, dataloader, epochs, learning_rate)

# Extract latent features and use for downstream tasks
latent_features_np_train = extract_latent_features(trained_model, X_comb)
# Continue with downstream tasks, e.g., training a GPR

