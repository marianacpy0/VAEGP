 self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, encoding_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(X_train, epochs=350, batch_size=128, learning_rate=0.001):
    input_dim = X_train.shape[1]
    encoding_dim = 81

    autoencoder = ComplexAutoencoder(input_dim, encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    autoencoder.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            input_data = data[0].to(device)
            optimizer.zero_grad()
            output_data = autoencoder(input_data)
            loss = criterion(output_data, input_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

    encoder_model = autoencoder.encoder
    return autoencoder, encoder_model

def extract_latent_features(encoder_model, data):
    encoder_model.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        latent_features = encoder_model(data_tensor)
    return latent_features.cpu().numpy()

def train_gpr(X, y):
    kernel = C(1.0, (1e-4, 1e1)) * Matern(length_scale=1.0, nu=1.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.017, n_restarts_optimizer=5, normalize_y=True)
    gpr.fit(X, y)
    return gpr

# Example usage
# X_train = your_training_data
# autoencoder, encoder_model = train_autoencoder(X_train)

# Extract latent features
# latent_features = extract_latent_features(encoder_model, X_train)

# Train GPR on latent features
# gpr = train_gpr(latent_features, y_train)

# Validate the model
df_1_val = pd.read_csv("path/to/val_1.csv")
df_2_val = pd.read_csv("path/to/val_2.csv")
df_val = pd.concat([df_1_val, df_2_val])

# Prepare validation data (assuming prepare_subsets function is defined)
X_val, y_val, sweep_aged_val, water_aged_val, _ = prepare_subsets(df_val, scaler_path=None)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

# Extract latent features from validation data
latent_features_val = extract_latent_features(encoder_model, X_val_tensor)

# Make predictions with GPR
water_pred = gpr.predict(latent_features_val)
df_results = pd.DataFrame({'sweep_new': sweep_aged_val['sweep_new'], 'water_ppm_pred': water_pred})
print(df_results.head())

