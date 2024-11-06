import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/mnt/data/765B922E-69D2-4E7B-8EF2-371C7371D64E.jpeg"
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Convert to grayscale for easier edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Canny edge detection to find edges
edges = cv2.Canny(gray, 50, 150)

# Detect contours based on edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Known measurements for reference lines in mm
reference_measurements = {
    'top_red': 55.58,   # Top red horizontal line
    'blue_vertical': 5.68,  # Blue vertical line
    'purple_vertical': 6.68  # Purple vertical line
}

# Variables to store the pixel lengths of detected reference lines
pixel_lengths = {
    'top_red': None,
    'blue_vertical': None,
    'purple_vertical': None
}

# Filter and measure contours that correspond to the reference lines
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    # Aspect ratio helps identify horizontal vs. vertical lines
    aspect_ratio = w / h
    
    # Identify the top red line based on its horizontal aspect ratio and width
    if aspect_ratio > 5 and 50 < w < width:  # Wide horizontal line
        pixel_lengths['top_red'] = w
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw in red for the top red line

    # Identify the blue line based on its vertical orientation and known height
    elif aspect_ratio < 0.5 and 5 < h < 20:  # Narrow vertical line within expected range
        if pixel_lengths['blue_vertical'] is None:
            pixel_lengths['blue_vertical'] = h
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw in blue for blue line

    # Identify the purple line based on its vertical orientation and known height
    elif aspect_ratio < 0.5 and 5 < h < 20:  # Narrow vertical line within expected range
        if pixel_lengths['purple_vertical'] is None and pixel_lengths['blue_vertical'] is not None:
            pixel_lengths['purple_vertical'] = h
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Draw in purple for purple line

# Calculate pixel-to-mm ratios if all reference lines are detected
if all(pixel_lengths.values()):
    horizontal_ratio = reference_measurements['top_red'] / pixel_lengths['top_red']
    vertical_ratio_blue = reference_measurements['blue_vertical'] / pixel_lengths['blue_vertical']
    vertical_ratio_purple = reference_measurements['purple_vertical'] / pixel_lengths['purple_vertical']
    vertical_ratio = (vertical_ratio_blue + vertical_ratio_purple) / 2  # Average for general vertical ratio

    print(f"Horizontal Pixel-to-MM Ratio: {horizontal_ratio} mm/pixel")
    print(f"Vertical Pixel-to-MM Ratio: {vertical_ratio} mm/pixel")

    # Display the ratios on the image for verification
    cv2.putText(image, f"Horizontal Ratio: {horizontal_ratio:.2f} mm/pixel", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"Vertical Ratio: {vertical_ratio:.2f} mm/pixel", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

else:
    print("Could not detect all reference lines. Adjust detection criteria or check image clarity.")

# Display the image with detected reference lines and ratios
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Reference Lines with Pixel-to-MM Ratios")
plt.show()

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


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/mnt/data/F5A3F673-11D2-4A58-AD2F-F4835C910BAF.jpeg"
image = cv2.imread(image_path)
height, width = image.shape[:2]  # Get the original image dimensions

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for black/grey in HSV
lower_dark = np.array([0, 0, 0])      
upper_dark = np.array([180, 50, 100])  # Adjusted to capture greyish areas

# Create a mask to isolate dark/greyish regions
dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

# Use morphological operations to refine the mask
kernel = np.ones((5, 5), np.uint8)
dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the mask
contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a copy of the original image to draw contours and measurements
contour_image = image.copy()

# Known measurements
top_border_mm = 5.68
bottom_border_mm = 6.68
top_border_pixels = None
bottom_border_pixels = None

# Variables to store contour information for the top and bottom borders
top_border_contour = None
bottom_border_contour = None

# Sort contours by width in descending order
sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[2], reverse=True)

# Identify top and bottom borders based on sorted contours and aspect ratio
for contour in sorted_contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h

    # Check if contour has a high aspect ratio, suggesting a horizontal line
    if aspect_ratio > 5 and w > 50:  # Adjust aspect ratio and width thresholds as needed
        if top_border_pixels is None:
            top_border_pixels = w
            top_border_contour = contour
            cv2.putText(contour_image, "Top Border", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Draw a red line across the top border
            cv2.line(contour_image, (x, y + h // 2), (x + w, y + h // 2), (0, 0, 255), 2)  # Red line for top border
        elif bottom_border_pixels is None:
            bottom_border_pixels = w
            bottom_border_contour = contour
            cv2.putText(contour_image, "Bottom Border", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Draw a green line across the bottom border
            cv2.line(contour_image, (x, y + h // 2), (x + w, y + h // 2), (0, 255, 0), 2)  # Green line for bottom border

    # Stop once top and bottom borders are identified
    if top_border_pixels and bottom_border_pixels:
        break

# Calculate pixel-to-mm ratios if both borders are detected
if top_border_pixels and bottom_border_pixels:
    top_ratio = top_border_mm / top_border_pixels
    bottom_ratio = bottom_border_mm / bottom_border_pixels
    pixel_to_mm_ratio = (top_ratio + bottom_ratio) / 2  # Average ratio
    print(f"Top Border Pixel Length: {top_border_pixels} pixels")
    print(f"Bottom Border Pixel Length: {bottom_border_pixels} pixels")
    print(f"Top Ratio: {top_ratio} mm/pixel")
    print(f"Bottom Ratio: {bottom_ratio} mm/pixel")
    print(f"Averaged Pixel-to-MM Ratio: {pixel_to_mm_ratio} mm/pixel")
else:
    print("Unable to calculate pixel-to-mm ratios. Check if the top and bottom borders are correctly detected.")

# Measure and annotate other contours based on pixel-to-mm ratio
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    # Convert pixel dimensions to millimeters
    width_mm = w * pixel_to_mm_ratio
    height_mm = h * pixel_to_mm_ratio
    
    # Annotate the dimensions on the image
    cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(contour_image, f"{width_mm:.2f}mm x {height_mm:.2f}mm", 
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the final image with measurements and highlighted reference lines
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title("Contours with Highlighted Top and Bottom Reference Lines")
plt.show()
