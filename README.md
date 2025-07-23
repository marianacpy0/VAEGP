import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 plots

sns.scatterplot(data=latent_df, x='z1', y='TAN', ax=axes[0])
axes[0].set_title('z1 vs TAN')

sns.scatterplot(data=latent_df, x='z2', y='TAN', ax=axes[1])
axes[1].set_title('z2 vs TAN')

sns.scatterplot(data=latent_df, x='z3', y='TAN', ax=axes[2])
axes[2].set_title('z3 vs TAN')

plt.tight_layout()
plt.show()




import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/mnt/data/F150604F-BD67-483F-9FAF-7298194A2EDB.jpeg"
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to highlight darker regions
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Use Canny edge detection
edges = cv2.Canny(adaptive_thresh, 50, 150)

# Use Hough Line Transform to detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20)

# Initialize variables to store the detected lines for reference
top_red_line = None
blue_vertical_line = None
purple_vertical_line = None

# Draw lines and identify reference lines based on orientation and length
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Check if the line is horizontal (top red line candidate)
        if abs(y2 - y1) < 10 and line_length > width / 4:  # Horizontal line criteria
            top_red_line = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw in red for top red line
            cv2.putText(image, "Top Red Line", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Check if the line is vertical (blue or purple line candidates)
        elif abs(x2 - x1) < 10 and line_length > height / 8:  # Vertical line criteria
            if blue_vertical_line is None:
                blue_vertical_line = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw in blue for blue line
                cv2.putText(image, "Blue Line", (x1 + 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            elif purple_vertical_line is None:
                purple_vertical_line = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Draw in purple for purple line
                cv2.putText(image, "Purple Line", (x1 + 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

# Display the result with detected lines
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Reference Lines with Hough Transform")
plt.show()

# Check if we successfully detected the required reference lines
if top_red_line is not None and blue_vertical_line is not None and purple_vertical_line is not None:
    # Calculate the lengths of the detected lines in pixels
    top_red_length_px = np.sqrt((top_red_line[2] - top_red_line[0]) ** 2 + (top_red_line[3] - top_red_line[1]) ** 2)
    blue_vertical_length_px = np.sqrt((blue_vertical_line[2] - blue_vertical_line[0]) ** 2 + (blue_vertical_line[3] - blue_vertical_line[1]) ** 2)
    purple_vertical_length_px = np.sqrt((purple_vertical_line[2] - purple_vertical_line[0]) ** 2 + (purple_vertical_line[3] - purple_vertical_line[1]) ** 2)
    
    # Known measurements in mm
    top_red_mm = 55.58
    blue_vertical_mm = 5.68
    purple_vertical_mm = 6.68

    # Calculate pixel-to-mm ratios
    horizontal_ratio = top_red_mm / top_red_length_px
    vertical_ratio_blue = blue_vertical_mm / blue_vertical_length_px
    vertical_ratio_purple = purple_vertical_mm / purple_vertical_length_px
    vertical_ratio = (vertical_ratio_blue + vertical_ratio_purple) / 2  # Average for general vertical ratio

    print(f"Horizontal Pixel-to-MM Ratio: {horizontal_ratio} mm/pixel")
    print(f"Vertical Pixel-to-MM Ratio: {vertical_ratio} mm/pixel")

    # Display the calculated ratios on the image
    cv2.putText(image, f"Horizontal Ratio: {horizontal_ratio:.2f} mm/pixel", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"Vertical Ratio: {vertical_ratio:.2f} mm/pixel", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the final image with ratios
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Reference Lines with Pixel-to-MM Ratios")
    plt.show()
else:
    print("One or more reference lines were not detected. Please adjust the parameters.")

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
