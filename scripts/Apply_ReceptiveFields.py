import numpy as np
from PIL import Image
import csv
from ReceptiveFields import generate_receptive_fields, extract_rf_statistics


# Loads in orginal image, un-blurred
# Converts to RGB to ensure we have 3 channels for the statistics
image = Image.open("img/visual_scene.png").convert("RGB")

# Convert the Pillow image into a NumPy array 
# so that we can perform statistical calculations on it
image_rgb = np.array(image)

# Convert to float to prevent integer overflow during statistical calculations
image_float = image_rgb.astype(np.float32)


# Defines parameters
image_shape = image_float.shape
fixation_point = (300, 300) # (x, y) coordinates of fixation point, where you are "looking"
base_radius = 15.0          # Minimum size of RF at the center
growth_rate = 0.05          # How fast they grow
overlap_density = 0.8       # Spacing between centers of RFs

# Generating the geometry
print("Generating Receptive Fields...")
receptive_fields = generate_receptive_fields(image_shape, fixation_point, base_radius, growth_rate, overlap_density)
print(f"Total RFs generated: {len(receptive_fields)}")

# Extract the stats
print("Extracting summary statistics...")
foveal_model_stats = extract_rf_statistics(image_float, receptive_fields)
print("Processing complete!")

# Looks at the data for the central-most RF and prints it
print("\nCenter RF Data:")
print(foveal_model_stats[0])
print(f"Total RFs processed: {len(foveal_model_stats)}")







# Saving the data to a CSV file
csv_filename = "results/foveal_statistics.csv"
print(f"\nSaving data to {csv_filename}...")

# Open the file in write mode 
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the column headers
    writer.writerow([
        'rf_x', 'rf_y', 'rf_radius', 'pixel_count', 
        'mean_R', 'mean_G', 'mean_B', 
        'var_R', 'var_G', 'var_B'
    ])
    
    # Loop through all processed RFs and write their data
    for stat in foveal_model_stats:
        row = [
            stat['rf']['x'],
            stat['rf']['y'],
            stat['rf']['r'],
            stat['pixel_count'],
            stat['mean'][0],      # Red Mean
            stat['mean'][1],      # Green Mean
            stat['mean'][2],      # Blue Mean
            stat['variance'][0],  # Red Variance
            stat['variance'][1],  # Green Variance
            stat['variance'][2]   # Blue Variance
        ]
        writer.writerow(row)

print("CSV save complete!")