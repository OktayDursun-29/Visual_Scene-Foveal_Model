import numpy as np
from PIL import Image, ImageDraw

def generate_receptive_fields(image_shape, fixation, base_radius, growth_rate, overlap_density):
    """
    Generates a list of Receptive Fields radiating outward in concentric rings.
    """
    height, width = image_shape[:2]
    fix_x, fix_y = fixation
    
    # Calculate the maximum distance to the furthest corner of the image
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    max_dist = np.max(np.linalg.norm(corners - np.array([fix_x, fix_y]), axis=1))
    
    rfs = []
    
    # 1. Add the central foveal RF
    rfs.append({"x": fix_x, "y": fix_y, "r": base_radius})
    
    # 2. Iterate outward in rings
    current_dist = base_radius * overlap_density
    
    while current_dist < max_dist + base_radius:
        # Calculate radius at the current distance based on growth rate
        r = base_radius + (current_dist * growth_rate)
        
        # Calculate circumference of the current ring
        circumference = 2 * np.pi * current_dist
        
        # Determine how many circles fit in this ring based on the overlap density
        spacing = r * overlap_density
        num_circles = max(1, int(circumference / spacing))
        
        # Generate the center coordinates (x,y) for this ring
        angles = np.linspace(0, 2 * np.pi, num_circles, endpoint=False)
        for angle in angles:
            cx = fix_x + current_dist * np.cos(angle)
            cy = fix_y + current_dist * np.sin(angle)
            
            # Only save the RF if it actually touches the image boundaries, else drop it
            if -r <= cx <= width + r and -r <= cy <= height + r:
                rfs.append({"x": cx, "y": cy, "r": r})
        
        # Step out to the next concentric ring
        current_dist += r * overlap_density
        
    return rfs


def extract_rf_statistics(image, rfs):
    """
    Iterates through the generated RFs, applies a circular mask, 
    and calculates the Mean and Variance for the RGB channels.
    """
    height, width = image.shape[:2]
    statistical_map = []
    
    for rf in rfs:
        cx, cy, r = rf["x"], rf["y"], rf["r"]
        
        # Define a square bounding box around the circle
        min_x = max(0, int(cx - r))
        max_x = min(width, int(cx + r) + 1)
        min_y = max(0, int(cy - r))
        max_y = min(height, int(cy + r) + 1)
        
        # Skip if the bounding box is entirely outside the image
        if min_x >= max_x or min_y >= max_y:
            continue
            
        # Crops the image to the bounding box
        crop = image[min_y:max_y, min_x:max_x]
        
        # Creates a coordinate grid to define the circular mask mathematically
        X, Y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
        
        # Equation of a circle: (X - cx)^2 + (Y - cy)^2 <= r^2
        circular_mask = ((X - cx)**2 + (Y - cy)**2) <= r**2
        
        # Extract only the pixels that fall inside the true circle
        pixels = crop[circular_mask]
        
        # Calculate the statistics make sure there are pixels to avoid dividing by zero
        if len(pixels) > 0:
            # axis=0 means it calculates the mean & var independently for R, G, and B
            mean_rgb = np.mean(pixels, axis=0)
            var_rgb = np.var(pixels, axis=0)
            
            # Some pixels might have identical colors, causing variance to be exactly 0
            # Have to add a tiny number to variance so the future likelihood math doesn't crash
            epsilon = 1e-6
            var_rgb = np.maximum(var_rgb, epsilon)
            
            statistical_map.append({
                "rf": rf,               # Original spatial data (x, y, radius)
                "pixel_count": len(pixels),
                "mean": mean_rgb,       # Array of [R_mean, G_mean, B_mean]
                "variance": var_rgb     # Array of [R_var, G_var, B_var]
            })
            
    return statistical_map
