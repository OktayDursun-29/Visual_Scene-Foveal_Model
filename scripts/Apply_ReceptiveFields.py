import os
import csv
import random
import numpy as np
from PIL import Image
from scipy.stats import norm
from ReceptiveFields import generate_receptive_fields, extract_rf_statistics
from Scene import Scene, Circle, Square, Triangle


def create_scene(width=800, height=800):
    #Sets up the static mathematical scene and renders it
    my_scene = Scene(background_color=(255, 255, 255))
    
    my_scene.add_object(Circle(x=500, y=400, radius=80, color=(0, 0, 255)))
    my_scene.add_object(Square(x=200, y=400, size=80, color=(0, 255, 0)))
    my_scene.add_object(Triangle(x=350, y=200, size=80, color=(255, 0, 0)))

    image = my_scene.render_to_image(width=width, height=height)
    return np.array(image).astype(np.float32)


def create_random_scene(width=800, height=800):
    # Sets up a scene with the same shapes, but at random locations, sizes, and colors
    my_scene = Scene(background_color=(255, 255, 255))
    
    # Helper to generate a random RGB tuple
    def get_random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Add Circle with random properties (keeping a 100 pixel safety margin from the edges)
    my_scene.add_object(Circle(
        x=random.randint(100, width - 100),
        y=random.randint(100, height - 100),
        radius=random.randint(30, 120),
        color=get_random_color()
    ))
    
    # Add Square
    my_scene.add_object(Square(
        x=random.randint(100, width - 100),
        y=random.randint(100, height - 100),
        size=random.randint(30, 120),
        color=get_random_color()
    ))
    
    # Add Triangle
    my_scene.add_object(Triangle(
        x=random.randint(100, width - 100),
        y=random.randint(100, height - 100),
        size=random.randint(30, 120),
        color=get_random_color()
    ))

    image = my_scene.render_to_image(width=width, height=height)
    return np.array(image).astype(np.float32)


def save_statistics_to_csv(stats, filename):
    # Saves receptive field statistics to a CSV file
    print(f"\nSaving data to {filename}...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    headers = [
        'rf_x', 'rf_y', 'rf_radius', 'pixel_count', 
        'mean_R', 'mean_G', 'mean_B', 
        'var_R', 'var_G', 'var_B'
    ]

    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        
        for stat in stats:
            row = [
                stat['rf']['x'], stat['rf']['y'], stat['rf']['r'], stat['pixel_count'],
                *stat['mean'],     
                *stat['variance']  
            ]
            writer.writerow(row)
            
    print("CSV save complete!")


def compute_scene_loglikelihood(observed_image_float, receptive_fields, predicted_stats):
    #Calculates log-likelihood of an observed image given predicted scene stats.
    observed_rf_data = extract_rf_statistics(observed_image_float, receptive_fields)
    observed_stats = [stat['mean'] for stat in observed_rf_data]
    
    loglikelihood = 0.0
    
    for obs_mean, pred_stat in zip(observed_stats, predicted_stats):
        mu = pred_stat['mean']
        sigma = np.sqrt(np.maximum(pred_stat['variance'], 1e-6)) 

        channel_loglikelihoods = norm.logpdf(obs_mean, loc=mu, scale=sigma)
        loglikelihood += np.sum(channel_loglikelihoods)
        
    return loglikelihood


def main():
    #  Configuration Parameters 
    IMAGE_WIDTH, IMAGE_HEIGHT = 800, 800
    FIXATION_POINT = (300, 300)
    BASE_RADIUS = 15.0          
    GROWTH_RATE = 0.05          
    OVERLAP_DENSITY = 0.8       
    CSV_FILENAME = "results/foveal_statistics.csv"

    # Scene Generation 
    print("Generating original predicted scene...")
    predicted_image = create_scene(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

    print("Generating randomized observed scene...")
    random_observed_image = create_random_scene(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

    # Foveal Model Processing
    print("\nGenerating Receptive Fields...")
    receptive_fields = generate_receptive_fields(
        predicted_image.shape, FIXATION_POINT, BASE_RADIUS, GROWTH_RATE, OVERLAP_DENSITY
    )
    print(f"Total RFs generated: {len(receptive_fields)}")

    print("Extracting summary statistics for the predicted scene...")
    predicted_stats = extract_rf_statistics(predicted_image, receptive_fields)

    # Save Results
    save_statistics_to_csv(predicted_stats, CSV_FILENAME)

    # Log-Likelihood Comparisons
    # Baseline comparison: Predicted scene vs itself
    self_log_lik = compute_scene_loglikelihood(predicted_image, receptive_fields, predicted_stats)
    
    # Mismatch comparison: Random observed scene vs predicted scene map
    random_log_lik = compute_scene_loglikelihood(random_observed_image, receptive_fields, predicted_stats)
    
    print("\nLog-Likelihood Comparison:")
    print(f"Original Scene (Self) Log-Likelihood:  {self_log_lik}")
    print(f"Randomized Scene Log-Likelihood:      {random_log_lik}")


if __name__ == "__main__":
    main()