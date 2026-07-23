import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as PlotCircle
from scipy.stats import norm
from Scene import Scene, Circle, Square, Triangle
from ReceptiveFields import generate_receptive_fields, predict_rf_stats
from Benchmark import benchmark
from ReceptiveFieldsJAX import predict_rf_stats_jax

# Creating fixed scene
def create_scene(width=800, height=800):
    # Creates mathematical scene and renders it
    my_scene = Scene(background_color=(255,255,255))
    my_scene.add_object(Circle(x=200, y=200, radius=60, color=(0,0,255)))
    my_scene.add_object(Square(x=350, y=250, size=80, color=(0,255,0)))
    my_scene.add_object(Triangle(x=450, y=400, size=80, color=(255,0,0)))
    return my_scene, np.array(my_scene.render_to_image(width, height)).astype(np.float32)

# Creating randomized scene
def create_random_scene(width=800, height=800):
    my_scene = Scene(background_color=(255,255,255))
    random_color = lambda: (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    my_scene.add_object(Circle(random.randint(100,width-100), random.randint(100,height-100), random.randint(30,120), random_color()))
    my_scene.add_object(Square(random.randint(100,width-100), random.randint(100,height-100), random.randint(30,120), random_color()))
    my_scene.add_object(Triangle(random.randint(100,width-100), random.randint(100,height-100), random.randint(30,120), random_color()))
    
    return my_scene, np.array(my_scene.render_to_image(width, height)).astype(np.float32)

# Saving stats
def save_statistics_to_csv(stats, filename):
    print(f"\nSaving data to {filename}...")
    if directory := os.path.dirname(filename):
        os.makedirs(directory, exist_ok=True)

    headers = ["rf_x", "rf_y", "rf_radius", "pixel_count", "mean_R", "mean_G", "mean_B", "var_R", "var_G", "var_B"]

    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        for stat in stats:
            rf = stat["rf"]
            writer.writerow([rf["x"], rf["y"], rf["r"], stat["pixel_count"], *stat["mean"], *stat["variance"]])

    print("CSV save complete!")

# Calculatin the likelihood
def compute_scene_loglikelihood(actual_stats, observed_stats):
    # Compares observed image RF statistics against predicted scene RF statistics 
   
    loglikelihood = 0.0

    for obs_mean, pred_stat in zip(observed_stats, actual_stats):
        mu = pred_stat["mean"]
        sigma = np.sqrt(np.maximum(pred_stat["variance"], 1e-6))
        loglikelihood += np.sum(norm.logpdf(obs_mean, loc=mu, scale=sigma))

    return loglikelihood

# Creating the three panel visualization
def visualize_rf_statistics(image, actual_stats, save_path="results/rf_visualization.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    img_uint8 = image.astype(np.uint8)

    # Panel 1
    axes[0].imshow(img_uint8)
    axes[0].set_title("Rendered Scene")
    axes[0].axis("off")

    # Panel 2
    axes[1].imshow(img_uint8)
    for stat in actual_stats:
        rf = stat["rf"]
        # Clip the color values to make sure they stay between 0.0 and 1.0
        color_val = np.clip(stat["mean"] / 255, 0.0, 1.0)
        axes[1].add_patch(PlotCircle((rf["x"], rf["y"]), rf["r"], fill=False, linewidth=3, color=color_val))
        axes[1].set_title("RFs Colored by Mean RGB")
        axes[1].axis("off")

    # Panel 3
    axes[2].imshow(img_uint8)
    variance_values = [np.mean(stat["variance"]) for stat in actual_stats]
    max_var = max(variance_values) if variance_values else 1

    for stat, var in zip(actual_stats, variance_values):
        rf = stat["rf"]
        axes[2].add_patch(PlotCircle((rf["x"], rf["y"]), rf["r"], fill=False, linewidth=3, color=plt.cm.coolwarm(var / max_var)))
    axes[2].set_title("RFs Colored by Variance")
    axes[2].axis("off")

    plt.tight_layout()
    
    # Making sure the output directory exists
    if directory := os.path.dirname(save_path):
        os.makedirs(directory, exist_ok=True)
        
    # Save and close instead of showing to avoid blocking execution in non-interactive environments
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

# Main function to run the entire process
def main():
    IMAGE_WIDTH, IMAGE_HEIGHT = 800, 800
    FIXATION_POINT, BASE_RADIUS, GROWTH_RATE, OVERLAP_DENSITY = (200, 200), 25.0, 0.08, 2.0
    CSV_FILENAME = "results/foveal_statistics.csv"

    print("Generating original predicted scene...")
    actual_scene, actual_image = create_scene(IMAGE_WIDTH, IMAGE_HEIGHT)

    print("Generating randomized observed scene...")
    random_scene, random_observed_image = create_random_scene(IMAGE_WIDTH, IMAGE_HEIGHT)

    print("\nGenerating receptive fields...")
    rfs = generate_receptive_fields(actual_image.shape, FIXATION_POINT, BASE_RADIUS, GROWTH_RATE, OVERLAP_DENSITY, target_rf_count=100)
    print(f"Total RFs generated: {len(rfs)}")

    # predicting stats directly from the scene using JAX
    print("\nPredicting stats using JAX...")
    actual_stats = predict_rf_stats_jax(actual_scene, rfs)

    # predicting stats for the random scene using JAX
    random_stats = predict_rf_stats_jax(random_scene, rfs)

    print("\nComputing likelihood...")
    
    # Extract the means from the predicted stats 
    actual_obs_means = [stat["mean"] for stat in actual_stats]
    random_obs_means = [stat["mean"] for stat in random_stats]

    actual_log_lik = compute_scene_loglikelihood(actual_stats, actual_obs_means)
    random_log_lik = compute_scene_loglikelihood(actual_stats, random_obs_means)

    print("\nLog-Likelihood Comparison:")
    print(f"Original Scene: {actual_log_lik}")
    print(f"Random Scene:   {random_log_lik}")

    print("\nCreating visualization...")
    save_file = "results/rf_visualization.png"
    visualize_rf_statistics(actual_image, actual_stats, save_path=save_file)
    print(f"Visualization successfully saved to {save_file}")

    print("\n--- Benchmarking Normal Python ---")
    # Benchmark the Normal Python/Shapely version
    benchmark(
        predict_rf_stats,
        actual_scene,
        rfs,
        runs=20
    )

    print("\n--- Benchmarking JAX ---")
    # Compile the JAX version by running it once before benchmarking
    predict_rf_stats_jax(actual_scene, rfs)

    # Benchmark the JAX version
    benchmark(
        predict_rf_stats_jax,
        actual_scene,
        rfs,
        runs=20
    )
    
if __name__ == "__main__":
    main()