import time

def benchmark(function, *args, runs=20, warmup=5):
    """
    Measure the average execution time of a function.
    """

    # Warm-up runs (important for JAX compilation)
    for _ in range(warmup):
        result = function(*args)

        # Wait for JAX computations if applicable
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    times = []

    for _ in range(runs):
        start = time.perf_counter()

        result = function(*args)

        # Wait for JAX computations if applicable
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

        end = time.perf_counter()
        times.append(end - start)

    average = sum(times) / len(times)

    print(f"\nBenchmark Results for {function.__name__}")
    print(f"Runs: {runs}")
    print(f"Average: {average:.6f} seconds ({average*1000:.3f} ms)")
    print(f"Fastest: {min(times):.6f} seconds")
    print(f"Slowest: {max(times):.6f} seconds")

    return average