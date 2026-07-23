import time

def benchmark(function, *args, runs=10):
    # Measures the average execution time of a function.
    

    # Warm-up
    function(*args)

    start = time.perf_counter()

    for _ in range(runs):
        function(*args)

    end = time.perf_counter()

    average = (end - start) / runs

    print(f"{function.__name__}: {average:.6f} seconds")

    return average