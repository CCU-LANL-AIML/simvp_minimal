import os
import time
import numpy as np
import argparse
from tqdm import tqdm


def apply_boiling_simulation(samples, iterations, increment=10, save_history=True):
    """
    Simulates boiling on a series of samples.

    Args:
        samples: Initial state
        iterations: Number of iterations to run
        increment: Temperature increment per step
        save_history: Whether to save all intermediate steps

    Returns:
        Final state and history (if save_history=True)
    """
    vmin, vmax = 0.0, 212.0

    shape = samples.shape
    samples = samples.squeeze()
    history = np.copy(samples).squeeze()

    for _ in range(iterations):
        diffusion_to_each_neighbor = samples * (1 / 8)
        samples = np.zeros_like(samples)

        # Apply diffusion with neighbors
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                samples += np.roll(np.roll(diffusion_to_each_neighbor, dx, axis=0), dy, axis=1)

        # Apply temperature increment and wrap around
        samples = ((samples + increment) % vmax) + vmin

        if save_history:
            history = np.append(history, np.copy(samples))

    samples = np.copy(samples).reshape(shape)

    if save_history:
        history = np.array(history).reshape((-1,) + shape)
        return samples, history
    else:
        return samples, None


def normalize_data_min_max(dataset, vmin=0.0, vmax=212.0):
    """Normalizes the dataset using min-max scaling to a range of [0, 1]."""
    return (dataset - vmin) / (vmax - vmin)


def create_initials(rows, cols, num_samples, datafolder_out):
    """Generates initial conditions for the simulation."""
    progress_iterator = tqdm(range(num_samples), desc="Generating initials")

    for i in progress_iterator:
        # Create random initial conditions (1-channel image)
        arr = np.random.uniform(0.0, 212.0, (1, rows, cols)).astype(np.float32)

        # Create a unique hash-based name similar to the original format
        import hashlib
        arr_bytes = arr.tobytes()
        hash_object = hashlib.sha256(arr_bytes)
        unique_hash = hash_object.hexdigest()

        name = f'{unique_hash}_boiling_{i}'
        folder = f'{datafolder_out}/{name}'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save initial condition
        np.save(f'{folder}/0.npy', arr)


def create_samples(total_frames, datafolder, increment):
    """Generates frame sequences for each initial condition."""
    folders = [f for f in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, f))]
    progress_iterator = tqdm(folders, desc="Generating samples")

    for sample_dir in progress_iterator:
        files = [f for f in os.listdir(f'{datafolder}/{sample_dir}') if f.endswith('.npy')]
        if len(files) != 1:
            continue

        # Load initial condition
        try:
            arr = np.load(f'{datafolder}/{sample_dir}/0.npy')
        except FileNotFoundError:
            print(f"Initial condition for {sample_dir} not found, skipping.")
            continue

        # Apply simulation to generate frames
        _, samples = apply_boiling_simulation(arr, total_frames - 1, increment)

        # Save each frame
        for j, sample in enumerate(samples):
            # Always normalize
            sample = normalize_data_min_max(sample)
            np.save(f'{datafolder}/{sample_dir}/{j}.npy', sample)


def main():
    parser = argparse.ArgumentParser(description="Generate boiling simulation dataset")

    parser.add_argument('--num_samples', type=int, required=True,
                        help='Number of samples to generate')
    parser.add_argument('--num_frames', type=int, required=True,
                        help='Number of frames per sample')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to store the generated data')
    parser.add_argument('--increment', type=int, default=10,
                        help='Boiling simulation increment value (default: 10)')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Size of the simulation grid (default: 64)')

    args = parser.parse_args()

    # Ensure data directory exists
    if not os.path.exists(args.data_path):
        print(f"Path '{args.data_path}' does not exist. Creating...")
        os.makedirs(args.data_path)

    # Generate initials
    print("Step 1: Generating initial conditions...")
    start_time = time.time()
    create_initials(
        args.image_size,
        args.image_size,
        args.num_samples,
        args.data_path
    )
    elapsed = time.time() - start_time
    print(f"Generating {args.num_samples} initials took {elapsed:.2f} seconds.")

    # Generate samples
    print("Step 2: Generating frame sequences...")
    start_time = time.time()
    create_samples(args.num_frames, args.data_path, args.increment)
    elapsed = time.time() - start_time
    print(f"Generating samples took {elapsed:.2f} seconds.")

    print(f"Dataset generation complete. Data saved to {args.data_path}")


if __name__ == '__main__':
    main()