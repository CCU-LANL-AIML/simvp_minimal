import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import hashlib

def generate_unique_id(experiment_record):
    """Generate a SHA-256 hash as a unique ID for the experiment record."""
    serialized_record = json.dumps(experiment_record, sort_keys=True)
    hash_object = hashlib.sha256(serialized_record.encode())
    return hash_object.hexdigest()

def generate_experiment_record(**params):
    """Generate a dictionary for the experiment with a unique ID."""
    unique_id = generate_unique_id({key: params[key] for key in params if key != 'id'})
    params['id'] = unique_id
    print(f'Unique Experiment ID: {unique_id}')
    return params

def save_experiment_record(experiment_record, filename):
    """Save the experiment record to a JSON file, creating directory if needed."""
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist. Creating.")
        os.makedirs(directory)
    with open(filename, 'w') as file:
        json.dump(experiment_record, file, indent=4)

def load_files(datafolder, num_samples=None, sample_start_index=0, total_length=0, init_final_only=False, reverse=False):
    """
    Load files from the data folder for creating data loaders.
    """
    # Get all sample folders
    folders = [f for f in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, f))]
    
    # Select random subset if num_samples is specified
    if num_samples is not None and num_samples < len(folders):
        folders = np.random.choice(folders, num_samples, replace=False)

    progress_iterator = tqdm(folders, desc="Loading samples")

    data = []
    for unique_id in progress_iterator:
        # Get all .npy files in the folder
        files = sorted([f for f in os.listdir(os.path.join(datafolder, unique_id)) if f.endswith('.npy')], 
                     key=lambda x: int(os.path.splitext(x)[0]))
        file_count = len(files)
        
        # Make sure we have enough frames
        sample_length = total_length if total_length > 0 else file_count
        if init_final_only:
            sample_length = 2  # Only initial and final frame
            
        if file_count < sample_length:
            print(f"Skipping {unique_id} due to insufficient data.")
            continue

        # Determine start index
        start_index = sample_start_index
        if start_index == -1:  # Random start
            start_index = np.random.randint(0, file_count - sample_length + 1)
        
        # Get file paths
        final_files = []
        for j in range(start_index, start_index + sample_length):
            if j < file_count:  # Ensure we don't exceed available files
                final_files.append(os.path.join(datafolder, unique_id, f"{j}.npy"))
        
        if reverse:
            final_files.reverse()

        if init_final_only and len(final_files) >= 2:
            data.append([final_files[0], final_files[-1]])
        else:
            data.append(final_files)

    return data

def train_val_test_split_files(data, train_ratio, val_ratio):
    """
    Split files into training, validation and test sets.
    """
    train_ratio = round(train_ratio, 3)
    val_ratio = round(val_ratio, 3)

    np.random.shuffle(data)

    train_size = round(len(data) * train_ratio)
    val_size = round(len(data) * val_ratio)

    train_data = data[:train_size] if train_ratio > 0 else []
    val_data = data[train_size:train_size + val_size] if val_ratio > 0 else []
    test_data = data[train_size + val_size:] if train_ratio + val_ratio < 1 else []

    splits = {}
    if train_data:
        splits['train'] = {
            'ratio': train_ratio,
            'samples': train_data
        }
    if val_data:
        splits['validation'] = {
            'ratio': val_ratio,
            'samples': val_data
        }
    if test_data:
        splits['test'] = {
            'ratio': round(1 - train_ratio - val_ratio, 3),
            'samples': test_data
        }

    return splits

def main():
    parser = argparse.ArgumentParser(
        description="Prepare data loaders for simulation dataset")
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the data folder')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training data ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation data ratio (default: 0.15)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Total number of samples to randomly select')
    parser.add_argument('--sample_start_index', type=int, default=0,
                        help='Starting index of the sequence, -1 for random')
    parser.add_argument('--total_length', type=int, default=0,
                        help='Total length of sequence to use, 0 for full sequence')
    parser.add_argument('--init_final_only', action='store_true',
                        help='Only use initial and final frames')
    parser.add_argument('--reverse', action='store_true',
                        help='Reverse the order of frames')
    
    args = parser.parse_args()
    
    # Check if data folder exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data folder in path {args.data_path} does not exist.")
    
    # Load the files
    print("Loading files from data folder...")
    files = load_files(
        args.data_path, 
        args.num_samples, 
        args.sample_start_index, 
        args.total_length, 
        args.init_final_only, 
        args.reverse
    )
    
    if not files:
        print("No valid files found. Please check your data folder and parameters.")
        return
    
    # Split the files
    print(f"Splitting {len(files)} samples into train/validation/test sets...")
    train_val_test_splits = train_val_test_split_files(
        files, 
        args.train_ratio, 
        args.val_ratio
    )
    
    # Save the loaders into json
    record = generate_experiment_record(**train_val_test_splits)
    loader_file = os.path.join(args.data_path, f"{record['id']}_loaders.json")
    save_experiment_record(record, loader_file)
    
    # Print summary
    print(f"Data loader configuration saved to: {loader_file}")
    print("Data split summary:")
    for split_name, split_data in train_val_test_splits.items():
        print(f"  {split_name}: {len(split_data['samples'])} samples ({split_data['ratio']*100:.1f}%)")

if __name__ == '__main__':
    main()
