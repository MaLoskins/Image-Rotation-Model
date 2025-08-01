# Licence headers and high-level comments are preserved.
import fiftyone as fo
import fiftyone.zoo as foz
import cv2
import numpy as np
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import shutil
import json

# --- Configuration ---
CONFIG = {
    "TEST_MODE": False,
    "SOURCE_DATASET_NAME": "coco-2017-train-20000",
    "FINAL_DATASET_NAME": "coco-2017-rotated",
    "PADDING_STRATEGY": 'reflect',  # Options: 'crop', 'reflect', 'random_bg', 'alpha_mask'
    "ANGLE_RANGE": (0, 360),  # Now includes full range
    "NUM_WORKERS": min(os.cpu_count() or 1, 8),  # Cap at 8 workers
    "QUALITY": 95,  # JPEG quality (when not using alpha_mask)
    "BATCH_SIZE": 100,  # Process samples in batches to manage memory
}
CONFIG.update({
    "SOURCE_MAX_SAMPLES": 5 if CONFIG["TEST_MODE"] else 1000,
    "NUM_ROTATIONS_PER_IMAGE": 2 if CONFIG["TEST_MODE"] else 6,
})


def rotate_image(image: np.ndarray, angle: float, strategy: str = 'crop', 
                 bg_color: tuple = (0, 0, 0)) -> np.ndarray:
    """Rotates an image, handling the created borders using various strategies.
    
    Args:
        image: Input image as numpy array
        angle: Rotation angle in degrees (0-360)
        strategy: Border handling strategy
        bg_color: Background color for 'crop' strategy
    
    Returns:
        Rotated image as numpy array
    """
    original_height, original_width = image.shape[:2]
    center = (original_width / 2, original_height / 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((original_height * sin) + (original_width * cos))
    new_height = int((original_height * cos) + (original_width * sin))
    
    # Adjust rotation matrix for new dimensions
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    if strategy == 'crop':
        # Rotate and crop to original size
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                               borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
        y_start = max(0, (new_height - original_height) // 2)
        x_start = max(0, (new_width - original_width) // 2)
        y_end = min(new_height, y_start + original_height)
        x_end = min(new_width, x_start + original_width)
        cropped = rotated[y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        if cropped.shape != (original_height, original_width, 3):
            result = np.full((original_height, original_width, 3), bg_color, dtype=np.uint8)
            y_offset = (original_height - cropped.shape[0]) // 2
            x_offset = (original_width - cropped.shape[1]) // 2
            result[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
            return result
        return cropped
        
    elif strategy == 'reflect':
        return cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                            borderMode=cv2.BORDER_REFLECT_101)
    
    elif strategy in ['random_bg', 'alpha_mask']:
        # Convert to RGBA
        if image.shape[2] == 3:
            img_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            img_rgba = image
            
        rotated_rgba = cv2.warpAffine(img_rgba, rotation_matrix, (new_width, new_height), 
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        
        if strategy == 'alpha_mask':
            return rotated_rgba
        
        # Random background
        background = np.random.randint(0, 256, (new_height, new_width, 3), dtype=np.uint8)
        alpha = rotated_rgba[:, :, 3] / 255.0
        foreground = rotated_rgba[:, :, :3]
        blended = (foreground * alpha[..., np.newaxis]) + (background * (1 - alpha[..., np.newaxis]))
        return blended.astype(np.uint8)
    
    raise ValueError(f"Unknown strategy: {strategy}")


def _rotation_worker(args: tuple) -> list[dict]:
    """Worker process for generating rotated images from a single source image."""
    filepath_str, gt_dict, output_dir, config = args
    filepath = Path(filepath_str)
    results = []

    try:
        img = cv2.imread(str(filepath))
        if img is None:
            print(f"Warning: Could not read {filepath}. Skipping.")
            return results
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return results

    # Extract original label if available
    original_label = None
    if gt_dict:
        if gt_dict["_cls"] == "fiftyone.core.labels.Classification":
            original_label = gt_dict.get("label")
        elif gt_dict["_cls"] == "fiftyone.core.labels.Detections" and gt_dict.get("detections"):
            original_label = gt_dict["detections"][0].get("label")

    # Determine file extension based on strategy
    file_ext = ".png" if config['PADDING_STRATEGY'] == 'alpha_mask' else ".jpg"
    
    # Generate multiple rotations
    for i in range(config['NUM_ROTATIONS_PER_IMAGE']):
        # Sample angle uniformly from full range
        angle = random.uniform(*config['ANGLE_RANGE'])
        
        # Apply rotation
        rotated_img = rotate_image(img, angle, strategy=config['PADDING_STRATEGY'])
        
        # Save rotated image
        new_filename = f"{filepath.stem}_rot{angle:.1f}_{i}{file_ext}"
        new_path = output_dir / new_filename
        
        if file_ext == ".jpg":
            cv2.imwrite(str(new_path), rotated_img, 
                       [cv2.IMWRITE_JPEG_QUALITY, config['QUALITY']])
        else:
            cv2.imwrite(str(new_path), rotated_img)

        # Prepare sample data
        sample_data = {
            "filepath": str(new_path),
            "rotation_angle": angle,
            "padding_strategy": config['PADDING_STRATEGY'],
            "is_rotated": True,
            "original_filepath": str(filepath),
            "tags": ["rotated"],
            "rotation_class": fo.Classification(label=f"{angle:.0f}°"),
        }
        
        if original_label:
            sample_data["original_label"] = original_label
        
        results.append(sample_data)
    
    return results


def get_dataset(name: str, split: str = "train", max_samples: int = 1000) -> fo.Dataset:
    """Loads a FiftyOne dataset if it exists, otherwise downloads it from the Zoo."""
    if fo.dataset_exists(name):
        print(f"Loading existing dataset '{name}'.")
        return fo.load_dataset(name)
    
    print(f"Loading and downloading dataset '{name}' from zoo...")
    return foz.load_zoo_dataset(
        "coco-2017", split=split, max_samples=max_samples,
        dataset_name=name, label_field="ground_truth"
    )


def main():
    """Main script execution with improved memory management and error handling."""
    print(f"--- Configuration ---")
    print(json.dumps(CONFIG, indent=2, default=str))
    print()
    
    start_time = time.time()

    # Load source dataset
    source_dataset = get_dataset(
        name=CONFIG["SOURCE_DATASET_NAME"], 
        max_samples=CONFIG["SOURCE_MAX_SAMPLES"]
    )
    source_dataset.persistent = True

    # Create output directory
    output_dir = Path(f"rotated_images_{CONFIG['PADDING_STRATEGY']}")
    if output_dir.exists() and CONFIG["TEST_MODE"]:
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Saving generated images to: {output_dir.resolve()}")

    # Prepare tasks
    tasks = [
        (s.filepath, s.ground_truth.to_dict() if s.ground_truth else None, output_dir, CONFIG) 
        for s in source_dataset.select_fields("ground_truth")
    ]
    
    # Process in batches to manage memory
    batch_size = CONFIG["BATCH_SIZE"]
    total_samples_created = 0
    failed_samples = 0
    
    # Delete old dataset if exists
    if fo.dataset_exists(CONFIG["FINAL_DATASET_NAME"]):
        fo.delete_dataset(CONFIG["FINAL_DATASET_NAME"])
        print(f"\nDeleted old version of dataset '{CONFIG['FINAL_DATASET_NAME']}'.")

    # Create final dataset by cloning source
    print(f"\nCreating final dataset '{CONFIG['FINAL_DATASET_NAME']}'...")
    final_dataset = source_dataset.clone(CONFIG["FINAL_DATASET_NAME"])
    final_dataset.persistent = True
    
    # Process samples in batches
    print(f"\nGenerating rotated samples with {CONFIG['NUM_WORKERS']} workers...")
    
    for batch_start in range(0, len(tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]
        batch_samples = []
        
        with ProcessPoolExecutor(max_workers=CONFIG['NUM_WORKERS']) as executor:
            future_to_task = {executor.submit(_rotation_worker, task): task for task in batch_tasks}
            
            with fo.ProgressBar(
                total=len(future_to_task), 
                start_msg=f"Processing batch {batch_start//batch_size + 1}"
            ) as pb:
                for future in as_completed(future_to_task):
                    try:
                        sample_data_list = future.result()
                        if sample_data_list:
                            batch_samples.extend([fo.Sample(**data) for data in sample_data_list])
                            total_samples_created += len(sample_data_list)
                    except Exception as e:
                        filepath = future_to_task[future][0]
                        print(f"\nError processing sample {filepath}: {e}")
                        failed_samples += 1
                    pb.update()
        
        # Add batch to dataset
        if batch_samples:
            final_dataset.add_samples(batch_samples)
            print(f"Added {len(batch_samples)} samples from batch {batch_start//batch_size + 1}")
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n--- Summary ---")
    print(f"Total samples in dataset: {len(final_dataset)}")
    print(f"Original samples: {CONFIG['SOURCE_MAX_SAMPLES']}")
    print(f"Rotated samples created: {total_samples_created}")
    print(f"Failed samples: {failed_samples}")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    # Verify dataset integrity
    rotated_count = len(final_dataset.match_tags("rotated"))
    print(f"\nVerification: Found {rotated_count} rotated samples in dataset")
    
    if CONFIG["TEST_MODE"]:
        print("\n--- Running Self-Contained Test ---")
        # Verify dataset
        loaded_test_ds = fo.load_dataset(CONFIG["FINAL_DATASET_NAME"])
        assert loaded_test_ds is not None, "Failed to load dataset"
        assert len(loaded_test_ds.match_tags("rotated")) == total_samples_created, "Rotated sample count mismatch"
        
        # Check sample files exist
        sample_path = Path(loaded_test_ds.match_tags("rotated").first().filepath)
        assert sample_path.exists(), "Sample file does not exist"
        
        print("--- All Tests Passed ---")
        
        # Cleanup
        fo.delete_dataset(CONFIG["SOURCE_DATASET_NAME"])
        fo.delete_dataset(CONFIG["FINAL_DATASET_NAME"])
        shutil.rmtree(output_dir)
        print("--- Test Cleanup Complete ---")
        return

    # Launch app
    print("\nLaunching FiftyOne App to view the final dataset...")
    session = fo.launch_app(final_dataset, auto=False)
    print("App is running. Press Ctrl+C in the console to exit.")
    
    try:
        session.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()