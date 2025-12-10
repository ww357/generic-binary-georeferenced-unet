from PIL import Image
import numpy as np
from patchify import patchify
import os
import glob
from pathlib import Path
from tqdm import tqdm
import module_model_architecture
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window
import re

#############################################################
# Functions in this module:
# - make patches
# - setup directories
# - deploy model on folder
# - constructing map mask
# - post morph descripts <- this one is not used
#############################################################

# notebook "PatchifyTitheMapForPrediction(Overlapping)" as a function:

def make_patches(map_name, input_map, output_dir, patch_size):
    """
    Create overlapping patches (50% overlap) using rasterio windowed reads.
    Reads only one patch into memory at a time (memory-friendly for large GeoTIFFs).
    """
    step = patch_size // 2
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_map) as src:
        width = src.width
        height = src.height
        bands = src.count

        i = 0
        y = 0
        while y < height:
            j = 0
            x = 0
            while x < width:
                w = min(patch_size, width - x)
                h = min(patch_size, height - y)
                window = Window(col_off=x, row_off=y, width=w, height=h)

                # read window -> shape (bands, h, w)
                arr = src.read(window=window)

                # transpose to (h, w, bands)
                arr = np.transpose(arr, (1, 2, 0))

                # if single band, convert to 3-channel RGB by repeating
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)

                # if window smaller than patch_size, paste into canvas
                if (w, h) != (patch_size, patch_size):
                    canvas = np.zeros((patch_size, patch_size, 3), dtype=arr.dtype)
                    canvas[0:h, 0:w, :] = arr
                    arr = canvas

                # ensure uint8 for PIL (convert/scaling if needed)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)

                patch_img = Image.fromarray(arr)
                patch_filename = f"{map_name}_{i}_{j}.png"
                patch_path = os.path.join(output_dir, patch_filename)
                patch_img.save(patch_path)

                j += 1
                x += step
            i += 1
            y += step

    return f"process finished. Patches with 50% overlap saved to '{output_dir}'"

def setup_directories(input_folder, output_folder, image_format, batch_size):
    """Create output directory for predicted masks and scan input folder"""
    os.makedirs(output_folder, exist_ok=True)
    output_dirs = {'masks': output_folder}
    
    # Scan input folder for images
    image_files = []
    if os.path.exists(input_folder):
        for ext in image_format:
            pattern = os.path.join(input_folder, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(input_folder, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))
    image_files = sorted(list(set(image_files)))
    
    print(f"\nFound {len(image_files)} images in input folder: {input_folder}")
    if len(image_files) == 0:
        print("WARNING: No images found! Please check your input folder path and supported formats.")
    else:
        print(f"First few files: {[os.path.basename(f) for f in image_files[:5]]}")
        if len(image_files) > 5:
            print(f"... and {len(image_files) - 5} more files")
    
    print(f"\nSetup completed!")
    print(f"Total images to process: {len(image_files)}")
    print(f"Estimated batches: {(len(image_files) + batch_size - 1) // batch_size}")
    
    return output_dirs, image_files


def deploy_model_on_folder(image_files, output_dirs, model_weights_path, patch_size, image_channels, batch_size, threshold):
    """Main function to deploy model on entire folder"""
    
    if len(image_files) == 0:
        print("No images to process. Exiting.")
        return None
    
    # Load the trained model
    print("Loading trained model...")
    model = module_model_architecture.build_attn_unet((patch_size, patch_size, image_channels), module_model_architecture.dice_loss)
    
    if not os.path.exists(model_weights_path):
        print(f"ERROR: Model weights file not found: {model_weights_path}")
        return None
    
    model.load_weights(model_weights_path)
    print("Model loaded successfully!")
    
    # Process images in batches
    all_results = []
    total_images = len(image_files)

    print(f"\nStarting batch processing...")
    print(f"Total images: {total_images}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {(total_images + batch_size - 1) // batch_size}")
    
    # Create progress bar
    with tqdm(total=total_images, desc="Processing images") as pbar:
        
        # Process in batches
        for i in range(0, total_images, batch_size):
            batch_images = image_files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_images + batch_size - 1) // batch_size
            
            pbar.set_description(f"Processing batch {batch_num}/{total_batches}")
            
            try:
                # Process current batch
                batch_results = module_model_architecture.process_batch(batch_images, model, output_dirs, patch_size, threshold, pbar)
                all_results.extend(batch_results)

            except Exception as e:
                print(f"\nError processing batch {batch_num}: {str(e)}")
                # Add failed results for this batch
                for img_path in batch_images:
                    all_results.append({
                        'filename': Path(img_path).stem,
                        'original_path': img_path,
                        'processed': False,
                        'error': str(e)
                    })
                pbar.update(len(batch_images))
    
    print("\nBatch processing completed!")
    return all_results, model

def constructing_map_mask(map_name, input_map, output_folder, patch_size):
    '''stitch overlapping predicted patches into georeferenced map'''
    with rasterio.open(input_map) as src:
        original_crs = src.crs
        original_transform = src.transform
        original_bounds = src.bounds
        original_height = src.height
        original_width = src.width
        original_dtype = src.dtypes[0]
    print(f"Original map dimensions: {original_width} x {original_height}")
    print(f"Original CRS: {original_crs}")
    print(f"Original bounds: {original_bounds}")
    step=int(patch_size/2)
    patch_files = glob.glob(os.path.join(output_folder, "*.png"))
    patch_files.sort()
    if not patch_files:
        raise FileNotFoundError(f"No patch files found in {output_folder}")
    print(f"Found {len(patch_files)} mask patches")
    # Regex: accept filenames like "patch_1_2.png", "Selworthy_1_2.png", or "Selworthy_1_2_mask.png"
    coord_re = re.compile(r'_(\d+)_(\d+)(?:_mask)?(?:\.\w+)?$')
    # Extract patch coordinates from filenames and organize
    patch_dict = {}
    max_i, max_j = 0, 0
    for patch_file in patch_files:
        filename = os.path.basename(patch_file)
        # try explicit "patch_i_j" first, else general coord_re
        m = re.search(r'patch_(\d+)_(\d+)', filename)
        if not m:
            m = coord_re.search(filename)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            patch_dict[(i, j)] = patch_file
            max_i = max(max_i, i)
            max_j = max(max_j, j)
        else:
            print(f"Warning: could not parse indices from {filename}")
    grid_rows = max_i + 1
    grid_cols = max_j + 1
    print(f"Patch grid indices: rows={grid_rows}, cols={grid_cols}")
    print(f"Detected patch_size={patch_size}, using step={step}")
    padded_height = (grid_rows - 1) * step + patch_size
    padded_width  = (grid_cols - 1) * step + patch_size
    # Calculate the padding that was originally applied
    pad_h = padded_height - original_height
    pad_w = padded_width - original_width
    print(f"Padded dimensions: {padded_width} x {padded_height}")
    print(f"Original padding applied: height={pad_h}, width={pad_w}")
    print(f"Will crop final image to: {original_width} x {original_height}")
    sample_patch_path = patch_files[0]
    sample_patch = Image.open(sample_patch_path)
    sample_array = np.array(sample_patch)
    if len(sample_array.shape) == 2:
        stitched_array = np.zeros((padded_height, padded_width), dtype=sample_array.dtype)
        channels = 1
    else:
        channels = sample_array.shape[2]
        stitched_array = np.zeros((padded_height, padded_width, channels), dtype=sample_array.dtype)
    print(f"Mask patches are {'grayscale' if channels == 1 else f'{channels}-channel'}")
    # stitching loop: places each patch into stitched_array using step and np.maximum
    for (i, j), pf in patch_dict.items():
        patch_img = Image.open(pf)
        patch_array = np.array(patch_img)
        # normalize shape for single-channel
        if channels == 1 and patch_array.ndim == 3:
            patch_array = patch_array[..., 0]
        start_row = int(i * step)
        end_row = start_row + patch_size
        start_col = int(j * step)
        end_col = start_col + patch_size
        # clamp to stitched array bounds (handles edge patches if any)
        sr = max(0, start_row); er = min(padded_height, end_row)
        sc = max(0, start_col); ec = min(padded_width, end_col)
        # compute patch slice indices (in case of clipping)
        pr0 = sr - start_row
        pr1 = pr0 + (er - sr)
        pc0 = sc - start_col
        pc1 = pc0 + (ec - sc)
        if er <= sr or ec <= sc:
            print(f"Skipping out-of-bounds patch {(i,j)}")
            continue
        if channels == 1:
            stitched_array[sr:er, sc:ec] = np.maximum(
                stitched_array[sr:er, sc:ec],
                patch_array[pr0:pr1, pc0:pc1]
            )
        else:
            stitched_array[sr:er, sc:ec, :] = np.maximum(
                stitched_array[sr:er, sc:ec, :],
                patch_array[pr0:pr1, pc0:pc1, :]
            )
    print("Patches stitched into stitched_array.")
    # Crop the stitched array to remove padding and match the original image size
    final_array = stitched_array[:original_height, :original_width]
    # Prepare output filename
    stitched_mask = f"images-for-prediction/{map_name}/{map_name}_stitched_mask.tif"
    # Determine the appropriate data type for the output
    # Common mask dtypes: uint8 for 0-255 values, bool for binary masks
    output_dtype = final_array.dtype
    # Prepare array for rasterio (handle different channel configurations)
    if channels == 1:
        # For grayscale, rasterio expects (bands, height, width)
        output_array = final_array.reshape(1, original_height, original_width)
        count = 1
    else:
        # For multi-channel, transpose to (bands, height, width)
        output_array = np.transpose(final_array, (2, 0, 1))
        count = channels
    # Save the georeferenced TIFF
    with rasterio.open(
        stitched_mask,
        'w',
        driver='GTiff',
        height=original_height,
        width=original_width,
        count=count,
        dtype=output_dtype,
        crs=original_crs,
        transform=original_transform,
        compress='lzw'  # Optional compression
    ) as dst:
        dst.write(output_array)
    print(f"Georeferenced mask saved as: {stitched_mask}")
    print(f"Output dimensions: {original_width} x {original_height}")
    print(f"Output CRS: {original_crs}")
    print(f"Output channels: {count}")
    with rasterio.open(stitched_mask) as verify:
        print(f"\nVerification:")
        print(f"Saved file CRS: {verify.crs}")
        print(f"Saved file bounds: {verify.bounds}")
        print(f"Saved file shape: {verify.shape}")

# THIS DID NOT WORK SO IGNORE:

def post_morph_descripts(map_name):
    import cv2 as cv
    import rasterio
    map_path = f"{map_name}/{map_name}_stitched_mask.tif"
    map = cv.imread(map_path, cv.IMREAD_GRAYSCALE)
    closekernel = np.ones((2,2),np.uint8)
    openkernel = np.ones((1,1), np.uint8)
    closing = cv.morphologyEx(map, cv.MORPH_CLOSE, closekernel)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, openkernel)
    output = f"{map_name}/{map_name}_stitched_mask_denoised.tif"
    with rasterio.open(map_path) as src:
        meta = src.meta.copy()
        meta.update(dtype="uint8", count=1)
        with rasterio.open(output, 'w', **meta) as dst:
            dst.write(opening, 1)
    print(f"Small errors fixed and saved as {output}")
