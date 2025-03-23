import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sunpy.map import Map
from matplotlib.colors import Normalize
from astropy.coordinates import SkyCoord
from sunpy.map import all_coordinates_from_map
import astropy.units as u  # Ensure correct unit usage
from scipy.ndimage import gaussian_filter
from PIL import Image


def get_data(fits_file):
    # Load the map from the FITS file
    s_map = Map(fits_file)

    # Get the center of the solar disk (assumed to be at the center of the image)
    center_x, center_y = s_map.data.shape[1] // 2, s_map.data.shape[0] // 2

    # Define solar radius in pixels (you can adjust this value based on the scale of your image)
    radius_pixels = s_map.rsun_obs.to(u.arcsec).value / s_map.scale[0].value  # Convert solar radius to pixels

    # Create a grid of coordinates to calculate radial distance from the center
    y, x = np.ogrid[:s_map.data.shape[0], :s_map.data.shape[1]]
    radial_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Mask for pixels inside and outside the solar disk
    inside_disk_mask = radial_distance <= radius_pixels  # Mask for pixels inside the solar disk
    outside_disk_mask = radial_distance > radius_pixels  # Mask for pixels outside the solar disk

    # Apply Limb Darkening Correction (LDC) only inside the solar disk
    coords = all_coordinates_from_map(s_map)
    radial_distance_coords = (np.sqrt(coords.Tx**2 + coords.Ty**2) / s_map.rsun_obs).value
    radial_distance_coords[radial_distance_coords >= 1] = np.nan
    ideal_correction = np.cos(radial_distance_coords * np.pi / 2)

    # Apply correction only to the region inside the solar disk
    condition = np.logical_not(np.isnan(np.ravel(ideal_correction)))
    map_list = np.ravel(s_map.data)[condition]
    correction_list = np.ravel(ideal_correction)[condition]

    # Fit a polynomial correction (degree 6) and apply the correction to the map
    fit = np.polyfit(correction_list, map_list, 20)
    poly_fit = np.poly1d(fit)
    map_correction = poly_fit(ideal_correction)

    # Apply the correction and normalization to the data
    data = s_map.data / map_correction

    # Normalize the data for better visualization (avoid extremes)
    data[inside_disk_mask] = Normalize(0.6, 1.2, clip=True)(data[inside_disk_mask]) * 2 - 1  # Adjust normalization range
    data = np.nan_to_num(data, nan=-1)  # Replace NaN with -1

    # Contrast Enhancement: Stretch the data to increase contrast inside the disk
    data_min = np.min(data[inside_disk_mask])
    data_max = np.max(data[inside_disk_mask])
    data[inside_disk_mask] = 2 * (data[inside_disk_mask] - data_min) / (data_max - data_min) - 1  # Stretch to range [-1, 1]

    # Clip the values to ensure that the minimum and maximum are in the desired range
    data[inside_disk_mask] = np.clip(data[inside_disk_mask], -1, 1)

    # Set the pixels outside the solar disk to a neutral background
    data[outside_disk_mask] = 0.2  # Set outside disk pixels to a neutral background (lighter gray)

    # Smooth the edge region to make the transition more natural
    edge_radius_start = radius_pixels - 10  # Reduced edge smoothing range
    edge_radius_end = radius_pixels  # Outermost pixel of the disk

    # Mask for edge pixels (10-pixel thickness around the circumference)
    edge_mask = (radial_distance >= edge_radius_start) & (radial_distance <= edge_radius_end)

    # Identify intermediate region inside the disk (not too bright and not too dark)
    intermediate_mask = (data > -0.2) & (data < 0.7)  # This range can be adjusted to fit your image

    # Use the values from the intermediate region to blend the edge
    intermediate_values = data[intermediate_mask]

    # Calculate the mean intensity of the intermediate region
    intermediate_mean = np.mean(intermediate_values)

    # Set edge pixels to the mean value of the intermediate region to blend smoothly
    data[edge_mask] = intermediate_mean

    # Darken filaments and sunspots (regions with low intensity) and turn them completely black
    filament_mask = (data <= -0.2)  # Identifying darker regions (filaments/sunspots)
    data[filament_mask] = -1  # Set filaments to completely black

    return data


def process_and_save_images(source_dir, output_dir):
    """Process all FITS files in source_dir and save them to output_dir maintaining the directory structure."""
    # Walk through all directories and subdirectories in source_dir
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".fits.fz"):  # Process only FITS files
                fits_file_path = os.path.join(root, file)
                print(f"Processing: {fits_file_path}")

                # Apply the processing function
                corrected_data = get_data(fits_file_path)

                # Create the corresponding output directory structure
                relative_path = os.path.relpath(fits_file_path, source_dir)
                output_path = os.path.join(output_dir, relative_path)

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Save the processed image
                img_save_path = output_path.replace(".fits.fz", ".jpg")  # Change extension to .jpg
                save_image_as_jpg(corrected_data, img_save_path)

                print(f"Processed and saved: {img_save_path}")


def save_image_as_jpg(image_data, save_path):
    """Convert numpy array to image and save as .jpg."""
    # Convert the image data to 8-bit for saving as JPEG
    image_data_8bit = np.uint8((image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255)

    img = Image.fromarray(image_data_8bit)
    img.save(save_path)
    print(f"Image saved as: {save_path}")


# Example usage
source_dir = '/content/drive/MyDrive/filament-detection-project/data/YOLO-data/gong-fits'
output_dir = '/content/drive/MyDrive/filament-detection-project/data/YOLO-data/gong-processed-jpgs'

# Process and save all images in the source directory
process_and_save_images(source_dir, output_dir)
