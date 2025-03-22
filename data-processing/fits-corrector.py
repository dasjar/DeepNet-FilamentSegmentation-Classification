# -*- coding: utf-8 -*-
"""fits-correction.ipynb

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sunpy.map import Map
from matplotlib.colors import Normalize
from astropy.coordinates import SkyCoord
from sunpy.map import all_coordinates_from_map
import astropy.units as u  # Ensure correct unit usage
from scipy.ndimage import gaussian_filter

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
    data[inside_disk_mask] = Normalize(0.8, 1.3, clip=True)(data[inside_disk_mask]) * 2 - 1  # Adjust normalization range
    data = np.nan_to_num(data, nan=-1)  # Replace NaN with -1

    # Contrast Enhancement: Stretch the data to increase contrast inside the disk
    data_min = np.min(data[inside_disk_mask])
    data_max = np.max(data[inside_disk_mask])
    data[inside_disk_mask] = 2 * (data[inside_disk_mask] - data_min) / (data_max - data_min) - 1  # Stretch to range [-1, 1]

    # Clip the values to ensure that the minimum and maximum are in the desired range
    data[inside_disk_mask] = np.clip(data[inside_disk_mask], -1, 1)

    # Set the pixels outside the solar disk to gray (0.5 in normalized range [0, 1])
    data[outside_disk_mask] = 0.2
     # Set outside disk pixels to gray (128 in normalized range)

    # Smooth the edge region to blend it seamlessly with the inner disk background
    edge_radius_start = radius_pixels - 30  # Start 30 pixels inward from the edge
    edge_radius_end = radius_pixels  # Outermost pixel of the disk

    # Mask for edge pixels (30-pixel thickness around the circumference)
    edge_mask = (radial_distance >= edge_radius_start) & (radial_distance <= edge_radius_end)

    # Apply Gaussian filter to the edge region for smooth transition
    data[edge_mask] = gaussian_filter(data[edge_mask], sigma=5)  # Apply Gaussian filter for smoother transition

    # Gradually blend the edge with the inner solar disk (smooth transition)
    data[edge_mask] = np.clip(data[edge_mask] + 0.1, -1, 1)  # Lighten edge pixels smoothly

    # Make white patches inside the solar disk whiter (further increase brightness of bright areas)
    # Apply the mask to the entire image and then use the mask to select the pixels
    white_patch_mask = (data >= 0.7)  # Identifying white patches (near the max intensity)
    data[white_patch_mask] = np.clip(data[white_patch_mask] + 0.1, -1, 1)  # Make white patches whiter

    # Darken filaments and sunspots (regions with low intensity)
    filament_mask = (data <= -0.2)  # Identifying darker regions (filaments/sunspots)
    data[filament_mask] = np.clip(data[filament_mask] - 0.1, -1, 1)  # Darken the filaments and sunspots

    return data, center_x, center_y, radius_pixels

# Load FITS file and display the raw image
fits_file = '/content/20110913000054Bh.fits.fz'

with fits.open(fits_file) as hdul:
    data = hdul[1].data

# Plot the raw FITS image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='gray', origin='lower')
#plt.colorbar()  # Add a color bar to show intensity scale
plt.axis('off')
plt.title('Raw FITS Image')

# Apply corrections to the image and get center and radius information
corrected_data, center_x, center_y, radius_pixels = get_data(fits_file)

# Plot the corrected image with the enhanced contrast and smoothened edge
plt.subplot(1, 2, 2)
plt.imshow(corrected_data, cmap='gray', origin='lower')
#plt.colorbar()  # Add a color bar to show intensity scale
plt.title('Corrected FITS image') # image with limb darkening removal &  brightness inhomogeneity removed
plt.axis('off')
# Show the plot
plt.show()

