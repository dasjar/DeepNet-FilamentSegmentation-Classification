import os
import json
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

# Paths
annotations_path = "/content/drive/MyDrive/filament-detection-project/data/magfilo_2024_v1.0.json"
download_dir = '/content/drive/MyDrive/filament-detection-project/data/YOLO-data/gong-fits'  # Directory for FITS files

# Load annotation file
with open(annotations_path, "r") as f:
    coco_annotations = json.load(f)

# Extract image ids, filenames, and URLs from annotations (no deduplication)
image_info = [(img["id"], img["file_name"], img["url"]) for img in coco_annotations["images"]]
print(f"Found {len(image_info)} image entries in annotations.")

def extract_date_from_filename(filename):
    """Extract year, month, and day from the 14-digit timestamp in the filename."""
    match = re.match(r"(\d{4})(\d{2})(\d{2})", filename)
    if match:
        year, month, day = match.groups()
        return year, month, day
    return None, None, None

def download_fits_file_from_dir(url, save_path):
    """Download a FITS file from a directory URL."""
    if os.path.exists(save_path):
        print(f"Already downloaded: {os.path.basename(save_path)}")
        return True

    try:
        # Request the URL (directory)
        print(f"Requesting directory URL: {url}")  # Print the URL being requested
        response = requests.get(url)

        if response.status_code == 200:
            # Parse the directory content using BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")
            # Find the file link (search for the correct .fits.fz file)
            file_link = None
            for link in soup.find_all("a"):
                file_name = link.get("href")
                if file_name and file_name.endswith(".fits.fz"):
                    file_link = url + "/" + file_name  # Build the full URL
                    break

            if file_link:
                # Download the FITS file
                fit_file_response = requests.get(file_link, stream=True)
                total_size = int(fit_file_response.headers.get("content-length", 0))

                with open(save_path, "wb") as file, tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    disable=False
                ) as bar:
                    for chunk in fit_file_response.iter_content(chunk_size=8192):
                        file.write(chunk)
                        bar.update(len(chunk))
                print(f"Downloaded: {os.path.basename(save_path)}")
                return True
            else:
                print(f"Failed to find .fits.fz file in the directory: {url}")
                return False
        else:
            print(f"Failed to access the directory (status code {response.status_code}): {url}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        return False

# Download all images in annotations
print("\nDownloading images from annotation URLs...\n")
total_downloaded = 0

for img_id, file_name, _ in image_info:
    # Extract date from filename to create the URL
    year, month, day = extract_date_from_filename(file_name)
    if not year or not month or not day:
        print(f"Skipping {file_name} due to incorrect format.")
        continue

    # Construct the base URL from the extracted year, month, and day
    base_url = f"https://gong2.nso.edu/HA/haf/{year}{month}/{year}{month}{day}/"

    # Now remove the .jpeg extension and only append .fits.fz for the FITS file download
    fits_file_name = file_name.split(".")[0]  # Remove the file extension (.jpeg)

    # Construct the final URL for the directory
    directory_url = base_url

    # Organize files into Year/Month/Day subdirectories
    sub_dir = os.path.join(download_dir, year, month, day)
    os.makedirs(sub_dir, exist_ok=True)

    # Save the file using the image's ID with .fits.fz extension
    img_save_path = os.path.join(sub_dir, f"{img_id}.fits.fz")  # Using the image ID with .fits.fz extension

    # Try downloading the FITS file from the directory
    if download_fits_file_from_dir(directory_url, img_save_path):
        total_downloaded += 1

print(f"\nDownload complete! Total images downloaded: {total_downloaded}")
