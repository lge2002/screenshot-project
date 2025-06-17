from django.core.management.base import BaseCommand
from django.conf import settings
from weather.models import CloudAnalysis
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from datetime import datetime
import os
import time
import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.ops import unary_union

results_for_json = []

class Command(BaseCommand):
    help = 'Automates screenshot capture from Windy.com, crops to ALL Tamil Nadu districts, masks with shapefile, and analyzes cloud levels.'

    def handle(self, **kwargs):
        self.stdout.write(self.style.SUCCESS('Starting Windy.com cloud analysis automation for all Tamil Nadu districts...'))

        # --- Load Shapefile Once Outside the Main Loop ---
        # This is more efficient as the shapefile is loaded only once per command execution
        shapefile_path = "C:/Users/tamilarasans/Downloads/gadm41_IND_3.json/gadm41_IND_3.json"
        if not os.path.exists(shapefile_path):
            self.stderr.write(self.style.ERROR(f"Critical Error: Shapefile not found at {shapefile_path}. Exiting."))
            return # Exit the command if shapefile is not found

        try:
            gdf = gpd.read_file(shapefile_path)
            # Filter for Tamil Nadu state based on 'NAME_1' field
            tamil_nadu_gdf = gdf[gdf['NAME_1'] == 'TamilNadu']
            if tamil_nadu_gdf.empty:
                self.stderr.write(self.style.ERROR("Error: 'TamilNadu' not found in shapefile under 'NAME_1'. Please check the shapefile content."))
                return # Exit if Tamil Nadu data is missing

            tamil_nadu_gdf = tamil_nadu_gdf.to_crs("EPSG:4326") # Ensure CRS is WGS84 (lat/lon)

            # Dynamically get all unique district names within Tamil Nadu
            # This will get all 38 district names (or however many are in your shapefile)
            all_tn_districts = tamil_nadu_gdf['NAME_2'].unique().tolist()
            if not all_tn_districts:
                self.stderr.write(self.style.ERROR("Error: No districts found for 'TamilNadu' under 'NAME_2' in shapefile. Exiting."))
                return

            self.stdout.write(f"Found {len(all_tn_districts)} districts in Tamil Nadu: {', '.join(all_tn_districts)}")

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error loading or processing shapefile initially: {e}. Exiting."))
            return

        while True:
            current_time = datetime.now()
            timestamp_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')

            # Base folder for this run's outputs
            base_folder = os.path.join(settings.BASE_DIR, "images", timestamp_str)
            full_image_folder = os.path.join(base_folder, "full")
            cropped_image_folder = os.path.join(base_folder, "cropped") # Re-added: Folder for the main Tamil Nadu cropped image
            os.makedirs(full_image_folder, exist_ok=True)
            os.makedirs(cropped_image_folder, exist_ok=True) # Re-added: Create the cropped image folder

            full_screenshot_path = os.path.join(full_image_folder, "windy_map_full.png")
            cropped_screenshot_path = os.path.join(cropped_image_folder, "tamil_nadu_cropped.png") # Re-added: Path for the main TN cropped image

            CROP_BOX = (551, 170, 1065, 687)

            driver = None
            try:
                chrome_options = webdriver.ChromeOptions()
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)

                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
                driver.set_window_size(1920, 1080) # Set a fixed window size for consistent cropping

                driver.get("https://www.windy.com/-Weather-radar-radar?radar,10.950,77.500,7")
                self.stdout.write(f"Navigated to Windy.com with radar layer active and offset coordinates.")

                wait = WebDriverWait(driver, 20) # Increased wait to allow full map load for TN

                try:
                    self.stdout.write('Attempting to dismiss cookie consent...')
                    cookie_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.cc-dismiss')))
                    cookie_button.click()
                    self.stdout.write('Cookie consent dismissed.')
                except Exception as e:
                    self.stdout.write(f"Could not find or dismiss cookie consent (might not be present): {e}. Continuing...")
                    pass

                time.sleep(10) # Give more time for the *entire* TN map to render
                driver.save_screenshot(full_screenshot_path)
                self.stdout.write(f"Full screenshot saved at: {full_screenshot_path}")

            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error during browser automation: {e}"))
                if driver:
                    driver.quit()
                self.stdout.write("Waiting 15 minutes before retry...\n")
                time.sleep(900)
                continue
            finally:
                if driver:
                    driver.quit() # Ensure browser is closed

            try:
                # Crop the main Tamil Nadu area from the full screenshot
                image = Image.open(full_screenshot_path)
                if not (0 <= CROP_BOX[0] < CROP_BOX[2] <= image.width and
                        0 <= CROP_BOX[1] < CROP_BOX[3] <= image.height):
                    self.stderr.write(self.style.ERROR("CROP_BOX coordinates are out of bounds. Skipping all district analysis for this run and waiting."))
                    time.sleep(900)
                    continue

                # The 'cropped_image' below is the base image of Tamil Nadu
                # from which individual districts will be masked.
                cropped_image = image.crop(CROP_BOX)
                cropped_image.save(cropped_screenshot_path) # Re-added: Save the main cropped Tamil Nadu image
                self.stdout.write(f"Cropped Tamil Nadu image saved at: {cropped_screenshot_path}")
                
                # Load cropped image as a NumPy array for processing
                img_pil = cropped_image.convert("RGB")
                img_np = np.array(img_pil)
                height, width, _ = img_np.shape

                final_min_lon = 74.82
                final_max_lon = 80.33
                final_min_lat = 7.97
                final_max_lat = 13.53

                # Define transform for rasterize (same for all districts within this cropped image)
                transform = from_bounds(final_min_lon, final_min_lat, final_max_lon, final_max_lat, width, height)


                # --- Loop through each district for masking and analysis ---
                current_run_results = [] # To store results for JSON output for this specific run

                for district_name in all_tn_districts:
                    self.stdout.write(f"\nProcessing district: {district_name}")

                    # Create district-specific folders for masked images
                    district_masked_folder = os.path.join(base_folder, "masked_cropped", district_name.replace(" ", "_")) # Replace spaces for folder names
                    os.makedirs(district_masked_folder, exist_ok=True)
                    masked_cropped_path = os.path.join(district_masked_folder, f"{district_name.lower().replace(' ', '_')}_masked.png")


                    # Select polygon(s) for the current district
                    current_district_gdf = tamil_nadu_gdf[tamil_nadu_gdf['NAME_2'] == district_name]

                    if current_district_gdf.empty:
                        self.stderr.write(self.style.WARNING(f"Warning: {district_name} not found in the filtered Tamil Nadu shapefile data. Skipping."))
                        continue # Skip to the next district in the list
                    
                    # Combine all geometries for the current district into one polygon
                    district_polygon = unary_union(current_district_gdf.geometry.to_list())

                    # Rasterize polygon to mask
                    mask = rasterize(
                        [district_polygon],
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        all_touched=True,
                        dtype=np.uint8
                    )

                    # Create transparent RGBA image for the current district
                    transparent_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

                    # Apply mask to original cropped image pixels
                    # Convert to RGBA for alpha channel handling during masking
                    original_rgba_image = cropped_image.convert("RGBA")
                    original_rgba_np = np.array(original_rgba_image)

                    for y in range(height):
                        for x in range(width):
                            if mask[y, x]: # If the pixel is within the current district polygon
                                r, g, b, a = original_rgba_np[y, x]
                                if a > 0: # Only copy pixels that are not fully transparent from the original
                                    transparent_image.putpixel((x, y), (r, g, b, 255))

                    transparent_image.save(masked_cropped_path)
                    self.stdout.write(f"Masked image of {district_name} saved at: {masked_cropped_path}")

                    # === Cloud Color Detection on Masked District Image ===
                    windy_legend = {
                        (42, 88, 142): "1.5 mm - Blue", (49, 152, 158): "2 mm - Cyan",
                        (58, 190, 140): "3 mm - Aqua Green", (109, 207, 102): "7 mm - Lime",
                        (192, 222, 72): "10 mm - Yellow Green", (241, 86, 59): "20 mm - Red",
                        (172, 64, 112): "30 mm - Purple"
                    }

                    def match_color(rgb_pixel, tolerance=60):
                        for legend_color_rgb, label in windy_legend.items():
                            diff = sum(abs(rgb_pixel[i] - legend_color_rgb[i]) for i in range(3))
                            if diff <= tolerance:
                                return label
                        return None
                    
                    image_district = transparent_image.convert('RGB')
                    pixels = list(image_district.getdata())
                    unique_pixels = set(pixels)
                    matched_colors = set()

                    for pixel_color in unique_pixels:
                        if pixel_color != (0, 0, 0): # Exclude black (transparent) background from analysis
                            label = match_color(pixel_color)
                            if label:
                                matched_colors.add(label)

                    timestamp_for_db = current_time # Use datetime object for Django model
                    color_text = ", ".join(sorted(matched_colors)) if matched_colors else "No significant cloud levels found for precipitation"
                    self.stdout.write(f"Analysis for {district_name}: {color_text}")

                    # Save results to Django model for the current district
                    try:
                        CloudAnalysis.objects.create(
                            city=district_name, # Use current district name
                            values=color_text,
                            type="Weather radar",
                            # timestamp_collected=timestamp_for_db
                        )
                        self.stdout.write(self.style.SUCCESS(f"Cloud analysis for {district_name} saved to database."))
                    except Exception as e:
                        self.stderr.write(self.style.ERROR(f"Error saving {district_name} to Django model: {e}"))

                    # Append results to the list for JSON output for the current district
                    current_run_results.append({
                        "city": district_name,
                        "values": color_text,
                        "type": "Weather radar",
                        "timestamp": timestamp_for_db.strftime('%Y-%m-%d %H:%M:%S')
                    })

            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error during image processing or shapefile handling for a district: {e}"))
                self.stdout.write("Waiting 15 minutes before retry...\n")
                time.sleep(900)
                continue

            # Save all accumulated results for this run to a single JSON file
            json_output_path = os.path.join(base_folder, "cloud_analysis_results.json")
            try:
                with open(json_output_path, "w") as json_file:
                    json.dump(current_run_results, json_file, indent=4) # Dump results from this run only
                self.stdout.write(f"All analysis results for this run saved to JSON at: {json_output_path}")
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error saving JSON file: {e}"))

            self.stdout.write("\nWaiting 15 minutes before next run...\n")
            time.sleep(900)



            