from django.core.management.base import BaseCommand
from django.conf import settings
from weather.models import CloudAnalysis
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# ADDED: These imports are necessary for the fallback hiding methods
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
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

# ADDED: Import the requests library for making HTTP POST requests
import requests

results_for_json = []

class Command(BaseCommand):
    help = 'Automates screenshot capture from Windy.com, crops to ALL Tamil Nadu districts, masks with shapefile, and analyzes cloud levels.'

    # ADDED: Your confirmed XPath for the blue dot
    BLUE_DOT_XPATH = '//*[@id="leaflet-map"]/div[1]/div[4]/div[2]'

    # UPDATED: Your specific API endpoint URL
    API_ENDPOINT_URL = "http://172.16.7.118:8003/api/tamilnadu/satellite/push.windy_radar_data.php"

    def handle(self, **kwargs):
        self.stdout.write(self.style.SUCCESS('Starting Windy.com cloud analysis automation for all Tamil Nadu districts...'))

        shapefile_path = "C:/Users/tamilarasans/Downloads/gadm41_IND_2.json/gadm41_IND_2.json"
        if not os.path.exists(shapefile_path):
            self.stderr.write(self.style.ERROR(f"Critical Error: Shapefile not found at {shapefile_path}. Exiting."))
            return

        try:
            gdf = gpd.read_file(shapefile_path)
            tamil_nadu_gdf = gdf[gdf['NAME_1'] == 'TamilNadu']
            if tamil_nadu_gdf.empty:
                self.stderr.write(self.style.ERROR("Error: 'TamilNadu' not found in shapefile under 'NAME_1'. Please check the shapefile content."))
                return

            tamil_nadu_gdf = tamil_nadu_gdf.to_crs("EPSG:4326")

            all_tn_districts = tamil_nadu_gdf['NAME_2'].unique().tolist()
            if not all_tn_districts:
                self.stderr.write(self.style.ERROR("Error: No districts found for 'TamilNadu' under 'NAME_2' in shapefile. Exiting."))
                return

            self.stdout.write(f"Found {len(all_tn_districts)} districts in Tamil Nadu: {', '.join(all_tn_districts)}")

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error loading or processing shapefile initially: {e}. Exiting."))
            return

        while True:
            self.stdout.write("\n" + "="*50)
            self.stdout.write("STARTING NEW 15-MINUTE CYCLE: Screenshot and Analysis")
            self.stdout.write("="*50 + "\n")

            current_time = datetime.now()
            timestamp_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')

            base_folder = os.path.join(settings.BASE_DIR, "images", timestamp_str)
            full_image_folder = os.path.join(base_folder, "full")
            cropped_image_folder = os.path.join(base_folder, "cropped")
            os.makedirs(full_image_folder, exist_ok=True)
            os.makedirs(cropped_image_folder, exist_ok=True)

            full_screenshot_path = os.path.join(full_image_folder, "windy_map_full.png")
            cropped_screenshot_path = os.path.join(cropped_image_folder, "tamil_nadu_cropped.png")

            CROP_BOX = (551, 170, 1065, 687)

            driver = None
            current_run_results = []

            try:
                chrome_options = webdriver.ChromeOptions()
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                # ADDED: Ensure window size is set as an argument for consistency
                chrome_options.add_argument("--window-size=1920,1080")

                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
                # REMOVED: driver.set_window_size(1920, 1080) as it's now set via options

                driver.get("https://www.windy.com/-Weather-radar-radar?radar,10.950,77.500,7")
                self.stdout.write(f"Navigated to Windy.com with radar layer active and offset coordinates.")

                wait = WebDriverWait(driver, 20)

                try:
                    self.stdout.write('Attempting to dismiss cookie consent...')
                    cookie_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.cc-dismiss')))
                    cookie_button.click()
                    self.stdout.write('Cookie consent dismissed.')
                    time.sleep(1)
                except Exception as e:
                    self.stdout.write(f"Could not find or dismiss cookie consent (might not be present): {e}. Continuing...")
                    pass

                self.stdout.write("Waiting for map to fully load (10 seconds)...")
                time.sleep(10) # Your original wait

                # --- START OF ADDED BLUE DOT HIDING LOGIC (Selenium methods) ---
                self.stdout.write("Attempting to hide the blue dot using JavaScript injection with confirmed XPath...")

                try:
                    # Line to find the element
                    # Using WebDriverWait for presence, more robust than direct find_element after sleep
                    dot_element = wait.until(EC.presence_of_element_located((By.XPATH, self.BLUE_DOT_XPATH)))

                    # Line to execute JavaScript to change its style
                    driver.execute_script("arguments[0].style.display = 'none';", dot_element)
                    self.stdout.write("SUCCESS (Attempted): Dot element's display set to 'none' via JavaScript using confirmed XPath.")
                    self.stdout.write("NOTE: This method is often ineffective for elements drawn on a canvas, the dot may still be visible in the screenshot.")
                    time.sleep(1)

                except Exception as e:
                    self.stdout.write(f"FAILED to hide dot via JavaScript at XPath '{self.BLUE_DOT_XPATH}': {e}.")
                    self.stdout.write("The element might not be present by this XPath, or another issue occurred. Trying fallback interactive methods (click on map, ESC key)...")

                    # Fallback 1: Click on a general area of the map canvas
                    try:
                        self.stdout.write("Fallback 1: Trying to click on map canvas.")
                        ActionChains(driver).move_by_offset(10, 10).click().perform()
                        self.stdout.write("Clicked on a general page area to dismiss dot.")
                        time.sleep(1)
                    except Exception as click_e:
                        self.stdout.write(f"Fallback 1 (click on map) failed: {click_e}. Trying next method.")

                    # Fallback 2: Press the ESC key
                    try:
                        self.stdout.write("Fallback 2: Trying to press ESC key.")
                        ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                        self.stdout.write("Pressed ESC key to dismiss dot.")
                        time.sleep(1)
                    except Exception as esc_e:
                        self.stdout.write(f"Fallback 2 (ESC key) failed: {esc_e}. The dot might still be visible.")
                # --- END OF ADDED BLUE DOT HIDING LOGIC ---

                # Adding a small wait after hiding attempts before screenshot
                time.sleep(2)

                self.stdout.write(f"Taking full screenshot and saving to: {full_screenshot_path}")
                driver.save_screenshot(full_screenshot_path)
                self.stdout.write("Screenshot saved successfully.")

            except Exception as e:
                self.stderr.write(self.style.ERROR(f"An unexpected error occurred during browser automation: {e}"))
                # No driver.quit() here, it's handled in finally block below
                self.stdout.write("Waiting 15 minutes before retry...\n")
                time.sleep(900)
                continue # Skip image processing and wait for next iteration
            finally:
                if driver:
                    driver.quit()

            # The rest of your image processing and analysis logic
            try:
                image = Image.open(full_screenshot_path).convert("RGB") # Ensure RGB for consistency
                if not (0 <= CROP_BOX[0] < CROP_BOX[2] <= image.width and
                        0 <= CROP_BOX[1] < CROP_BOX[3] <= image.height):
                    self.stderr.write(self.style.ERROR("CROP_BOX coordinates are out of bounds. Skipping all district analysis for this run and waiting."))
                    time.sleep(900)
                    continue

                cropped_image = image.crop(CROP_BOX)
                cropped_image.save(cropped_screenshot_path)
                self.stdout.write(f"Cropped Tamil Nadu image saved at: {cropped_screenshot_path}")

                img_pil = cropped_image.convert("RGB")
                img_np = np.array(img_pil)
                height, width, _ = img_np.shape

                final_min_lon = 74.80
                final_max_lon = 80.37
                final_min_lat = 7.98
                final_max_lat = 13.53

                transform = from_bounds(final_min_lon, final_min_lat, final_max_lon, final_max_lat, width, height)

                for district_name in all_tn_districts:
                    self.stdout.write(f"\nProcessing district: {district_name}")

                    district_masked_folder = os.path.join(base_folder, "masked_cropped", district_name.replace(" ", "_"))
                    os.makedirs(district_masked_folder, exist_ok=True)
                    masked_cropped_path = os.path.join(district_masked_folder, f"{district_name.lower().replace(' ', '_')}_masked.png")

                    current_district_gdf = tamil_nadu_gdf[tamil_nadu_gdf['NAME_2'] == district_name]

                    if current_district_gdf.empty:
                        self.stderr.write(self.style.WARNING(f"Warning: {district_name} not found in the filtered Tamil Nadu shapefile data. Skipping."))
                        continue

                    district_polygon = unary_union(current_district_gdf.geometry.to_list())

                    mask = rasterize(
                        [district_polygon],
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        all_touched=True,
                        dtype=np.uint8
                    )

                    # Create a blank RGBA image to paste only the masked pixels
                    transparent_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

                    # Convert cropped_image to RGBA to get alpha channel for accurate pixel copying
                    original_rgba_image = cropped_image.convert("RGBA")
                    original_rgba_np = np.array(original_rgba_image)

                    # Iterate over pixels within the mask and copy them to the new transparent image
                    for y in range(height):
                        for x in range(width):
                            if mask[y, x]: # If this pixel is part of the district shape
                                r, g, b, a = original_rgba_np[y, x]
                                if a > 0: # Only copy if the original pixel isn't fully transparent
                                    transparent_image.putpixel((x, y), (r, g, b, 255)) # Make it fully opaque in the masked output

                    transparent_image.save(masked_cropped_path)
                    self.stdout.write(f"Masked image of {district_name} saved at: {masked_cropped_path}")

                    # The Windy legend colors in RGB format
                    windy_legend = {
                        (42, 88, 142): "1.5 mm - Blue",
                        (49, 152, 158): "2 mm - Cyan",
                        (58, 190, 140): "3 mm - Aqua Green",
                        (109, 207, 102): "7 mm - Lime",
                        (192, 222, 72): "10 mm - Yellow Green",
                        (241, 86, 59): "20 mm - Red",
                        (172, 64, 112): "30 mm - Purple"
                    }

                    def match_color_robust(rgb_pixel, legend, max_tolerance=60):
                        """
                        Finds the best matching color from the legend for a given pixel,
                        returning the label of the closest match within tolerance.
                        Uses Euclidean distance for better color similarity measurement.
                        """
                        best_match_label = None
                        min_distance = float('inf')

                        for legend_color_rgb, label in legend.items():
                            # Calculate Euclidean distance in RGB space
                            distance = ((rgb_pixel[0] - legend_color_rgb[0])**2 +
                                        (rgb_pixel[1] - legend_color_rgb[1])**2 +
                                        (rgb_pixel[2] - legend_color_rgb[2])**2)**0.5

                            if distance <= max_tolerance and distance < min_distance:
                                min_distance = distance
                                best_match_label = label
                        return best_match_label

                    image_district_for_analysis = transparent_image.convert('RGB')
                    pixels_to_analyze = list(image_district_for_analysis.getdata())
                    
                    # Use a set to collect unique labels found in the district
                    matched_colors = set()

                    # Iterate through all pixels in the masked image (which are now either actual map data or black from transparency)
                    for pixel_color in pixels_to_analyze:
                        # Exclude pure black pixels, which represent the transparent background outside the mask
                        if pixel_color != (0, 0, 0):
                            label = match_color_robust(pixel_color, windy_legend, max_tolerance=60) # You can adjust this tolerance
                            if label:
                                matched_colors.add(label)

                    timestamp_for_db = current_time
                    # Format the output for the 'values' field: comma-separated and sorted
                    color_text = ", ".join(sorted(matched_colors)) if matched_colors else "No significant cloud levels found for precipitation"
                    self.stdout.write(f"Analysis for {district_name}: {color_text}")

                    try:
                        CloudAnalysis.objects.create(
                            city=district_name,
                            values=color_text,
                            type="Weather radar",
                        )
                        self.stdout.write(self.style.SUCCESS(f"Cloud analysis for {district_name} saved to database."))
                    except Exception as e:
                        self.stderr.write(self.style.ERROR(f"Error saving {district_name} to Django model: {e}"))

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

            json_output_path = os.path.join(base_folder, "cloud_analysis_results.json")
            try:
                with open(json_output_path, "w") as json_file:
                    json.dump(current_run_results, json_file, indent=4)
                self.stdout.write(f"All analysis results for this run saved to JSON at: {json_output_path}")
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error saving JSON file: {e}"))

            # --- ADDED: POST method to send data to the API endpoint ---
            # Using the provided URL
            if self.API_ENDPOINT_URL and current_run_results:
                self.stdout.write(f"Attempting to send analysis data to {self.API_ENDPOINT_URL} via POST...")
                headers = {
                    'Content-Type': 'application/json',
                    # Add any required authentication headers here if the PHP script needs them.
                    # For example:
                    # 'Authorization': 'Bearer YOUR_AUTH_TOKEN',
                    # 'X-API-Key': 'YOUR_API_KEY',
                }

                try:
                    response = requests.post(self.API_ENDPOINT_URL, json=current_run_results, headers=headers)
                    response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

                    self.stdout.write(self.style.SUCCESS(f"Data successfully POSTed to {self.API_ENDPOINT_URL}."))
                    self.stdout.write(f"API Response Status Code: {response.status_code}")
                    try:
                        self.stdout.write(f"API Response JSON: {response.json()}")
                    except json.JSONDecodeError:
                        self.stdout.write(f"API Response Text: {response.text}")
                except requests.exceptions.HTTPError as http_err:
                    self.stderr.write(self.style.ERROR(f"HTTP error during POST request: {http_err}"))
                    self.stderr.write(self.style.ERROR(f"Response from API: {http_err.response.text}"))
                except requests.exceptions.ConnectionError as conn_err:
                    self.stderr.write(self.style.ERROR(f"Connection error during POST request (Is the server at {self.API_ENDPOINT_URL} reachable?): {conn_err}"))
                except requests.exceptions.Timeout as timeout_err:
                    self.stderr.write(self.style.ERROR(f"Timeout error during POST request: {timeout_err}"))
                except requests.exceptions.RequestException as req_err:
                    self.stderr.write(self.style.ERROR(f"An unexpected error occurred during POST request: {req_err}"))
            else:
                if not self.API_ENDPOINT_URL:
                    self.stdout.write(self.style.WARNING("API_ENDPOINT_URL is not set. Skipping POST request."))
                if not current_run_results:
                    self.stdout.write(self.style.WARNING("No analysis results to send. Skipping POST request."))
            # --- END OF ADDED POST method ---

            self.stdout.write("\nWaiting 15 minutes before next run...\n")
            time.sleep(900)