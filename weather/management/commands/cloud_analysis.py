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

results_for_json = []

class Command(BaseCommand):
    def handle(self, **kwargs):
        while True:
            current_time = datetime.now()
            timestamp_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')

            base_folder = os.path.join(settings.BASE_DIR, "images", timestamp_str)
            full_image_folder = os.path.join(base_folder, "full")
            cropped_image_folder = os.path.join(base_folder, "cropped")
            masked_cropped_folder = os.path.join(base_folder, "masked_cropped")
            os.makedirs(full_image_folder, exist_ok=True)
            os.makedirs(cropped_image_folder, exist_ok=True)
            os.makedirs(masked_cropped_folder, exist_ok=True)

            full_screenshot_path = os.path.join(full_image_folder, "windy_map_full.png")
            cropped_screenshot_path = os.path.join(cropped_image_folder, "windy_map_cropped.png")
            masked_cropped_path = os.path.join(masked_cropped_folder, "erode_masked.png")

            # Define Tamil Nadu crop region (based on screen size and Windy position)
            CROP_BOX = (551, 170, 1065, 687)

            driver = None
            try:
                chrome_options = webdriver.ChromeOptions()
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)

                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
                driver.set_window_size(1920, 1080)
                driver.get("https://www.windy.com/?10.936,77.487,7")

                wait = WebDriverWait(driver, 12)
                try:
                    cookie_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.cc-dismiss')))
                    cookie_button.click()
                except:
                    pass

                try:
                    cloud_button = wait.until(EC.element_to_be_clickable(
                        (By.XPATH, '//*[@id="plugin-rhpane-top"]/section/section/a[1]/div[2]')
                    ))
                    cloud_button.click()
                except Exception as e:
                    print("Failed to click cloud layer button:", e)

                time.sleep(5)
                driver.save_screenshot(full_screenshot_path)
                print("Full screenshot saved at:", full_screenshot_path)

            except Exception as e:
                print(f"Error during browser automation: {e}")
                if driver:
                    driver.quit()
                time.sleep(900)  # Wait 15 min before retry
                continue

            if driver:
                driver.quit()

            # Crop Tamil Nadu area
            image = Image.open(full_screenshot_path)
            if not (0 <= CROP_BOX[0] < CROP_BOX[2] <= image.width and
                    0 <= CROP_BOX[1] < CROP_BOX[3] <= image.height):
                print("CROP_BOX out of bounds. Skipping crop.")
                time.sleep(900)
                continue
            cropped_image = image.crop(CROP_BOX)
            cropped_image.save(cropped_screenshot_path)
            print(f"Cropped image saved at: {cropped_screenshot_path}")

            # Load Tamil Nadu cropped image as numpy
            img_pil = cropped_image.convert("RGB")
            img_np = np.array(img_pil)
            height, width, _ = img_np.shape

            # Geo info - Tamil Nadu bounds of the cropped image (you must set these carefully)
            final_min_lon = 74.65
            final_max_lon = 80.45
            final_min_lat = 7.99
            final_max_lat = 13.55

            # Load shapefile - adjust this path to your shapefile location
            shapefile_path = "C:/Users/tamilarasans/Downloads/gadm41_IND_3.json/gadm41_IND_3.json"
            gdf = gpd.read_file(shapefile_path)
            gdf = gdf[gdf['NAME_1'] == 'TamilNadu']
            gdf = gdf.to_crs("EPSG:4326")

            # Select Erode polygon(s)
            district_name = "Coimbatore"
            erode_gdf = gdf[gdf['NAME_2'] == district_name]

            if erode_gdf.empty:
                print(f"Erode district not found in shapefile.")
            else:
                from shapely.ops import unary_union
                # Combine all geometries for Erode into one polygon
                erode_polygon = unary_union(erode_gdf.geometry.to_list())

                # Define transform for rasterize
                transform = from_bounds(final_min_lon, final_min_lat, final_max_lon, final_max_lat, width, height)

                # Rasterize polygon to mask
                mask = rasterize(
                    [erode_polygon],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8
                )

                # Create transparent RGBA image to hold masked pixels
                from PIL import Image as PILImage
                transparent_image = PILImage.new("RGBA", (width, height), (0, 0, 0, 0))

                # Apply mask to original cropped image pixels
                for y in range(height):
                    for x in range(width):
                        if mask[y, x]:
                            r, g, b = img_np[y, x]
                            transparent_image.putpixel((x, y), (r, g, b, 255))

                transparent_image.save(masked_cropped_path)
                print(f"Masked image of Erode saved at: {masked_cropped_path}")

            # === Cloud Color Detection on Tamil Nadu Cropped Image ===
            windy_legend = {
                (42, 88, 142): "1.5 mm - Blue",
                (49, 152, 158): "2 mm - Cyan",
                (58, 190, 140): "3 mm - Aqua Green",
                (109, 207, 102): "7 mm - Lime",
                (192, 222, 72): "10 mm - Yellow Green",
                (241, 86, 59): "20 mm - Red",
                (172, 64, 112): "30 mm - Purple"
            }

            def match_color(rgb_pixel, tolerance=60):
                for legend_color in windy_legend:
                    diff = sum(abs(rgb_pixel[i] - legend_color[i]) for i in range(3))
                    if diff <= tolerance:
                        return windy_legend[legend_color]
                return None
            
            image_city = transparent_image.convert('RGB')  # Use masked transparent image instead
            pixels = list(image_city.getdata())
            unique_pixels = set(pixels)
            matched_colors = set()

            for pixel_color in unique_pixels:
                label = match_color(pixel_color)
                if label:
                    matched_colors.add(label)


            timestamp_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            city = "Coimbatore"
            color_text = ", ".join(sorted(matched_colors)) if matched_colors else "No Windy legend colors found"
            print(f"{city}: {color_text}")

            CloudAnalysis.objects.create(
                city=city,
                values=color_text,
                type="Weather radar"
            )

            results_for_json.append({
                "city": city,
                "values": color_text,
                "type": "Weather radar",
                "timestamp": timestamp_now
            })

            json_output_path = os.path.join(base_folder, "cloud_analysis_results.json")
            with open(json_output_path, "w") as json_file:
                json.dump(results_for_json, json_file, indent=4)

            print(f"\nJSON saved at: {json_output_path}")
            print("Waiting 15 minutes before next run...\n")
            time.sleep(900)
