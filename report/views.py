# report/views.py

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template.loader import render_to_string # Used for rendering HTML for Playwright
import os
from django.conf import settings
from datetime import datetime, timedelta, date, time
import pytz

# --- Playwright Imports for PDF generation ---
from playwright.sync_api import sync_playwright
# -------------------------------------------

# --- Import your actual CloudAnalysis model ---
from weather.models import CloudAnalysis

# --- Image Processing Imports ---
import geopandas as gpd
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from sklearn.cluster import KMeans
from shapely.ops import unary_union
import warnings
import io
import base64
import shutil # For cleaning up directories

warnings.filterwarnings("ignore")

# --- GLOBAL CONFIGURATION FOR IMAGE GENERATION (MUST MATCH YOUR SETUP) ---
SHAPEFILE_PATH = os.path.join(settings.BASE_DIR, 'weather', 'management', 'commands', 'gadm41_IND_2.json')

# These are the alignment values for the FULL TN map with overlay.
FINAL_MIN_LON = 74.80
FINAL_MAX_LON = 80.37
FINAL_MIN_LAT = 7.98
FINAL_MAX_LAT = 13.53
# -------------------------------------------------------------------------


# --- HELPER FUNCTION: Saves a PIL image and returns its ABSOLUTE web URL ---
def save_image_and_get_url(request, image_pil, report_base_dir, timestamp_dt, image_type_name):
    """
    Saves a PIL image to a specific subfolder structure within the report's directory
    and returns its ABSOLUTE URL.

    Args:
        request (HttpRequest): The Django request object, needed to build absolute URI.
        image_pil (PIL.Image.Image): The PIL image object to save.
        report_base_dir (str): The full path to the base directory for this report
                                (e.g., settings.MEDIA_ROOT/report_images/2025-06-27_Coimbatore_10-00-10-15)
        timestamp_dt (datetime): The timestamp associated with this image.
        image_type_name (str): A descriptive name for the image type (e.g., 'cropped_tn').

    Returns:
        str: The ABSOLUTE URL to access the saved image from the web.
    """
    # Create a unique subfolder based on the image's timestamp within the report folder
    image_timestamp_folder = timestamp_dt.strftime('%H-%M-%S') # Only time for subfolder

    # Construct the full path where the image will be saved on the server
    save_dir = os.path.join(report_base_dir, image_timestamp_folder)
    os.makedirs(save_dir, exist_ok=True) # Ensure directory exists

    # Define the filename within that timestamp folder
    file_name = f"{image_type_name}.png"
    full_path = os.path.join(save_dir, file_name)

    try:
        # Save the image
        image_pil.save(full_path, format="PNG")

        # Construct the ABSOLUTE URL for the browser
        # This path must be relative to settings.MEDIA_URL first, then converted to absolute
        relative_path_from_media_root = os.path.relpath(full_path, settings.MEDIA_ROOT)
        # Use request.build_absolute_uri to get the full http://... URL
        # IMPORTANT: Replace os.sep with '/' for URL paths
        image_url = request.build_absolute_uri(os.path.join(settings.MEDIA_URL, relative_path_from_media_root).replace(os.sep, '/'))
        return image_url
    except Exception as e:
        print(f"Error saving image {full_path}: {e}")
        return None


# --- HELPER FUNCTION: Encapsulates core image generation logic to return PIL images ---
def _generate_image_data_for_timestamp(
    base_image_path_for_this_timestamp, timestamp_dt, selected_district, gdf_tn
):
    """
    Generates PIL Image objects for different views (cropped, masked, overlay).
    """
    try:
        if not os.path.exists(base_image_path_for_this_timestamp):
            print(f"Warning: Base image '{base_image_path_for_this_timestamp}' not found. Skipping image processing for this timestamp.")
            return None

        img_pil = Image.open(base_image_path_for_this_timestamp).convert("RGB")
        img_np = np.array(img_pil)
        height, width, _ = img_np.shape
        transform = from_bounds(FINAL_MIN_LON, FINAL_MAX_LON, FINAL_MIN_LAT, FINAL_MAX_LAT, width, height)

        output_images = {
            'cropped_tn': img_pil, # Directly return the PIL image
            'masked_district': None,
            'aligned_overlay_tn': None,
        }

        # 1. Masked District Image
        if selected_district == 'All Districts' or gdf_tn.empty:
            output_images['masked_district'] = img_pil # If no specific district, use full cropped
        else:
            district_rows_for_name = gdf_tn[gdf_tn['NAME_2'].str.lower() == selected_district.lower()]
            if not district_rows_for_name.empty:
                all_district_geometries = district_rows_for_name.geometry.to_list()
                district_polygon_for_mask = unary_union(all_district_geometries)
                
                mask = rasterize(
                    [district_polygon_for_mask],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8
                )
                mask_boolean = mask.astype(bool)
                cropped_district_img_np = np.zeros_like(img_np)
                cropped_district_img_np[mask_boolean] = img_np[mask_boolean]
                output_images['masked_district'] = Image.fromarray(cropped_district_img_np)
            else:
                print(f"Warning: District '{selected_district}' not found in shapefile for masked image generation at {timestamp_dt}.")
                output_images['masked_district'] = None 

        # 2. Overall TN Map with District Outlines (and highlighted district)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_np, extent=[FINAL_MIN_LON, FINAL_MAX_LON, FINAL_MIN_LAT, FINAL_MAX_LAT])
        gdf_tn.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

        if selected_district != 'All Districts':
            district_rows_for_name_for_highlight = gdf_tn[gdf_tn['NAME_2'].str.lower() == selected_district.lower()]
            if not district_rows_for_name_for_highlight.empty:
                district_rows_for_name_for_highlight.boundary.plot(ax=ax, edgecolor='cyan', linewidth=2, linestyle='--', label=selected_district)
                ax.set_title(f"Aligned Screenshot with {selected_district} Highlighted ({timestamp_dt.strftime('%H:%M')})")
                ax.legend()
            else:
                ax.set_title(f"Aligned Screenshot (District '{selected_district}' not found for highlight) ({timestamp_dt.strftime('%H:%M')})")
        else:
            ax.set_title(f"Aligned Screenshot with All TN District Outlines ({timestamp_dt.strftime('%H:%M')})")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal')
        plt.tight_layout()

        # Save matplotlib figure to a BytesIO object, then open with PIL
        buffer = io.BytesIO()
        plt.savefig(buffer, format="PNG", bbox_inches='tight', pad_inches=0.1)
        buffer.seek(0)
        output_images['aligned_overlay_tn'] = Image.open(buffer).convert("RGB")
        
        buffer.close()
        plt.close(fig) # Always close the figure!

        return output_images

    except FileNotFoundError:
        print(f"Warning: Base map image '{base_image_path_for_this_timestamp}' not found for {timestamp_dt}. Skipping generation for this timestamp.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during image generation for {timestamp_dt}: {e}")
        return None
    finally:
        # Ensure the figure is always closed if created, even if an error occurs
        if 'fig' in locals() and fig:
            plt.close(fig)

# --- Main view for displaying the report in the browser ---
def report_view(request):
    selected_date_str = request.GET.get('date')
    selected_district = request.GET.get('district', 'All Districts')
    if selected_district == "" or selected_district is None:
        selected_district = "All Districts"

    start_time_hour_str = request.GET.get('start_time_hour')
    start_time_minute_str = request.GET.get('start_time_minute')
    end_time_hour_str = request.GET.get('end_time_hour')
    end_time_minute_str = request.GET.get('end_time_minute')

    selected_image_view = request.GET.get('image_view_type', '')

    filter_date = None
    if selected_date_str:
        try:
            filter_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"Warning: Invalid date format '{selected_date_str}'. Defaulting to today's date.")
            filter_date = date.today()
    else:
        filter_date = date.today()
    
    target_date_display_str = filter_date.strftime('%Y-%m-%d')

    selected_start_time_for_template = ''
    selected_end_time_for_template = ''

    filter_start_datetime = None
    filter_end_datetime = None

    current_timezone = pytz.timezone(settings.TIME_ZONE) if settings.USE_TZ else None

    if not start_time_hour_str or not start_time_minute_str:
        now = datetime.now()
        current_minute = now.minute
        current_hour = now.hour

        from_minute = (current_minute // 15) * 15
        if current_minute % 15 != 0:
            from_minute = ((current_minute // 15) + 1) * 15
            if from_minute == 60:
                from_minute = 0
                current_hour = (current_hour + 1) % 24

        from_hour = current_hour
        
        start_time_obj = time(from_hour, from_minute)
        selected_start_time_for_template = start_time_obj.strftime("%H:%M")
        filter_start_datetime = datetime.combine(filter_date, start_time_obj)

        to_minute = from_minute + 15
        to_hour = from_hour
        if to_minute == 60:
            to_minute = 0
            to_hour = (to_hour + 1) % 24
        
        end_time_obj = time(to_hour, to_minute)
        selected_end_time_for_template = end_time_obj.strftime("%H:%M")
        filter_end_datetime = datetime.combine(filter_date, end_time_obj)

    else:
        try:
            start_hour = int(start_time_hour_str)
            start_minute = int(start_time_minute_str)
            end_hour = int(end_time_hour_str)
            end_minute = int(end_time_minute_str)

            start_time_obj = time(start_hour, start_minute)
            end_time_obj = time(end_hour, end_minute)

            selected_start_time_for_template = start_time_obj.strftime("%H:%M")
            selected_end_time_for_template = end_time_obj.strftime("%H:%M")

            filter_start_datetime = datetime.combine(filter_date, start_time_obj)
            filter_end_datetime = datetime.combine(filter_date, end_time_obj)

        except ValueError as e:
            print(f"Error parsing time parameters: {e}. Defaulting to full day.")
            filter_start_datetime = datetime.combine(filter_date, time(0, 0, 0))
            filter_end_datetime = datetime.combine(filter_date, time(23, 59, 59, 999999))
            selected_start_time_for_template = "00:00"
            selected_end_time_for_template = "23:59"


    if current_timezone:
        if filter_start_datetime:
            filter_start_datetime = current_timezone.localize(filter_start_datetime)
        if filter_end_datetime:
            filter_end_datetime = current_timezone.localize(filter_end_datetime)


    print(f"\n--- Generating Images for Date: {target_date_display_str}, District: {selected_district}, Image View: {selected_image_view} ---")
    print(f"Time Range Filter (Backend): {filter_start_datetime} - {filter_end_datetime}")
    print(f"SHAPEFILE_PATH configured as: {SHAPEFILE_PATH}")

    generated_images_for_display = [] 

    gdf = None
    gdf_tn = None

    try:
        if not os.path.exists(SHAPEFILE_PATH):
            raise FileNotFoundError(f"Shapefile not found at {SHAPEFILE_PATH}.")
        gdf = gpd.read_file(SHAPEFILE_PATH)
        gdf_tn = gdf[
            (gdf['NAME_1'].str.strip().str.lower() == 'tamilnadu') |
            (gdf['NAME_1'].str.strip().str.lower() == 'tamil nadu')
        ].to_crs("EPSG:4326")
        print(f"Shapefile loaded successfully.")
    except FileNotFoundError as fnfe:
        print(f"CRITICAL ERROR: Shapefile (for generation) not found: {fnfe}")
    except Exception as e:
        print(f"ERROR loading shapefile: {e}")


    if os.path.exists(settings.MEDIA_ROOT) and os.path.isdir(settings.MEDIA_ROOT) and gdf_tn is not None:
        available_image_timestamps_and_folders = []
        for d in os.listdir(settings.MEDIA_ROOT): 
            full_path_to_folder = os.path.join(settings.MEDIA_ROOT, d)
            if os.path.isdir(full_path_to_folder):
                try:
                    folder_timestamp_str = d 
                    folder_datetime_obj = datetime.strptime(folder_timestamp_str, '%Y-%m-%d_%H-%M-%S')
                    
                    if current_timezone:
                        folder_datetime_obj = current_timezone.localize(folder_datetime_obj)

                    if folder_datetime_obj.date() == filter_date and \
                       filter_start_datetime <= folder_datetime_obj < filter_end_datetime:
                        available_image_timestamps_and_folders.append((folder_datetime_obj, full_path_to_folder))
                except (ValueError, IndexError):
                    pass 
        
        available_image_timestamps_and_folders.sort(key=lambda x: x[0])

        print(f"Found {len(available_image_timestamps_and_folders)} image folders within the selected time range for {target_date_display_str}.")

        for timestamp_dt, folder_path in available_image_timestamps_and_folders:
            base_image_path_for_this_timestamp = os.path.join(folder_path, 'cropped', 'tamil_nadu_cropped.png')
            
            # Call the helper to get PIL image data
            pil_images = _generate_image_data_for_timestamp(
                base_image_path_for_this_timestamp, timestamp_dt, selected_district, gdf_tn
            )

            if pil_images: 
                current_image_set = {
                    'timestamp': timestamp_dt,
                    'cropped_tn': None,
                    'masked_district': None,
                    'aligned_overlay_tn': None,
                }

                # Convert PIL images to Base64 for browser display (This view uses Base64)
                for img_type, pil_img in pil_images.items():
                    if pil_img:
                        buffer = io.BytesIO()
                        pil_img.save(buffer, format="PNG")
                        current_image_set[img_type] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
                        buffer.close()
                    elif img_type == 'masked_district' and selected_district == 'All Districts':
                        current_image_set['masked_district'] = current_image_set['cropped_tn']
                    else:
                        current_image_set[img_type] = None 

                generated_images_for_display.append(current_image_set)
    else:
        print(f"Image generation skipped for browser display: Base media directory or shapefile not loaded successfully.")

    # --- Filtering CloudAnalysis data ---
    cloud_analysis_query = CloudAnalysis.objects.filter(
        timestamp__date=filter_date 
    )

    if selected_district != 'All Districts':
        cloud_analysis_query = cloud_analysis_query.filter(city__iexact=selected_district)

    if filter_start_datetime and filter_end_datetime:
        cloud_analysis_query = cloud_analysis_query.filter(
            timestamp__gte=filter_start_datetime,
            timestamp__lt=filter_end_datetime 
        )
    
    EXCLUDE_NO_PRECIP_MESSAGE = "No significant cloud levels found for precipitation"
    cloud_analysis_query = cloud_analysis_query.exclude(values__iexact=EXCLUDE_NO_PRECIP_MESSAGE)

    filtered_cloud_analysis_data = list(cloud_analysis_query.order_by('city', 'timestamp'))
    
    print(f"Fetched {len(filtered_cloud_analysis_data)} weather data points for {target_date_display_str} and {selected_district} (excluding 'no precipitation' values).")

    full_available_districts = []
    try:
        if not os.path.exists(SHAPEFILE_PATH):
            print(f"ERROR: Shapefile for district list not found at {SHAPEFILE_PATH}. Falling back to default list.")
            full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']
        else:
            temp_gdf = gpd.read_file(SHAPEFILE_PATH)
            gdf_tn_districts_for_list = temp_gdf[
                (temp_gdf['NAME_1'].str.strip().str.lower() == 'tamilnadu') |
                (temp_gdf['NAME_1'].str.strip().str.lower() == 'tamil nadu')
            ]
            
            if 'NAME_2' in gdf_tn_districts_for_list.columns:
                unique_districts = gdf_tn_districts_for_list['NAME_2'].dropna().unique().tolist()
                full_available_districts = sorted(unique_districts)
                print(f"Dynamically loaded {len(full_available_districts)} districts from shapefile.")
            else:
                print("Warning: 'NAME_2' column not found in shapefile for district extraction. Falling back to default list.")
                full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']

    except Exception as e:
        print(f"Error loading districts from shapefile for dropdown: {e}. Falling back to default list.")
        full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']


    context = {
        'generated_images_for_display': generated_images_for_display, 
        'selected_date': filter_date.strftime('%Y-%m-%d'),
        'selected_district': selected_district,
        'available_districts': full_available_districts,
        'cloud_analysis_data': filtered_cloud_analysis_data, 
        'selected_image_view': selected_image_view,
        'selected_start_time': selected_start_time_for_template,
        'selected_end_time': selected_end_time_for_template,
    }
    return render(request, 'report/report.html', context)

def save_report_images_for_timestamp(timestamp_dt, report_specific_media_dir, selected_district=None, radar_data_for_plot=None):
    image_paths = {}

    timestamp_folder_name = timestamp_dt.strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(report_specific_media_dir, timestamp_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Generate/Save Cropped TN Image (Radar Only) ---
    # Assuming the source for this is a fixed file or generated separately.
    # Replace this with your actual generation logic if it's not just loading a file.
    cropped_tn_source_path = os.path.join(settings.MEDIA_ROOT, 'radar_data_base_folders', timestamp_folder_name, "tamil_nadu_cropped.png") 
    # ^^^ ADJUST THIS SOURCE PATH to where your raw radar images are stored
    
    if os.path.exists(cropped_tn_source_path):
        try:
            pil_cropped_tn = Image.open(cropped_tn_source_path).copy()
            output_file_name = "cropped_tn.png"
            output_path = os.path.join(output_dir, output_file_name)
            pil_cropped_tn.save(output_path, format="PNG")
            image_paths['cropped_tn'] = os.path.relpath(output_path, settings.MEDIA_ROOT)
            print(f"Saved cropped_tn: {image_paths['cropped_tn']}") # Debug
        except Exception as e:
            print(f"Error saving cropped_tn from {cropped_tn_source_path}: {e}")
    else:
        print(f"DEBUG: Source cropped_tn.png not found at {cropped_tn_source_path}")


    # --- 2. Generate/Save Masked District Image (if needed, not part of combined_full_tn directly) ---
    # (Leaving this in for completeness, though not strictly for 'combined_full_tn')
    if selected_district:
        masked_district_source_path = os.path.join(settings.MEDIA_ROOT, 'radar_data_base_folders', timestamp_folder_name, f"{selected_district.replace(' ', '_')}_masked.png")
        if os.path.exists(masked_district_source_path):
            try:
                pil_masked_district = Image.open(masked_district_source_path).copy()
                output_file_name = "masked_district.png"
                output_path = os.path.join(output_dir, output_file_name)
                pil_masked_district.save(output_path, format="PNG")
                image_paths['masked_district'] = os.path.relpath(output_path, settings.MEDIA_ROOT)
                print(f"Saved masked_district: {image_paths['masked_district']}") # Debug
            except Exception as e:
                print(f"Error saving masked_district from {masked_district_source_path}: {e}")
        else:
            print(f"DEBUG: Source masked_district.png not found at {masked_district_source_path}")

    base_map_path = os.path.join(settings.BASE_DIR, 'static', 'radar_report', 'tamil_nadu_base_map_with_districts.png') 
    # ^^^ Make sure this path points to your actual base map image file
    
    if os.path.exists(base_map_path): # Check if base map exists
        try:
            # Load the base map
            base_map_pil = Image.open(base_map_path).convert("RGBA") # Ensure RGBA for transparency if needed

            # --- Your ACTUAL Matplotlib/PIL overlay logic goes here ---
            # Example (pseudo-code, replace with your real plotting logic):
            if radar_data_for_plot: # Only if you have radar data to overlay
                fig, ax = plt.subplots(figsize=(base_map_pil.width / 100, base_map_pil.height / 100), dpi=100) # Match base map dimensions
                ax.imshow(base_map_pil)

                ax.axis('off') # Hide axes
                ax.set_position([0, 0, 1, 1]) # Make plot fill the figure
                
                buf = io.BytesIO()
                plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0)
                plt.close(fig) # Close the plot to free memory
                buf.seek(0)
                pil_tn_overlay = Image.open(buf).convert("RGBA") # Convert to RGBA after plotting

            else: # If no radar data, just use the base map as the "overlay" image
                pil_tn_overlay = base_map_pil.copy()

            output_file_name = "tn_overlay.png" # You can call it aligned_overlay_tn.png if you prefer
            output_path = os.path.join(output_dir, output_file_name)
            pil_tn_overlay.save(output_path, format="PNG")
            image_paths['aligned_overlay_tn'] = os.path.relpath(output_path, settings.MEDIA_ROOT)
            print(f"Saved tn_overlay: {image_paths['aligned_overlay_tn']}") # Debug

        except Exception as e:
            print(f"Error generating or saving tn_overlay: {e}")
    else:
        print(f"DEBUG: Base map for overlay not found at {base_map_path}")

    return image_paths


# --- RESTORED AND MODIFIED VIEW: For processing, IMAGE SAVING, and PDF GENERATION ---
def download_report_pdf(request):
    selected_date_str = request.GET.get('date')
    selected_district = request.GET.get('district', 'All Districts')
    if selected_district == "" or selected_district is None:
        selected_district = "All Districts"

    start_time_hour_str = request.GET.get('start_time_hour')
    start_time_minute_str = request.GET.get('start_time_minute')
    end_time_hour_str = request.GET.get('end_time_hour')
    end_time_minute_str = request.GET.get('end_time_minute')

    selected_image_view = request.GET.get('image_view_type', '') 

    filter_date = None
    if selected_date_str:
        try:
            filter_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"Warning: Invalid date format '{selected_date_str}'. Defaulting to today's date.")
            filter_date = date.today()
    else:
        filter_date = date.today()
    
    target_date_display_str = filter_date.strftime('%Y-%m-%d')

    selected_start_time_for_template = ''
    selected_end_time_for_template = ''

    filter_start_datetime = None
    filter_end_datetime = None

    current_timezone = pytz.timezone(settings.TIME_ZONE) if settings.USE_TZ else None

    if not start_time_hour_str or not start_time_minute_str:
        now = datetime.now()
        current_minute = now.minute
        current_hour = now.hour

        from_minute = (current_minute // 15) * 15
        if current_minute % 15 != 0:
            from_minute = ((current_minute // 15) + 1) * 15
            if from_minute == 60:
                from_minute = 0
                current_hour = (current_hour + 1) % 24

        from_hour = current_hour
        
        start_time_obj = time(from_hour, from_minute)
        selected_start_time_for_template = start_time_obj.strftime("%H:%M")
        filter_start_datetime = datetime.combine(filter_date, start_time_obj)

        to_minute = from_minute + 15
        to_hour = from_hour
        if to_minute == 60:
            to_minute = 0
            to_hour = (to_hour + 1) % 24
        
        end_time_obj = time(to_hour, to_minute)
        selected_end_time_for_template = end_time_obj.strftime("%H:%M")
        filter_end_datetime = datetime.combine(filter_date, end_time_obj)

    else:
        try:
            start_hour = int(start_time_hour_str)
            start_minute = int(start_time_minute_str)
            end_hour = int(end_time_hour_str)
            end_minute = int(end_time_minute_str)

            start_time_obj = time(start_hour, start_minute)
            end_time_obj = time(end_hour, end_minute)

            selected_start_time_for_template = start_time_obj.strftime("%H:%M")
            selected_end_time_for_template = end_time_obj.strftime("%H:%M")

            filter_start_datetime = datetime.combine(filter_date, start_time_obj)
            filter_end_datetime = datetime.combine(filter_date, end_time_obj)

        except ValueError as e:
            print(f"Error parsing time parameters: {e}. Defaulting to full day.")
            filter_start_datetime = datetime.combine(filter_date, time(0, 0, 0))
            filter_end_datetime = datetime.combine(filter_date, time(23, 59, 59, 999999))
            selected_start_time_for_template = "00:00"
            selected_end_time_for_template = "23:59"


    if current_timezone:
        if filter_start_datetime:
            filter_start_datetime = current_timezone.localize(filter_start_datetime)
        if filter_end_datetime:
            filter_end_datetime = current_timezone.localize(filter_end_datetime)

    # --- Define the base folder for this specific report's images ---
    filename_date = selected_date_str if selected_date_str else datetime.now().strftime('%Y-%m-%d')
    filename_district = selected_district.replace(' ', '_') if selected_district else 'All_Districts'
    filename_start_time = selected_start_time_for_template.replace(':', '-') if selected_start_time_for_template else '00-00'
    filename_end_time = selected_end_time_for_template.replace(':', '-') if selected_end_time_for_template else '23-59'

    report_folder_name = f"{filename_date}_{filename_district}_{filename_start_time}-{filename_end_time}"
    
    # Path where images for this report will be saved temporarily for PDF generation
    report_specific_media_dir = os.path.join(settings.MEDIA_ROOT, 'report_images', report_folder_name)

    # Clean up previous report images for this specific folder if it exists
    if os.path.exists(report_specific_media_dir):
        print(f"Cleaning up existing report image directory: {report_specific_media_dir}")
        try:
            shutil.rmtree(report_specific_media_dir)
        except OSError as e:
            print(f"Error removing directory {report_specific_media_dir}: {e}")
    
    os.makedirs(report_specific_media_dir, exist_ok=True) 
    print(f"PDF Generation Process: Saving images to: {report_specific_media_dir}")

    # --- List to store URLs of saved images for PDF template ---
    generated_images_for_pdf_template = [] 

    gdf = None
    gdf_tn = None

    # Load shapefile (same logic as in report_view)
    try:
        if not os.path.exists(SHAPEFILE_PATH):
            raise FileNotFoundError(f"Shapefile not found at {SHAPEFILE_PATH}.")
        gdf = gpd.read_file(SHAPEFILE_PATH)
        gdf_tn = gdf[
            (gdf['NAME_1'].str.strip().str.lower() == 'tamilnadu') |
            (gdf['NAME_1'].str.strip().str.lower() == 'tamil nadu')
        ].to_crs("EPSG:4326")
        print(f"PDF Generation Process: Shapefile loaded successfully.")
    except FileNotFoundError as fnfe:
        print(f"PDF Generation Process: CRITICAL ERROR: Shapefile not found: {fnfe}")
    except Exception as e:
        print(f"PDF Generation Process: ERROR loading shapefile: {e}")

    # --- Block for iterating through available image timestamps and saving them ---
    if os.path.exists(settings.MEDIA_ROOT) and os.path.isdir(settings.MEDIA_ROOT) and gdf_tn is not None:
        available_image_timestamps_and_folders = []
        # Adjusted logic to search for _all_ folders in MEDIA_ROOT
        for d in os.listdir(settings.MEDIA_ROOT):
            full_path_to_folder = os.path.join(settings.MEDIA_ROOT, d)
            if os.path.isdir(full_path_to_folder) and d.startswith(filter_date.strftime('%Y-%m-%d')): # Only consider folders for the selected date
                try:
                    # Parse timestamp from folder name, e.g., '2025-06-27_10-00-00'
                    folder_timestamp_str_raw = d.split('_')[0] + '_' + d.split('_')[1] + '_' + d.split('_')[2] # Extract YYYY-MM-DD_HH-MM-SS
                    folder_datetime_obj = datetime.strptime(folder_timestamp_str_raw, '%Y-%m-%d_%H-%M-%S')
                    
                    if current_timezone:
                        folder_datetime_obj = current_timezone.localize(folder_datetime_obj)

                    if folder_datetime_obj.date() == filter_date and \
                       filter_start_datetime <= folder_datetime_obj < filter_end_datetime:
                        available_image_timestamps_and_folders.append((folder_datetime_obj, full_path_to_folder))
                except (ValueError, IndexError):
                    # print(f"Skipping non-standard folder name or format: {d}") # Debugging
                    pass # Ignore folders not matching the expected timestamp format
        
        available_image_timestamps_and_folders.sort(key=lambda x: x[0])

        print(f"PDF Generation Process: Found {len(available_image_timestamps_and_folders)} image folders for saving within the selected time range.")

        for timestamp_dt, folder_path in available_image_timestamps_and_folders:
            base_image_path_for_this_timestamp = os.path.join(folder_path, 'cropped', 'tamil_nadu_cropped.png')
            
            pil_images = _generate_image_data_for_timestamp(
                base_image_path_for_this_timestamp, timestamp_dt, selected_district, gdf_tn
            )

            if pil_images: 
                current_image_set_for_pdf = {
                    'timestamp': timestamp_dt,
                    'cropped_tn': None,
                    'masked_district': None,
                    'aligned_overlay_tn': None,
                }
                # Save PIL images to files and get their ABSOLUTE URLs for the PDF template
                for img_type, pil_img in pil_images.items():
                    if pil_img:
                        url = save_image_and_get_url(
                            request, pil_img, report_specific_media_dir, timestamp_dt, img_type
                        )
                        current_image_set_for_pdf[img_type] = url
                    elif img_type == 'masked_district' and selected_district == 'All Districts':
                        # If masked_district is None but 'All Districts' is selected, reuse cropped_tn
                        current_image_set_for_pdf['masked_district'] = current_image_set_for_pdf['cropped_tn']
                    else:
                        current_image_set_for_pdf[img_type] = None 
                
                generated_images_for_pdf_template.append(current_image_set_for_pdf)

            else:
                print(f"PDF Generation Process: Skipping images for {timestamp_dt} due to generation failure.")
    else:
        print(f"PDF Generation Process skipped: Base media directory or shapefile not loaded successfully.")

    # --- Filtering CloudAnalysis data (same logic as in report_view) ---
    cloud_analysis_query = CloudAnalysis.objects.filter(
        timestamp__date=filter_date
    )

    if selected_district != 'All Districts':
        cloud_analysis_query = cloud_analysis_query.filter(city__iexact=selected_district)

    if filter_start_datetime and filter_end_datetime:
        cloud_analysis_query = cloud_analysis_query.filter(
            timestamp__gte=filter_start_datetime,
            timestamp__lt=filter_end_datetime
        )
    
    EXCLUDE_NO_PRECIP_MESSAGE = "No significant cloud levels found for precipitation"
    cloud_analysis_query = cloud_analysis_query.exclude(values__iexact=EXCLUDE_NO_PRECIP_MESSAGE)

    filtered_cloud_analysis_data = list(cloud_analysis_query.order_by('city', 'timestamp'))
    
    print(f"Data Fetch: Fetched {len(filtered_cloud_analysis_data)} weather data points for PDF.")

    full_available_districts = []
    try:
        if not os.path.exists(SHAPEFILE_PATH):
            print(f"ERROR: Shapefile for district list not found at {SHAPEFILE_PATH}. Falling back to default list.")
            full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']
        else:
            temp_gdf = gpd.read_file(SHAPEFILE_PATH)
            gdf_tn_districts_for_list = temp_gdf[
                (temp_gdf['NAME_1'].str.strip().str.lower() == 'tamilnadu') |
                (temp_gdf['NAME_1'].str.strip().str.lower() == 'tamil nadu')
            ]
            
            if 'NAME_2' in gdf_tn_districts_for_list.columns:
                unique_districts = gdf_tn_districts_for_list['NAME_2'].dropna().unique().tolist()
                full_available_districts = sorted(unique_districts)
            else:
                full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']

    except Exception as e:
        print(f"Error loading districts from shapefile for dropdown: {e}. Falling back to default list.")
        full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']


    context_for_pdf = {
        'generated_images_for_display': generated_images_for_pdf_template, # Pass URLs for PDF
        'selected_date': filter_date.strftime('%Y-%m-%d'),
        'selected_district': selected_district,
        'available_districts': full_available_districts, # Needed for the template, even if not directly displayed in PDF
        'cloud_analysis_data': filtered_cloud_analysis_data,
        'selected_image_view': selected_image_view,
        'selected_start_time': selected_start_time_for_template,
        'selected_end_time': selected_end_time_for_template,
    }

    # --- Playwright PDF Generation ---
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch() # You might need headless=True in production
            page = browser.new_page()
            html_content = render_to_string('report/report_pdf.html', context_for_pdf, request=request)
            page.set_content(html_content)
            page.wait_for_load_state("networkidle", timeout=30000) # Increased timeout to 30 seconds

            pdf_buffer = page.pdf(
                format="A4",
                print_background=True, # Ensures background colors/images from CSS are printed
                margin={
                    "top": "20px",
                    "bottom": "20px",
                    "left": "20px",
                    "right": "20px"
                }
            )
            browser.close()

            response = HttpResponse(pdf_buffer, content_type='application/pdf')
            pdf_filename = f"Weather_Report_{report_folder_name}.pdf"
            response['Content-Disposition'] = f'attachment; filename="{pdf_filename}"'
            
            print(f"PDF generated successfully: {pdf_filename}")
            return response
            
    except Exception as e:
        print(f"Error generating PDF: {e}")
        # Return a JSON response with an error message to the user's browser
        return JsonResponse({'status': 'error', 'message': f'Failed to generate PDF: {e}'}, status=500)
    finally:
        # Clean up the temporarily saved images after PDF generation
        if os.path.exists(report_specific_media_dir):
            try:
                shutil.rmtree(report_specific_media_dir)
                print(f"Cleaned up temporary report images directory: {report_specific_media_dir}")
            except OSError as e:
                print(f"Error removing temporary directory {report_specific_media_dir} during cleanup: {e}")