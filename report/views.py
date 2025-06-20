# report/views.py

from django.shortcuts import render
import os
from django.conf import settings
from datetime import datetime, timedelta, date, time
import pytz

# --- Import your actual CloudAnalysis model ---
from weather.models import CloudAnalysis 

# --- Image Processing Imports ---
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
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

warnings.filterwarnings("ignore")

# --- GLOBAL CONFIGURATION FOR IMAGE GENERATION (MUST MATCH YOUR SETUP) ---
SHAPEFILE_PATH = os.path.join(settings.BASE_DIR, 'weather', 'management', 'commands', 'gadm41_IND_2.json') 

# These are the alignment values for the FULL TN map with overlay.
FINAL_MIN_LON = 74.80
FINAL_MAX_LON = 80.37
FINAL_MIN_LAT = 7.98
FINAL_MAX_LAT = 13.53
# -------------------------------------------------------------------------


def report_view(request):
    selected_date_str = request.GET.get('date')
    selected_district = request.GET.get('district', 'All Districts')
    if selected_district == "" or selected_district is None:
        selected_district = "All Districts"

    # --- UPDATED: Get time filter parameters from new frontend names ---
    start_time_hour_str = request.GET.get('start_time_hour')
    start_time_minute_str = request.GET.get('start_time_minute')
    end_time_hour_str = request.GET.get('end_time_hour')
    end_time_minute_str = request.GET.get('end_time_minute')
    # --- END UPDATED ---

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

    # --- Prepare selected time strings for template context (HH:MM format) ---
    selected_start_time_for_template = ''
    selected_end_time_for_template = ''

    # --- Construct datetime objects for actual filtering ---
    filter_start_datetime = None
    filter_end_datetime = None

    current_timezone = pytz.timezone(settings.TIME_ZONE) if settings.USE_TZ else None

    # Logic to set default times if not provided in GET parameters (e.g., initial page load)
    if not start_time_hour_str or not start_time_minute_str:
        now = datetime.now()
        current_minute = now.minute
        current_hour = now.hour

        # Round up minutes to the nearest 15 for 'From Time'
        from_minute = (current_minute // 15) * 15
        if current_minute % 15 != 0:
            from_minute = ((current_minute // 15) + 1) * 15
            if from_minute == 60:
                from_minute = 0
                current_hour = (current_hour + 1) % 24

        from_hour = current_hour
        
        # Set default 'From Time'
        start_time_obj = time(from_hour, from_minute)
        selected_start_time_for_template = start_time_obj.strftime("%H:%M")
        filter_start_datetime = datetime.combine(filter_date, start_time_obj)

        # Set default 'To Time' (15 minutes after 'From Time')
        to_minute = from_minute + 15
        to_hour = from_hour
        if to_minute == 60:
            to_minute = 0
            to_hour = (to_hour + 1) % 24
        
        end_time_obj = time(to_hour, to_minute)
        selected_end_time_for_template = end_time_obj.strftime("%H:%M")
        filter_end_datetime = datetime.combine(filter_date, end_time_obj)

    else:
        # If time parameters are provided, use them
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
            # If parsing fails, fall back to full day (00:00 to 23:59:59)
            filter_start_datetime = datetime.combine(filter_date, time(0, 0, 0))
            filter_end_datetime = datetime.combine(filter_date, time(23, 59, 59, 999999))
            selected_start_time_for_template = "00:00"
            selected_end_time_for_template = "23:59"


    # Make filter datetimes timezone-aware if USE_TZ is true
    if current_timezone:
        if filter_start_datetime:
            filter_start_datetime = current_timezone.localize(filter_start_datetime)
        if filter_end_datetime:
            filter_end_datetime = current_timezone.localize(filter_end_datetime)


    # --- DEBUGGING PRINTS ---
    print(f"\n--- Generating Images for Date: {target_date_display_str}, District: {selected_district}, Image View: {selected_image_view} ---")
    print(f"Time Range Filter (Backend): {filter_start_datetime} - {filter_end_datetime}")
    print(f"SHAPEFILE_PATH configured as: {SHAPEFILE_PATH}")
    # --- END DEBUGGING PRINTS ---

    # --- LOGIC TO FIND THE CORRECT TIMESTAMPED FOLDER (for radar images) ---
    image_base_media_path = settings.MEDIA_ROOT 

    found_date_folder_with_time = None
    if os.path.exists(image_base_media_path) and os.path.isdir(image_base_media_path):
        all_timestamp_folders = []
        for d in os.listdir(image_base_media_path):
            full_path = os.path.join(image_base_media_path, d)
            if os.path.isdir(full_path):
                try:
                    folder_date_part_str = d.split('_')[0]
                    folder_date = datetime.strptime(folder_date_part_str, '%Y-%m-%d').date()
                    
                    if folder_date == filter_date:
                        all_timestamp_folders.append(d) 
                except (ValueError, IndexError):
                    pass 
        
        if all_timestamp_folders:
            all_timestamp_folders.sort()
            found_date_folder_with_time = all_timestamp_folders[-1] 
            print(f"Found timestamped folder for date {target_date_display_str}: {found_date_folder_with_time}")
        else:
            print(f"No timestamped folder found for date: {target_date_display_str} within {image_base_media_path}")
    else:
        print(f"Image base directory '{image_base_media_path}' does not exist or is not a directory. Please check settings.MEDIA_ROOT.")

    cropped_tn_image_b64 = None
    masked_district_image_b64 = None 
    aligned_overlay_tn_b64 = None

    # --- Start Image Generation/Loading ---
    img_pil = None
    img_np = None
    gdf = None
    gdf_tn = None
    
    # Try loading base radar image from the timestamped folder
    if found_date_folder_with_time:
        BASE_MAP_IMAGE_SOURCE_FOR_THIS_RUN = os.path.join(
            settings.MEDIA_ROOT, 
            found_date_folder_with_time, 
            'cropped', 
            'tamil_nadu_cropped.png'
        )
        print(f"Attempting to load base map image from: {BASE_MAP_IMAGE_SOURCE_FOR_THIS_RUN}")
        try:
            img_pil = Image.open(BASE_MAP_IMAGE_SOURCE_FOR_THIS_RUN).convert("RGB")
            img_np = np.array(img_pil)
            print(f"Base map image loaded successfully.")
        except FileNotFoundError:
            print(f"ERROR: Base map image '{BASE_MAP_IMAGE_SOURCE_FOR_THIS_RUN}' not found.")
        except Exception as e:
            print(f"ERROR loading base map image: {e}")

    # Try loading shapefile
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

    # Only proceed with image generation if both base image and shapefile loaded
    if img_pil is not None and gdf_tn is not None:
        height, width, _ = img_np.shape
        transform = from_bounds(FINAL_MIN_LON, FINAL_MIN_LAT, FINAL_MAX_LON, FINAL_MAX_LAT, width, height)

        try:
            # 1. Generate Cropped Tamil Nadu (Radar Only) - this is just the source image
            buffer = io.BytesIO()
            img_pil.save(buffer, format="PNG")
            cropped_tn_image_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            print(f"Generated Base64 for Cropped TN image.")

            # 2. Generate Masked District Image (Logic adjusted for 'All Districts' selected for this view)
            if selected_image_view == 'masked_coimbatore': # Note: 'masked_coimbatore' is a fixed value, consider if this should be dynamic based on selected_district
                if selected_district == 'All Districts':
                    masked_district_image_b64 = cropped_tn_image_b64 
                    print(f"Shape-Masked District view selected with 'All Districts': defaulting to full TN radar image.")
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

                        buffer = io.BytesIO()
                        Image.fromarray(cropped_district_img_np).save(buffer, format="PNG")
                        masked_district_image_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
                        buffer.close()
                        print(f"Generated Base64 for Masked '{selected_district}' image.")
                    else:
                        print(f"Warning: District '{selected_district}' not found in shapefile for masked image generation.")
            else:
                print(f"Masked image generation skipped: specific district not selected or not applicable for current view.")


            # 3. Generate Overall TN Map with District Outlines (and highlighted district)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img_np, extent=[FINAL_MIN_LON, FINAL_MAX_LON, FINAL_MIN_LAT, FINAL_MAX_LAT])
            gdf_tn.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

            if selected_district != 'All Districts':
                district_rows_for_name_for_highlight = gdf_tn[gdf_tn['NAME_2'].str.lower() == selected_district.lower()]
                if not district_rows_for_name_for_highlight.empty:
                    district_rows_for_name_for_highlight.boundary.plot(ax=ax, edgecolor='cyan', linewidth=2, linestyle='--', label=selected_district)
                    ax.set_title(f"Aligned Screenshot with {selected_district} Highlighted")
                    ax.legend()
                else:
                    ax.set_title(f"Aligned Screenshot (District '{selected_district}' not found for highlight)")
            else:
                ax.set_title("Aligned Screenshot with All TN District Outlines")

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect('equal')
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format="PNG", bbox_inches='tight', pad_inches=0.1)
            aligned_overlay_tn_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close(fig)
            print(f"Generated Base64 for Aligned Overlay TN image.")

        except FileNotFoundError as fnfe:
            print(f"CRITICAL ERROR (Image/Shapefile): {fnfe}")
            print(f"Please ensure '{BASE_MAP_IMAGE_SOURCE_FOR_THIS_RUN}' and '{SHAPEFILE_PATH}' exist and are accessible.")
            cropped_tn_image_b64 = None
            masked_district_image_b64 = None
            aligned_overlay_tn_b64 = None
        except Exception as e:
            print(f"An unexpected error occurred during image generation: {e}")
            cropped_tn_image_b64 = None
            masked_district_image_b64 = None
            aligned_overlay_tn_b64 = None
    else:
        print(f"Image generation skipped: Base image or shapefile not loaded successfully.")


    image_urls = {
        'cropped_tn': cropped_tn_image_b64,
        'masked_coimbatore': masked_district_image_b64, 
        'aligned_overlay_tn': aligned_overlay_tn_b64,
    }

    # --- Filtering CloudAnalysis data ---
    cloud_analysis_query = CloudAnalysis.objects.filter(
        timestamp__date=filter_date # Filter by date part of timestamp
    )

    if selected_district != 'All Districts':
        cloud_analysis_query = cloud_analysis_query.filter(city__iexact=selected_district)

    # Apply time filters if valid filter_start_datetime and filter_end_datetime were constructed
    if filter_start_datetime and filter_end_datetime:
        # Using __gte for greater than or equal to, and __lt for less than (exclusive end)
        # This correctly defines the interval: [start_time, end_time)
        cloud_analysis_query = cloud_analysis_query.filter(
            timestamp__gte=filter_start_datetime,
            timestamp__lt=filter_end_datetime # Changed from __lte to __lt for standard interval
        )
    
    # NEW FILTER: Exclude records where 'values' indicates no significant precipitation
    # Assuming the exact string is "No significant cloud levels found for precipitation"
    # You might want to make this string a constant or configuration if it changes often.
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
        'image_urls': image_urls,
        'selected_date': filter_date.strftime('%Y-%m-%d'),
        'selected_district': selected_district,
        'available_districts': full_available_districts,
        'cloud_analysis_data': filtered_cloud_analysis_data, 
        'selected_image_view': selected_image_view,
        # --- UPDATED: Pass the HH:MM strings to context for frontend dropdowns ---
        'selected_start_time': selected_start_time_for_template,
        'selected_end_time': selected_end_time_for_template,
        # --- END UPDATED ---
    }
    return render(request, 'report/report.html', context)