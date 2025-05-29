import streamlit as st
import ee
import urllib.request
import os
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import io
import tempfile
import folium
from streamlit_folium import folium_static
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Deforestation Analysis Tool",
    page_icon="🌳",
    layout="wide"
)

def authenticate_ee():
    """Initialize Earth Engine with user credentials or service account"""
    try:
        # Try to initialize without explicit authentication
        ee.Authenticate()
        ee.Initialize(project='ee-vr5189')  # Use your project ID
        return True
    except Exception as e:
        st.error(f"Authentication Error: {str(e)}")
        st.info("If you get an authentication error, please run 'earthengine authenticate' in your terminal first.")
        return False

def get_satellite_image(latitude, longitude, start_date, end_date, buffer_size=5000, 
                        collection_name='COPERNICUS/S2_SR', cloud_cover_max=20, gamma=1.4):
    """Retrieve a satellite image for the specified location and time period"""
    try:
        # Create a point and region
        point = ee.Geometry.Point([longitude, latitude])
        region = point.buffer(buffer_size).bounds()
        
        # Filter collection by date and location
        collection = ee.ImageCollection(collection_name) \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        
        # Get the first (least cloudy) image
        image = collection.first()
        
        if image is None:
            return None, f"No images found for the period {start_date} to {end_date}. Try adjusting parameters."
        
        # Get image date
        image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        
        # Set visualization parameters for true color
        vis_params = {
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'gamma': gamma
        }
        
        # Get image URL
        url = image.getThumbURL({
            'region': region,
            'dimensions': 2048,
            'format': 'png',
            **vis_params
        })
        
        # Download the image
        with urllib.request.urlopen(url) as response:
            image_bytes = response.read()
        
        # Get NDVI visualization
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndvi_vis = {
            'min': -0.2,
            'max': 0.8,
            'palette': ['red', 'orange', 'yellow', 'green', 'darkgreen']
        }
        
        ndvi_url = ndvi.getThumbURL({
            'region': region,
            'dimensions': 2048,
            'format': 'png',
            **ndvi_vis
        })
        
        # Download the NDVI image
        with urllib.request.urlopen(ndvi_url) as response:
            ndvi_bytes = response.read()
        
        # Prepare metadata
        metadata = {
            'id': image.id().getInfo(),
            'date': image_date,
            'collection': collection_name,
            'cloud_cover': image.get('CLOUDY_PIXEL_PERCENTAGE'),
            'coordinates': f"{latitude}, {longitude}",
            'url': url,
            'ndvi_url': ndvi_url,
            'image': image,  # Store the actual EE image object
            'region': region  # Store the region for later use
        }
        
        return {
            'rgb': image_bytes, 
            'ndvi': ndvi_bytes
        }, metadata
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def calculate_deforestation(before_metadata, after_metadata, threshold=0.2):
    """Calculate deforestation between two time periods"""
    try:
        # Get the images and region
        before_image = before_metadata['image']
        after_image = after_metadata['image']
        region = before_metadata['region']
        
        # Calculate NDVI for both periods
        before_ndvi = before_image.normalizedDifference(['B8', 'B4']).rename('before_ndvi')
        after_ndvi = after_image.normalizedDifference(['B8', 'B4']).rename('after_ndvi')
        
        # Calculate the difference in NDVI
        ndvi_diff = before_ndvi.subtract(after_ndvi).rename('ndvi_diff')
        
        # Create a mask for significant vegetation loss (deforestation)
        # Areas where NDVI decreased by more than the threshold
        deforestation = ndvi_diff.gt(threshold)
        
        # Apply a mask to keep only forested areas in the before image (NDVI > 0.4)
        forest_mask = before_ndvi.gt(0.4)
        deforestation = deforestation.updateMask(forest_mask).rename('deforestation')
        
        # Calculate some statistics (area)
        area_image = deforestation.multiply(ee.Image.pixelArea())
        deforested_area = area_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=10,
            maxPixels=1e9
        ).get('deforestation')
        
        # Convert to hectares
        deforested_hectares = ee.Number(deforested_area).divide(10000).round()
        
        # Get the deforestation visualization
        deforestation_vis = deforestation.visualize(**{
            'min': 0,
            'max': 1,
            'palette': ['white', 'red']
        })
        
        # Create a blended image for visualization
        rgb = after_image.visualize(**{
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        })
        
        # Blend the RGB image with the deforestation mask
        blended = rgb.blend(deforestation_vis)
        
        # Get the image URL
        url = blended.getThumbURL({
            'region': region,
            'dimensions': 2048,
            'format': 'png'
        })
        
        # Download the deforestation visualization
        with urllib.request.urlopen(url) as response:
            deforestation_bytes = response.read()
        
        # Return the visualization and statistics
        return {
            'deforestation_image': deforestation_bytes,
            'deforested_hectares': deforested_hectares.getInfo(),
            'url': url
        }
        
    except Exception as e:
        st.error(f"Error calculating deforestation: {str(e)}")
        return None

def display_comparison(before_images, before_metadata, after_images, after_metadata, deforestation_result):
    """Display the comparison between two time periods"""
    st.header("Deforestation Analysis Results")
    
    # Display summary statistics
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Before Period")
        st.write(f"**Date:** {before_metadata['date']}")
        
    with col2:
        st.subheader("After Period")
        st.write(f"**Date:** {after_metadata['date']}")
        
    with col3:
        st.subheader("Change Detected")
        st.write(f"**Deforested Area:** {deforestation_result['deforested_hectares']} hectares")
        percent_change = (deforestation_result['deforested_hectares'] / 10000) * 100  # Assuming region is in hectares
        st.write(f"**Percent Change:** {percent_change:.2f}%")
    
    # Display the images in tabs
    tabs = st.tabs(["RGB Comparison", "NDVI Comparison", "Deforestation Map"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.image(before_images['rgb'], caption=f"Before: {before_metadata['date']}", use_column_width=True)
        with col2:
            st.image(after_images['rgb'], caption=f"After: {after_metadata['date']}", use_column_width=True)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.image(before_images['ndvi'], caption=f"NDVI Before: {before_metadata['date']}", use_column_width=True)
            st.info("NDVI (Normalized Difference Vegetation Index) is a measure of vegetation health. Higher values (green) indicate dense vegetation, while lower values (red/orange) indicate sparse vegetation or non-vegetated areas.")
        with col2:
            st.image(after_images['ndvi'], caption=f"NDVI After: {after_metadata['date']}", use_column_width=True)
    
    with tabs[2]:
        st.image(deforestation_result['deforestation_image'], caption="Deforestation Map (Red areas show detected deforestation)", use_column_width=True)
        st.warning("Red areas indicate where significant vegetation loss (likely deforestation) has occurred between the two periods.")
    
    # Display a map with both time periods
    st.subheader("Interactive Map View")
    lat, lon = [float(x.strip()) for x in before_metadata['coordinates'].split(',')]
    
    # Create map
    m = folium.Map(location=[lat, lon], zoom_start=12, control_scale=True)
    
    # Add different base layers
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite',
        overlay=False
    ).add_to(m)
    
    folium.TileLayer(
        name='OpenStreetMap',
        overlay=False
    ).add_to(m)
    
    # Add custom layers using the URLs
    folium.raster_layers.ImageOverlay(
        image=before_metadata['url'],
        bounds=[[lat-0.1, lon-0.1], [lat+0.1, lon+0.1]],
        name=f"Before: {before_metadata['date']}",
        opacity=0.7
    ).add_to(m)
    
    folium.raster_layers.ImageOverlay(
        image=after_metadata['url'],
        bounds=[[lat-0.1, lon-0.1], [lat+0.1, lon+0.1]],
        name=f"After: {after_metadata['date']}",
        opacity=0.7
    ).add_to(m)
    
    folium.raster_layers.ImageOverlay(
        image=deforestation_result['url'],
        bounds=[[lat-0.1, lon-0.1], [lat+0.1, lon+0.1]],
        name="Deforestation Map",
        opacity=0.7
    ).add_to(m)
    
    # Add a marker at the center
    folium.Marker(
        [lat, lon],
        popup=f"Analysis Center Point",
        tooltip="Center"
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright').add_to(m)
    
    # Display the map
    folium_static(m)

def main():
    st.title("🌳 Deforestation Analysis Tool")
    st.markdown("Compare satellite imagery to detect and analyze deforestation between two time periods")
    st.sidebar.success("Deforestation Analysis Tool")
    st.sidebar.info("Compare satellite imagery to detect and analyze deforestation between two time periods")

    
    # Authentication
    if not authenticate_ee():
        st.stop()
    
    # Location input
    st.header("1. Select Location")
    col1, col2 = st.columns(2)
    
    with col1:
        latitude = st.number_input("Latitude", 
                                  value=-3.4653,  # Amazon default
                                  min_value=-90.0, 
                                  max_value=90.0,
                                  format="%.5f")
        
    with col2:
        longitude = st.number_input("Longitude", 
                                   value=-62.2159,  # Amazon default
                                   min_value=-180.0, 
                                   max_value=180.0,
                                   format="%.5f")
    
    # Time period selection
    st.header("2. Select Time Periods")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Before Period")
        before_start = st.date_input("Start Date (Before)", 
                                     value=(datetime.now() - timedelta(days=730)),  # 2 years ago
                                     max_value=datetime.now() - timedelta(days=60))
        before_end = st.date_input("End Date (Before)", 
                                   value=(datetime.now() - timedelta(days=670)),  # About 60 days after before_start
                                   max_value=datetime.now() - timedelta(days=60))
        
    with col2:
        st.subheader("After Period")
        after_start = st.date_input("Start Date (After)", 
                                    value=(datetime.now() - timedelta(days=60)),
                                    max_value=datetime.now())
        after_end = st.date_input("End Date (After)", 
                                  value=datetime.now(),
                                  max_value=datetime.now())
    
    # Advanced options
    st.header("3. Configure Analysis Parameters")
    with st.expander("Analysis Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            buffer_size = st.slider("Area Size (meters)", 
                                   min_value=1000, 
                                   max_value=20000, 
                                   value=5000,
                                   step=1000,
                                   help="Size of the region around the point in meters")
            
            cloud_cover = st.slider("Max Cloud Cover (%)", 
                                   min_value=0, 
                                   max_value=100, 
                                   value=20,
                                   help="Maximum cloud coverage percentage")
        
        with col2:
            collection = st.selectbox("Image Collection", 
                                     options=[
                                         'COPERNICUS/S2_SR',  # Sentinel-2 Surface Reflectance
                                         'LANDSAT/LC08/C02/T1_TOA',  # Landsat 8
                                         'LANDSAT/LC09/C02/T1_TOA',  # Landsat 9
                                     ],
                                     help="Satellite data source - Sentinel-2 provides the highest resolution (10m)")
            
            deforestation_threshold = st.slider("Deforestation Threshold", 
                                              min_value=0.1, 
                                              max_value=0.5, 
                                              value=0.2,
                                              step=0.05,
                                              help="Threshold for NDVI difference to classify as deforestation")
    
    # Run analysis button
    analyze_button = st.button("Run Deforestation Analysis", type="primary")
    
    if analyze_button:
        with st.spinner("Analyzing satellite imagery for deforestation..."):
            # Get before period image
            before_images, before_metadata = get_satellite_image(
                latitude=latitude,
                longitude=longitude,
                start_date=before_start.strftime('%Y-%m-%d'),
                end_date=before_end.strftime('%Y-%m-%d'),
                buffer_size=buffer_size,
                collection_name=collection,
                cloud_cover_max=cloud_cover
            )
            
            if before_images is None:
                st.error(before_metadata)  # Display error message
                st.stop()
            
            # Get after period image
            after_images, after_metadata = get_satellite_image(
                latitude=latitude,
                longitude=longitude,
                start_date=after_start.strftime('%Y-%m-%d'),
                end_date=after_end.strftime('%Y-%m-%d'),
                buffer_size=buffer_size,
                collection_name=collection,
                cloud_cover_max=cloud_cover
            )
            
            if after_images is None:
                st.error(after_metadata)  # Display error message
                st.stop()
            
            # Calculate deforestation
            deforestation_result = calculate_deforestation(
                before_metadata, 
                after_metadata,
                threshold=deforestation_threshold
            )
            
            if deforestation_result:
                # Display comparison
                display_comparison(before_images, before_metadata, after_images, after_metadata, deforestation_result)
                
                # Add a download button for a report
                st.download_button(
                    label="Download Analysis Report",
                    data=f"""
                    Deforestation Analysis Report
                    -----------------------------
                    Location: {latitude}, {longitude}
                    
                    Before Period: {before_start} to {before_end}
                    Image Date: {before_metadata['date']}
                    
                    After Period: {after_start} to {after_end}
                    Image Date: {after_metadata['date']}
                    
                    Results:
                    - Deforested Area: {deforestation_result['deforested_hectares']} hectares
                    - Analysis Threshold: NDVI difference > {deforestation_threshold}
                    
                    Analysis performed using Google Earth Engine and {collection} data
                    """,
                    file_name=f"deforestation_analysis_{before_metadata['date']}_to_{after_metadata['date']}.txt",
                    mime="text/plain"
                )
    
    # Instructions
    if not analyze_button:
        st.info("""
        ### How to Use This Tool
        
        1. **Select Location**: Enter the latitude and longitude of your area of interest. The default is set to a region in the Amazon rainforest.
        
        2. **Select Time Periods**: 
           - The "Before" period should be earlier in time
           - The "After" period should be more recent
           - Typical timeframes are 1-2 years apart to observe meaningful changes
        
        3. **Configure Analysis Parameters**:
           - Area Size: Larger values cover more area but may take longer to process
           - Cloud Cover: Lower values give clearer images but may limit available data
           - Deforestation Threshold: Controls sensitivity of deforestation detection
        
        4. **Run Analysis**: Click the button to analyze and compare the selected periods
        
        ### About Deforestation Detection
        
        This tool uses the Normalized Difference Vegetation Index (NDVI) to detect changes in forest cover. 
        Areas that show significant NDVI decrease between the two periods are classified as deforestation.
        """)
        
        # Example locations
        st.subheader("Example Locations")
        examples = {
            "Amazon Rainforest, Brazil": {"lat": -3.4653, "lon": -62.2159},
            "Borneo, Indonesia": {"lat": 0.9619, "lon": 114.5548},
            "Congo Basin, DRC": {"lat": -0.7893, "lon": 20.3981}
        }
        
        cols = st.columns(len(examples))
        for i, (name, data) in enumerate(examples.items()):
            with cols[i]:
                st.button(f"Use {name}", key=f"example_{i}", on_click=lambda lat=data["lat"], lon=data["lon"]: 
                         [st.session_state.update({'latitude': lat, 'longitude': lon})])

if __name__ == "__main__":
    main()
