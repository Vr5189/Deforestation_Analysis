import streamlit as st
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Simple Webcam Monitoring",
    page_icon="🌳",
    layout="wide"
)

# Create directory for storing captured images
if not os.path.exists("captured_frames"):
    os.makedirs("captured_frames")

# Add title and description
st.title("Simple Webcam Monitoring")
st.markdown("Capture frames and compare vegetation changes")

# Sidebar configuration
st.sidebar.header("Settings")

# API key input with default value
api_key = st.sidebar.text_input("Roboflow API Key", value="pjZtbsAzjhkBsKvruel1", type="password")

# Model parameters
confidence = st.sidebar.slider("Confidence Threshold", min_value=1, max_value=100, value=40)

# Function to process the image
def process_image(image_bytes, api_key, confidence, temp_filename):
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("deforestation-satellite-imagery-335n4")
    model = project.version(3).model
    
    # Save the image bytes to a temporary file
    with open(temp_filename, "wb") as f:
        f.write(image_bytes)
    
    # Make prediction
    result = model.predict(temp_filename, confidence=confidence).json()
    
    # Extract detections
    detections = sv.Detections.from_inference(result)
    
    # Read the image for annotation
    image = cv2.imread(temp_filename)
    
    # Only use mask annotator (no labels)
    mask_annotator = sv.MaskAnnotator()
    
    # Apply only segmentation masks to the image
    annotated_image = mask_annotator.annotate(scene=image, detections=detections)
    
    # Convert from BGR to RGB
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Calculate area percentages
    image_area = image.shape[0] * image.shape[1]  # total pixel area
    class_areas = {}
    
    for pred in result["predictions"]:
        class_name = pred["class"]
        
        # Calculate approximate area from bounding box if polygon points not available
        if "width" in pred and "height" in pred:
            area = pred["width"] * pred["height"]
        else:
            # Fallback to points if available
            if "points" in pred:
                points = np.array(pred["points"])
                area = cv2.contourArea(points)
            else:
                area = 0
        
        if class_name in class_areas:
            class_areas[class_name] += area
        else:
            class_areas[class_name] = area
    
    # Calculate percentages
    class_percentages = {}
    for class_name, area in class_areas.items():
        percentage = (area / image_area) * 100
        class_percentages[class_name] = percentage
    
    return annotated_image_rgb, class_percentages

# Function to compare results (focusing on tree and farmland only)
def compare_results(earlier_percentages, later_percentages):
    # Initialize with empty values for tree and farmland in case they are not detected
    comparison = {
        "tree": {
            "earlier_percentage": 0,
            "later_percentage": 0,
            "absolute_change": 0,
            "relative_change": 0
        },
        "farmland": {
            "earlier_percentage": 0,
            "later_percentage": 0,
            "absolute_change": 0,
            "relative_change": 0
        }
    }
    
    # Process only tree and farmland classes
    for class_name in ["tree", "farmland"]:
        # Get values or default to 0 if class not detected
        earlier_pct = earlier_percentages.get(class_name, 0)
        later_pct = later_percentages.get(class_name, 0)
        
        # Calculate absolute and relative changes
        abs_change = later_pct - earlier_pct
        rel_change = 0
        if earlier_pct > 0:
            rel_change = (abs_change / earlier_pct) * 100
        
        comparison[class_name] = {
            "earlier_percentage": earlier_pct,
            "later_percentage": later_pct,
            "absolute_change": abs_change,
            "relative_change": rel_change
        }
    
    return comparison

# Function to capture a frame from webcam
def capture_frame(camera_index=0):
    """Capture a single frame from the webcam"""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return None
    
    # Capture a single frame
    ret, frame = cap.read()
    
    # Release the webcam
    cap.release()
    
    if not ret:
        st.error("Error: Could not capture frame.")
        return None
    
    # Convert to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Save to buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    
    return image_bytes

# Main layout
camera_col, earlier_col, later_col = st.columns(3)

# Camera column for taking pictures
with camera_col:
    st.header("Take Pictures")
    
    # Camera selection
    camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0, help="Select your webcam (0 for default)")
    
    # Take earlier capture button
    if st.button("Take First/Earlier Picture"):
        with st.spinner("Capturing..."):
            image_bytes = capture_frame(camera_index)
            
            if image_bytes:
                # Save the earlier capture
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_frames/earlier_{timestamp}.jpg"
                
                with open(filename, "wb") as f:
                    f.write(image_bytes)
                
                # Store in session state
                st.session_state["earlier_image"] = {
                    "bytes": image_bytes,
                    "filename": filename,
                    "timestamp": timestamp
                }
                
                st.success("First picture captured!")
            else:
                st.error("Failed to capture image.")
    
    # Take later capture button
    if st.button("Take Second/Later Picture"):
        with st.spinner("Capturing..."):
            image_bytes = capture_frame(camera_index)
            
            if image_bytes:
                # Save the later capture
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_frames/later_{timestamp}.jpg"
                
                with open(filename, "wb") as f:
                    f.write(image_bytes)
                
                # Store in session state
                st.session_state["later_image"] = {
                    "bytes": image_bytes,
                    "filename": filename,
                    "timestamp": timestamp
                }
                
                st.success("Second picture captured!")
            else:
                st.error("Failed to capture image.")
    
    # Compare button (only show if both images are available)
    if "earlier_image" in st.session_state and "later_image" in st.session_state:
        if st.button("Compare Pictures", type="primary"):
            with st.spinner("Processing and comparing images..."):
                # Process earlier image
                earlier_annotated, earlier_percentages = process_image(
                    st.session_state["earlier_image"]["bytes"],
                    api_key,
                    confidence,
                    "earlier_temp.jpg"
                )
                
                # Process later image
                later_annotated, later_percentages = process_image(
                    st.session_state["later_image"]["bytes"],
                    api_key,
                    confidence,
                    "later_temp.jpg"
                )
                
                # Compare results
                comparison = compare_results(earlier_percentages, later_percentages)
                
                # Store results in session state
                st.session_state["results"] = {
                    "earlier_annotated": earlier_annotated,
                    "later_annotated": later_annotated,
                    "earlier_percentages": earlier_percentages,
                    "later_percentages": later_percentages,
                    "comparison": comparison
                }
                
                st.success("Analysis complete!")

# Earlier image column
with earlier_col:
    st.header("First/Earlier Image")
    
    if "earlier_image" in st.session_state:
        # Show the original image
        st.image(Image.open(io.BytesIO(st.session_state["earlier_image"]["bytes"])), 
                caption=f"Captured at: {st.session_state['earlier_image']['timestamp']}", 
                use_column_width=True)
        
        # Show the annotated image if available
        if "results" in st.session_state:
            st.subheader("Tree/Farmland Detection")
            st.image(st.session_state["results"]["earlier_annotated"], use_column_width=True)
            
            # Show percentages
            earlier_percentages = st.session_state["results"]["earlier_percentages"]
            tree_pct = earlier_percentages.get("tree", 0)
            farmland_pct = earlier_percentages.get("farmland", 0)
            
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                st.metric("Tree Coverage", f"{tree_pct:.2f}%")
            with e_col2:
                st.metric("Farmland Coverage", f"{farmland_pct:.2f}%")
    else:
        st.info("No first image captured yet. Use the 'Take First/Earlier Picture' button.")

# Later image column
with later_col:
    st.header("Second/Later Image")
    
    if "later_image" in st.session_state:
        # Show the original image
        st.image(Image.open(io.BytesIO(st.session_state["later_image"]["bytes"])), 
                caption=f"Captured at: {st.session_state['later_image']['timestamp']}",
                use_column_width=True)
        
        # Show the annotated image if available
        if "results" in st.session_state:
            st.subheader("Tree/Farmland Detection")
            st.image(st.session_state["results"]["later_annotated"], use_column_width=True)
            
            # Show percentages
            later_percentages = st.session_state["results"]["later_percentages"]
            tree_pct = later_percentages.get("tree", 0)
            farmland_pct = later_percentages.get("farmland", 0)
            
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                st.metric("Tree Coverage", f"{tree_pct:.2f}%")
            with l_col2:
                st.metric("Farmland Coverage", f"{farmland_pct:.2f}%")
    else:
        st.info("No second image captured yet. Use the 'Take Second/Later Picture' button.")

# Display comparison results if available
if "results" in st.session_state:
    st.header("Comparison Results")
    
    comparison = st.session_state["results"]["comparison"]
    
    # Create three columns for comparison metrics
    c1, c2, c3 = st.columns(3)
    
    with c1:
        tree_change = comparison["tree"]["absolute_change"]
        change_color = "normal"
        if tree_change < -5:
            change_color = "inverse"  # Red for decrease in trees
        
        st.metric(
            label="Tree Coverage Change",
            value=f"{tree_change:.2f}%",
            delta=f"{comparison['tree']['relative_change']:.2f}%",
            delta_color=change_color
        )
    
    with c2:
        farmland_change = comparison["farmland"]["absolute_change"]
        change_color = "normal"
        if farmland_change > 5:
            change_color = "inverse"  # Red for increase in farmland (at expense of trees)
        
        st.metric(
            label="Farmland Coverage Change",
            value=f"{farmland_change:.2f}%",
            delta=f"{comparison['farmland']['relative_change']:.2f}%",
            delta_color=change_color
        )
    
    with c3:
        # Determine if deforestation likely occurred
        if tree_change < -5 and farmland_change > 0:
            st.error("⚠️ Significant vegetation change detected. Tree cover has decreased while farmland has increased.")
        elif tree_change < -5:
            st.warning("⚠️ Tree cover has decreased significantly.")
        elif tree_change > 5:
            st.success("✅ Tree cover has increased.")
        else:
            st.info("No significant changes detected in tree cover.")
    
    # Bar chart comparing tree and farmland percentages
    st.subheader("Visual Comparison")
    
    # Bar chart for percentages
    fig, ax = plt.subplots(figsize=(10, 5))
    
    classes = ["Tree", "Farmland"]
    earlier_values = [comparison["tree"]["earlier_percentage"], comparison["farmland"]["earlier_percentage"]]
    later_values = [comparison["tree"]["later_percentage"], comparison["farmland"]["later_percentage"]]
    
    x = np.arange(len(classes))
    width = 0.35
    
    ax.bar(x - width/2, earlier_values, width, label='First Image')
    ax.bar(x + width/2, later_values, width, label='Second Image')
    
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Tree and Farmland Coverage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    
    st.pyplot(fig)


# import streamlit as st
# from roboflow import Roboflow
# import supervision as sv
# import cv2
# import numpy as np
# from PIL import Image
# import io
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from datetime import datetime, timedelta
# import time

# # Set page configuration
# st.set_page_config(
#     page_title="Simple Webcam Monitoring",
#     page_icon="🌳",
#     layout="wide"
# )

# # Create directory for storing captured images
# if not os.path.exists("captured_frames"):
#     os.makedirs("captured_frames")

# # Add title and description
# st.title("Simple Webcam Monitoring")
# st.markdown("Capture frames and compare vegetation changes")

# # Sidebar configuration
# st.sidebar.header("Settings")

# # API key input with default value
# api_key = st.sidebar.text_input("Roboflow API Key", value="pjZtbsAzjhkBsKvruel1", type="password")

# # Model parameters
# confidence = st.sidebar.slider("Confidence Threshold", min_value=1, max_value=100, value=40)

# # Time interval for scheduling (in minutes)
# auto_capture_interval = st.sidebar.number_input("Auto-capture interval (minutes)", min_value=1, max_value=120, value=30)

# # Function to process the image
# def process_image(image_bytes, api_key, confidence, temp_filename):
#     # Initialize Roboflow
#     rf = Roboflow(api_key=api_key)
#     project = rf.workspace().project("deforestation-satellite-imagery-335n4")
#     model = project.version(3).model
    
#     # Save the image bytes to a temporary file
#     with open(temp_filename, "wb") as f:
#         f.write(image_bytes)
    
#     # Make prediction
#     result = model.predict(temp_filename, confidence=confidence).json()
    
#     # Extract detections
#     detections = sv.Detections.from_inference(result)
    
#     # Read the image for annotation
#     image = cv2.imread(temp_filename)
    
#     # Only use mask annotator (no labels)
#     mask_annotator = sv.MaskAnnotator()
    
#     # Apply only segmentation masks to the image
#     annotated_image = mask_annotator.annotate(scene=image, detections=detections)
    
#     # Convert from BGR to RGB
#     annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
#     # Calculate area percentages
#     image_area = image.shape[0] * image.shape[1]  # total pixel area
#     class_areas = {}
    
#     for pred in result["predictions"]:
#         class_name = pred["class"]
        
#         # Calculate approximate area from bounding box if polygon points not available
#         if "width" in pred and "height" in pred:
#             area = pred["width"] * pred["height"]
#         else:
#             # Fallback to points if available
#             if "points" in pred:
#                 points = np.array(pred["points"])
#                 area = cv2.contourArea(points)
#             else:
#                 area = 0
        
#         if class_name in class_areas:
#             class_areas[class_name] += area
#         else:
#             class_areas[class_name] = area
    
#     # Calculate percentages
#     class_percentages = {}
#     for class_name, area in class_areas.items():
#         percentage = (area / image_area) * 100
#         class_percentages[class_name] = percentage
    
#     return annotated_image_rgb, class_percentages

# # Function to compare results (focusing on tree and farmland only)
# def compare_results(earlier_percentages, later_percentages):
#     # Initialize with empty values for tree and farmland in case they are not detected
#     comparison = {
#         "tree": {
#             "earlier_percentage": 0,
#             "later_percentage": 0,
#             "absolute_change": 0,
#             "relative_change": 0
#         },
#         "farmland": {
#             "earlier_percentage": 0,
#             "later_percentage": 0,
#             "absolute_change": 0,
#             "relative_change": 0
#         }
#     }
    
#     # Process only tree and farmland classes
#     for class_name in ["tree", "farmland"]:
#         # Get values or default to 0 if class not detected
#         earlier_pct = earlier_percentages.get(class_name, 0)
#         later_pct = later_percentages.get(class_name, 0)
        
#         # Calculate absolute and relative changes
#         abs_change = later_pct - earlier_pct
#         rel_change = 0
#         if earlier_pct > 0:
#             rel_change = (abs_change / earlier_pct) * 100
        
#         comparison[class_name] = {
#             "earlier_percentage": earlier_pct,
#             "later_percentage": later_pct,
#             "absolute_change": abs_change,
#             "relative_change": rel_change
#         }
    
#     return comparison

# # Function to capture a frame from webcam
# def capture_frame(camera_index=0):
#     """Capture a single frame from the webcam"""
#     cap = cv2.VideoCapture(camera_index)
    
#     if not cap.isOpened():
#         st.error("Error: Could not open webcam.")
#         return None
    
#     # Capture a single frame
#     ret, frame = cap.read()
    
#     # Release the webcam
#     cap.release()
    
#     if not ret:
#         st.error("Error: Could not capture frame.")
#         return None
    
#     # Convert to PIL Image
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_image = Image.fromarray(frame_rgb)
    
#     # Save to buffer
#     buffer = io.BytesIO()
#     pil_image.save(buffer, format="JPEG")
#     image_bytes = buffer.getvalue()
    
#     return image_bytes

# # SIMPLER APPROACH: Use a scheduled capture time instead of background threads
# # Initialize session state
# if "mode" not in st.session_state:
#     st.session_state["mode"] = "ready"  # Modes: ready, scheduled, captured_both

# if "earlier_capture_time" not in st.session_state:
#     st.session_state["earlier_capture_time"] = None

# if "scheduled_capture_time" not in st.session_state:
#     st.session_state["scheduled_capture_time"] = None

# # Main layout
# camera_col, earlier_col, later_col = st.columns(3)

# # Camera column for taking pictures
# with camera_col:
#     st.header("Take Pictures")
    
#     # Camera selection
#     camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0, help="Select your webcam (0 for default)")
    
#     # Take earlier capture button
#     if st.button("Take First/Earlier Picture"):
#         with st.spinner("Capturing..."):
#             image_bytes = capture_frame(camera_index)
            
#             if image_bytes:
#                 # Save the earlier capture
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filename = f"captured_frames/earlier_{timestamp}.jpg"
                
#                 with open(filename, "wb") as f:
#                     f.write(image_bytes)
                
#                 # Store in session state
#                 st.session_state["earlier_image"] = {
#                     "bytes": image_bytes,
#                     "filename": filename,
#                     "timestamp": timestamp
#                 }
                
#                 # Record the capture time
#                 st.session_state["earlier_capture_time"] = datetime.now()
                
#                 # Reset mode to ready and clear any scheduled capture
#                 st.session_state["mode"] = "ready"
#                 st.session_state["scheduled_capture_time"] = None
                
#                 if "later_image" in st.session_state:
#                     del st.session_state["later_image"]
#                 if "results" in st.session_state:
#                     del st.session_state["results"]
                
#                 st.success("First picture captured!")
#                 st.rerun()
#             else:
#                 st.error("Failed to capture image.")
    
#     # Schedule later capture button (only show if first image is captured)
#     if "earlier_image" in st.session_state and st.session_state["mode"] == "ready":
#         st.subheader("Schedule Second Image")
        
#         if st.button(f"Schedule Second Image in {auto_capture_interval} minutes"):
#             scheduled_time = datetime.now() + timedelta(minutes=auto_capture_interval)
#             st.session_state["scheduled_capture_time"] = scheduled_time
#             st.session_state["mode"] = "scheduled"
#             st.success(f"Second image scheduled for {scheduled_time.strftime('%H:%M:%S')}")
#             st.rerun()
    
#     # Take later capture manually button
#     if "earlier_image" in st.session_state:
#         if st.button("Take Second/Later Picture Now"):
#             with st.spinner("Capturing..."):
#                 image_bytes = capture_frame(camera_index)
                
#                 if image_bytes:
#                     # Save the later capture
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     filename = f"captured_frames/later_{timestamp}.jpg"
                    
#                     with open(filename, "wb") as f:
#                         f.write(image_bytes)
                    
#                     # Store in session state
#                     st.session_state["later_image"] = {
#                         "bytes": image_bytes,
#                         "filename": filename,
#                         "timestamp": timestamp
#                     }
                    
#                     # Set mode to captured both
#                     st.session_state["mode"] = "captured_both"
#                     st.session_state["scheduled_capture_time"] = None
                    
#                     st.success("Second picture captured!")
#                     st.rerun()
#                 else:
#                     st.error("Failed to capture image.")
    
#     # Show scheduled time if in scheduled mode
#     if st.session_state["mode"] == "scheduled" and st.session_state["scheduled_capture_time"]:
#         scheduled_time = st.session_state["scheduled_capture_time"]
#         current_time = datetime.now()
        
#         # Check if it's time to capture
#         if current_time >= scheduled_time:
#             st.info("It's time to capture the second image!")
            
#             with st.spinner("Auto-capturing second image..."):
#                 image_bytes = capture_frame(camera_index)
                
#                 if image_bytes:
#                     # Save the later capture
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     filename = f"captured_frames/later_{timestamp}.jpg"
                    
#                     with open(filename, "wb") as f:
#                         f.write(image_bytes)
                    
#                     # Store in session state
#                     st.session_state["later_image"] = {
#                         "bytes": image_bytes,
#                         "filename": filename,
#                         "timestamp": timestamp
#                     }
                    
#                     # Set mode to captured both
#                     st.session_state["mode"] = "captured_both"
#                     st.session_state["scheduled_capture_time"] = None
                    
#                     st.success("Second picture auto-captured!")
#                     st.rerun()
#                 else:
#                     st.error("Failed to auto-capture. Please try manual capture.")
#                     st.session_state["mode"] = "ready"
#         else:
#             time_remaining = scheduled_time - current_time
#             minutes = int(time_remaining.total_seconds() // 60)
#             seconds = int(time_remaining.total_seconds() % 60)
            
#             st.info(f"Second image scheduled for {scheduled_time.strftime('%H:%M:%S')} ({minutes}m {seconds}s remaining)")
            
#             if st.button("Cancel Scheduled Capture"):
#                 st.session_state["mode"] = "ready"
#                 st.session_state["scheduled_capture_time"] = None
#                 st.warning("Scheduled capture cancelled")
#                 st.rerun()
            
#             # Add refresh button
#             if st.button("Refresh Countdown"):
#                 st.rerun()

#     # Compare button (only show if both images are available)
#     if "earlier_image" in st.session_state and "later_image" in st.session_state and "results" not in st.session_state:
#         if st.button("Compare Pictures", type="primary"):
#             with st.spinner("Processing and comparing images..."):
#                 # Process earlier image
#                 earlier_annotated, earlier_percentages = process_image(
#                     st.session_state["earlier_image"]["bytes"],
#                     api_key,
#                     confidence,
#                     "earlier_temp.jpg"
#                 )
                
#                 # Process later image
#                 later_annotated, later_percentages = process_image(
#                     st.session_state["later_image"]["bytes"],
#                     api_key,
#                     confidence,
#                     "later_temp.jpg"
#                 )
                
#                 # Compare results
#                 comparison = compare_results(earlier_percentages, later_percentages)
                
#                 # Store results in session state
#                 st.session_state["results"] = {
#                     "earlier_annotated": earlier_annotated,
#                     "later_annotated": later_annotated,
#                     "earlier_percentages": earlier_percentages,
#                     "later_percentages": later_percentages,
#                     "comparison": comparison
#                 }
                
#                 st.success("Analysis complete!")
#                 st.rerun()

# # Earlier image column
# with earlier_col:
#     st.header("First/Earlier Image")
    
#     if "earlier_image" in st.session_state:
#         # Show the original image
#         st.image(Image.open(io.BytesIO(st.session_state["earlier_image"]["bytes"])), 
#                 caption=f"Captured at: {st.session_state['earlier_image']['timestamp']}", 
#                 use_column_width=True)
        
#         # Show the annotated image if available
#         if "results" in st.session_state:
#             st.subheader("Tree/Farmland Detection")
#             st.image(st.session_state["results"]["earlier_annotated"], use_column_width=True)
            
#             # Show percentages
#             earlier_percentages = st.session_state["results"]["earlier_percentages"]
#             tree_pct = earlier_percentages.get("tree", 0)
#             farmland_pct = earlier_percentages.get("farmland", 0)
            
#             e_col1, e_col2 = st.columns(2)
#             with e_col1:
#                 st.metric("Tree Coverage", f"{tree_pct:.2f}%")
#             with e_col2:
#                 st.metric("Farmland Coverage", f"{farmland_pct:.2f}%")
#     else:
#         st.info("No first image captured yet. Use the 'Take First/Earlier Picture' button.")

# # Later image column
# with later_col:
#     st.header("Second/Later Image")
    
#     if "later_image" in st.session_state:
#         # Show the original image
#         st.image(Image.open(io.BytesIO(st.session_state["later_image"]["bytes"])), 
#                 caption=f"Captured at: {st.session_state['later_image']['timestamp']}",
#                 use_column_width=True)
        
#         # Show the annotated image if available
#         if "results" in st.session_state:
#             st.subheader("Tree/Farmland Detection")
#             st.image(st.session_state["results"]["later_annotated"], use_column_width=True)
            
#             # Show percentages
#             later_percentages = st.session_state["results"]["later_percentages"]
#             tree_pct = later_percentages.get("tree", 0)
#             farmland_pct = later_percentages.get("farmland", 0)
            
#             l_col1, l_col2 = st.columns(2)
#             with l_col1:
#                 st.metric("Tree Coverage", f"{tree_pct:.2f}%")
#             with l_col2:
#                 st.metric("Farmland Coverage", f"{farmland_pct:.2f}%")
#     else:
#         if st.session_state["mode"] == "scheduled":
#             scheduled_time = st.session_state["scheduled_capture_time"]
#             current_time = datetime.now()
#             time_remaining = scheduled_time - current_time
#             minutes = int(time_remaining.total_seconds() // 60)
#             seconds = int(time_remaining.total_seconds() % 60)
            
#             st.info(f"Second image will be captured at {scheduled_time.strftime('%H:%M:%S')} ({minutes}m {seconds}s remaining)")
#         else:
#             st.info("No second image captured yet. Either schedule a capture or take one manually.")

# # Display comparison results if available
# if "results" in st.session_state:
#     st.header("Comparison Results")
    
#     comparison = st.session_state["results"]["comparison"]
    
#     # Create three columns for comparison metrics
#     c1, c2, c3 = st.columns(3)
    
#     with c1:
#         tree_change = comparison["tree"]["absolute_change"]
#         change_color = "normal"
#         if tree_change < -5:
#             change_color = "inverse"  # Red for decrease in trees
        
#         st.metric(
#             label="Tree Coverage Change",
#             value=f"{tree_change:.2f}%",
#             delta=f"{comparison['tree']['relative_change']:.2f}%",
#             delta_color=change_color
#         )
    
#     with c2:
#         farmland_change = comparison["farmland"]["absolute_change"]
#         change_color = "normal"
#         if farmland_change > 5:
#             change_color = "inverse"  # Red for increase in farmland (at expense of trees)
        
#         st.metric(
#             label="Farmland Coverage Change",
#             value=f"{farmland_change:.2f}%",
#             delta=f"{comparison['farmland']['relative_change']:.2f}%",
#             delta_color=change_color
#         )
    
#     with c3:
#         # Determine if deforestation likely occurred
#         if tree_change < -5 and farmland_change > 0:
#             st.error("⚠️ Significant vegetation change detected. Tree cover has decreased while farmland has increased.")
#         elif tree_change < -5:
#             st.warning("⚠️ Tree cover has decreased significantly.")
#         elif tree_change > 5:
#             st.success("✅ Tree cover has increased.")
#         else:
#             st.info("No significant changes detected in tree cover.")
    
#     # Bar chart comparing tree and farmland percentages
#     st.subheader("Visual Comparison")
    
#     # Bar chart for percentages
#     fig, ax = plt.subplots(figsize=(10, 5))
    
#     classes = ["Tree", "Farmland"]
#     earlier_values = [comparison["tree"]["earlier_percentage"], comparison["farmland"]["earlier_percentage"]]
#     later_values = [comparison["tree"]["later_percentage"], comparison["farmland"]["later_percentage"]]
    
#     x = np.arange(len(classes))
#     width = 0.35
    
#     ax.bar(x - width/2, earlier_values, width, label='First Image')
#     ax.bar(x + width/2, later_values, width, label='Second Image')
    
#     ax.set_ylabel('Coverage (%)')
#     ax.set_title('Tree and Farmland Coverage Comparison')
#     ax.set_xticks(x)
#     ax.set_xticklabels(classes)
#     ax.legend()
    
#     st.pyplot(fig)

# # Add footer with explanation
# st.markdown("---")
# st.markdown("""
# ### How to use this app:
# 1. Take the first image using the 'Take First/Earlier Picture' button
# 2. Choose one of the following options:
#    - Schedule the second image capture for a later time (e.g., 30 minutes from now)
#    - Take the second image immediately with 'Take Second/Later Picture Now'
# 3. When both images are captured, click 'Compare Pictures' to analyze the changes
# 4. To refresh the countdown timer, click the 'Refresh Countdown' button

# ### Notes:
# - Keep this browser tab open for the scheduled capture to work
# - When it's time for the scheduled capture, the app will automatically take the second picture
# - You may need to interact with the page (by clicking a button) near the scheduled time for best results
# """)