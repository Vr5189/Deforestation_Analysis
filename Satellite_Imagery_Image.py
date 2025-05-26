# # import streamlit as st
# # from roboflow import Roboflow
# # import supervision as sv
# # import cv2
# # import numpy as np
# # from PIL import Image
# # import io

# # # Set page configuration
# # st.set_page_config(
# #     page_title="Deforestation Detection",
# #     page_icon="🌳",
# #     layout="wide"
# # )

# # # Add title and description
# # st.title("Deforestation Detection in Satellite Imagery")
# # st.markdown("Upload a satellite image to detect deforestation areas.")

# # # API key input with default value (you might want to use st.secrets in production)
# # api_key = st.sidebar.text_input("Roboflow API Key", value="pjZtbsAzjhkBsKvruel1", type="password")

# # # Model parameters
# # confidence = st.sidebar.slider("Confidence Threshold", min_value=1, max_value=100, value=40)
# # overlap = st.sidebar.slider("Overlap Threshold", min_value=1, max_value=100, value=30)

# # # File uploader
# # uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

# # # Function to process the image
# # def process_image(image_bytes, api_key, confidence, overlap):
# #     # Initialize Roboflow
# #     rf = Roboflow(api_key=api_key)
# #     project = rf.workspace().project("deforestation-satellite-imagery-335n4")
# #     model = project.version(3).model
    
# #     # Save the image bytes to a temporary file
# #     with open("temp_image.jpg", "wb") as f:
# #         f.write(image_bytes)
    
# #     # Make prediction
# #     result = model.predict("temp_image.jpg", confidence=confidence).json()
    
# #     # Extract labels and detections
# #     labels = [item["class"] for item in result["predictions"]]
# #     detections = sv.Detections.from_inference(result)
    
# #     # Read the image for annotation
# #     image = cv2.imread("temp_image.jpg")
    
# #     # Annotate the image
# #     label_annotator = sv.LabelAnnotator()
# #     mask_annotator = sv.MaskAnnotator()
    
# #     annotated_image = mask_annotator.annotate(scene=image, detections=detections)
# #     annotated_image = label_annotator.annotate(
# #         scene=annotated_image, detections=detections, labels=labels
# #     )
    
# #     # Convert from BGR to RGB
# #     annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
# #     return annotated_image_rgb, result

# # # Main logic
# # if uploaded_file is not None:
# #     # Convert the file to bytes
# #     image_bytes = uploaded_file.getvalue()
    
# #     # Create a column layout
# #     col1, col2 = st.columns(2)
    
# #     with col1:
# #         st.subheader("Original Image")
# #         image = Image.open(io.BytesIO(image_bytes))
# #         st.image(image, use_column_width=True)
    
# #     # Process button
# #     if st.button("Detect Deforestation"):
# #         with st.spinner('Processing image...'):
# #             try:
# #                 # Process the image
# #                 annotated_image, result = process_image(image_bytes, api_key, confidence, overlap)
                
# #                 # Display results
# #                 with col2:
# #                     st.subheader("Detected Deforestation")
# #                     st.image(annotated_image, use_column_width=True)
                
# #                 # Show prediction details
# #                 st.subheader("Detection Results")
                
# #                 if "predictions" in result and len(result["predictions"]) > 0:
# #                     # Create a summary of detected classes
# #                     classes = {}
# #                     for pred in result["predictions"]:
# #                         class_name = pred["class"]
# #                         confidence = pred["confidence"]
# #                         if class_name in classes:
# #                             classes[class_name]["count"] += 1
# #                             classes[class_name]["confidences"].append(confidence)
# #                         else:
# #                             classes[class_name] = {
# #                                 "count": 1,
# #                                 "confidences": [confidence]
# #                             }
                    
# #                     # Display summary
# #                     for class_name, data in classes.items():
# #                         avg_confidence = sum(data["confidences"]) / len(data["confidences"])
# #                         st.write(f"Detected {data['count']} instances of '{class_name}' with average confidence of {avg_confidence:.2f}")
                    
# #                     # Show detailed predictions in an expander
# #                     with st.expander("View Detailed Predictions"):
# #                         st.json(result["predictions"])
# #                 else:
# #                     st.info("No deforestation detected in the image.")
                    
# #             except Exception as e:
# #                 st.error(f"Error processing the image: {e}")
# # else:
# #     # Display example image
# #     st.info("Please upload an image to begin detection.")
    
# # # Add footer
# # st.sidebar.markdown("---")
# # st.sidebar.info(
# #     "This app uses Roboflow's deforestation detection model to identify deforested "
# #     "areas in satellite imagery. Adjust the confidence and overlap thresholds to "
# #     "fine-tune the detection results."
# # )


# import streamlit as st
# from roboflow import Roboflow
# import supervision as sv
# import cv2
# import numpy as np
# from PIL import Image
# import io

# # Set page configuration
# st.set_page_config(
#     page_title="Deforestation Detection",
#     page_icon="🌳",
#     layout="wide"
# )

# # Add title and description
# st.title("Deforestation Detection in Satellite Imagery")
# st.markdown("Upload a satellite image to detect deforestation areas.")

# # API key input with default value (you might want to use st.secrets in production)
# api_key = st.sidebar.text_input("Roboflow API Key", value="pjZtbsAzjhkBsKvruel1", type="password")

# # Model parameters
# confidence = st.sidebar.slider("Confidence Threshold", min_value=1, max_value=100, value=40)
# overlap = st.sidebar.slider("Overlap Threshold", min_value=1, max_value=100, value=30)

# # File uploader
# uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

# # Function to process the image
# def process_image(image_bytes, api_key, confidence, overlap):
#     # Initialize Roboflow
#     rf = Roboflow(api_key=api_key)
#     project = rf.workspace().project("deforestation-satellite-imagery-335n4")
#     model = project.version(3).model
    
#     # Save the image bytes to a temporary file
#     with open("temp_image.jpg", "wb") as f:
#         f.write(image_bytes)
    
#     # Make prediction
#     result = model.predict("temp_image.jpg", confidence=confidence).json()
    
#     # Extract detections
#     detections = sv.Detections.from_inference(result)
    
#     # Read the image for annotation
#     image = cv2.imread("temp_image.jpg")
    
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
    
#     return annotated_image_rgb, result, class_percentages

# # Main logic
# if uploaded_file is not None:
#     # Convert the file to bytes
#     image_bytes = uploaded_file.getvalue()
    
#     # Create a column layout
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Original Image")
#         image = Image.open(io.BytesIO(image_bytes))
#         st.image(image, use_column_width=True)
    
#     # Process button
#     if st.button("Detect Deforestation"):
#         with st.spinner('Processing image...'):
#             try:
#                 # Process the image
#                 annotated_image, result, class_percentages = process_image(image_bytes, api_key, confidence, overlap)
                
#                 # Display results
#                 with col2:
#                     st.subheader("Segmentation Result")
#                     st.image(annotated_image, use_column_width=True)
                
#                 # Show prediction details
#                 st.subheader("Detection Metrics")
                
#                 if "predictions" in result and len(result["predictions"]) > 0:
#                     # Create metrics display
#                     metric_cols = st.columns(len(class_percentages))
                    
#                     # Display class percentages as metrics
#                     for i, (class_name, percentage) in enumerate(class_percentages.items()):
#                         with metric_cols[i]:
#                             st.metric(
#                                 label=f"{class_name}",
#                                 value=f"{percentage:.2f}%",
#                                 help=f"Approximate percentage of the image area covered by {class_name}"
#                             )
                    
#                     # Create a summary of detected classes
#                     st.subheader("Detailed Statistics")
#                     classes = {}
#                     for pred in result["predictions"]:
#                         class_name = pred["class"]
#                         confidence = pred["confidence"]
#                         if class_name in classes:
#                             classes[class_name]["count"] += 1
#                             classes[class_name]["confidences"].append(confidence)
#                         else:
#                             classes[class_name] = {
#                                 "count": 1,
#                                 "confidences": [confidence]
#                             }
                    
#                     # Display statistics in a dataframe
#                     stats_data = []
#                     for class_name, data in classes.items():
#                         avg_confidence = sum(data["confidences"]) / len(data["confidences"])
#                         max_confidence = max(data["confidences"])
#                         min_confidence = min(data["confidences"])
#                         stats_data.append({
#                             "Class": class_name,
#                             "Count": data["count"],
#                             "Area %": f"{class_percentages.get(class_name, 0):.2f}%",
#                             "Avg Confidence": f"{avg_confidence:.2f}",
#                             "Max Confidence": f"{max_confidence:.2f}",
#                             "Min Confidence": f"{min_confidence:.2f}",
#                         })
                    
#                     st.dataframe(stats_data)
                    
#                     # Show detailed predictions in an expander
#                     with st.expander("View Detailed Predictions"):
#                         st.json(result["predictions"])
#                 else:
#                     st.info("No deforestation detected in the image.")
                    
#             except Exception as e:
#                 st.error(f"Error processing the image: {e}")
# else:
#     # Display example image
#     st.info("Please upload an image to begin detection.")
    
# # Add footer
# st.sidebar.markdown("---")
# st.sidebar.info(
#     "This app uses Roboflow's deforestation detection model to identify deforested "
#     "areas in satellite imagery. The visualization shows only segmentation masks "
#     "without labels. The metrics show the percentage of each detected class."
# )


import streamlit as st
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Deforestation Analysis Over Time",
    page_icon="🌳",
    layout="wide"
)

# Add title and description
st.title("Deforestation Analysis Over Time")
st.markdown("Upload two satellite images from different time periods to analyze changes in Tree cover and farmland.")

# Sidebar configuration
st.sidebar.header("Model Configuration")

# API key input with default value
# api_key = st.sidebar.text_input("Roboflow API Key", value="pjZtbsAzjhkBsKvruel1", type="password")
api_key = "pjZtbsAzjhkBsKvruel1"
# Model parameters
confidence = st.sidebar.slider("Confidence Threshold", min_value=1, max_value=100, value=40)
# overlap = st.sidebar.slider("Overlap Threshold", min_value=1, max_value=100, value=30)

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
    
    return annotated_image_rgb, result, class_percentages, image.shape[0] * image.shape[1]

# Function to compare results (focusing on Tree and farmland only)
def compare_results(earlier_percentages, later_percentages, earlier_area, later_area):
    # Initialize with empty values for Tree and farmland in case they are not detected
    comparison = {
        "Tree": {
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
    
    # Process only Tree and farmland classes
    for class_name in ["Tree", "farmland"]:
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
    
    # Add ratio of Tree to farmland
    if comparison["farmland"]["earlier_percentage"] > 0:
        comparison["Tree_to_farmland_earlier"] = comparison["Tree"]["earlier_percentage"] / comparison["farmland"]["earlier_percentage"]
    else:
        comparison["Tree_to_farmland_earlier"] = float('inf') if comparison["Tree"]["earlier_percentage"] > 0 else 0
        
    if comparison["farmland"]["later_percentage"] > 0:
        comparison["Tree_to_farmland_later"] = comparison["Tree"]["later_percentage"] / comparison["farmland"]["later_percentage"]
    else:
        comparison["Tree_to_farmland_later"] = float('inf') if comparison["Tree"]["later_percentage"] > 0 else 0
    
    return comparison

# Create tabs for the workflow
tab1, tab2, tab3 = st.tabs(["1. Upload Images", "2. View Results", "3. Compare Changes"])

# Tab 1: Upload images
with tab1:
    st.header("Upload Satellite Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Earlier Image")
        earlier_label = st.text_input("Label for earlier image (e.g., '2020')", "Earlier")
        earlier_file = st.file_uploader("Choose earlier satellite image", type=["jpg", "jpeg", "png"], key="earlier")
        
        if earlier_file:
            st.session_state["earlier_image"] = earlier_file.getvalue()
            st.session_state["earlier_label"] = earlier_label
            st.image(Image.open(io.BytesIO(st.session_state["earlier_image"])), use_column_width=True)
    
    with col2:
        st.subheader("Later Image")
        later_label = st.text_input("Label for later image (e.g., '2023')", "Later")
        later_file = st.file_uploader("Choose later satellite image", type=["jpg", "jpeg", "png"], key="later")
        
        if later_file:
            st.session_state["later_image"] = later_file.getvalue()
            st.session_state["later_label"] = later_label
            st.image(Image.open(io.BytesIO(st.session_state["later_image"])), use_column_width=True)
    
    if earlier_file and later_file:
        if st.button("Process Images", key="process_button"):
            st.session_state["process_clicked"] = True
            with st.spinner("Processing images..."):
                # Process earlier image
                earlier_annotated, earlier_result, earlier_percentages, earlier_area = process_image(
                    st.session_state["earlier_image"], 
                    api_key, 
                    confidence,
                    "earlier_temp.jpg"
                )
                
                # Process later image
                later_annotated, later_result, later_percentages, later_area = process_image(
                    st.session_state["later_image"], 
                    api_key, 
                    confidence,
                    "later_temp.jpg"
                )
                
                # Store results in session state
                st.session_state["earlier_annotated"] = earlier_annotated
                st.session_state["earlier_result"] = earlier_result
                st.session_state["earlier_percentages"] = earlier_percentages
                st.session_state["earlier_area"] = earlier_area
                
                st.session_state["later_annotated"] = later_annotated
                st.session_state["later_result"] = later_result
                st.session_state["later_percentages"] = later_percentages
                st.session_state["later_area"] = later_area
                
                # Compute comparison (focused on Tree and farmland)
                st.session_state["comparison"] = compare_results(
                    earlier_percentages, 
                    later_percentages,
                    earlier_area,
                    later_area
                )
                
                st.success("Processing complete! Go to the 'View Results' and 'Compare Changes' tabs to see the analysis.")

# Tab 2: View segmentation results
with tab2:
    st.header("Segmentation Results")
    
    if "process_clicked" in st.session_state and st.session_state["process_clicked"]:
        col1, col2 = st.columns(2)
        
        with col1:
            if "earlier_label" in st.session_state:
                st.subheader(f"Earlier Image ({st.session_state['earlier_label']})")
            else:
                st.subheader("Earlier Image")
                
            if "earlier_annotated" in st.session_state:
                st.image(st.session_state["earlier_annotated"], use_column_width=True)
                
                # Display metrics for earlier image (focusing on Tree and farmland)
                if "earlier_percentages" in st.session_state:
                    st.write("Land Coverage:")
                    
                    # Filter for Tree and farmland only
                    earlier_percentages = st.session_state["earlier_percentages"]
                    Tree_pct = earlier_percentages.get("Tree", 0)
                    farmland_pct = earlier_percentages.get("farmland", 0)
                    
                    # Display metrics
                    col1a, col1b = st.columns(2)
                    with col1a:
                        st.metric("Tree Coverage", f"{Tree_pct:.2f}%")
                    with col1b:
                        st.metric("Farmland Coverage", f"{farmland_pct:.2f}%")
        
        with col2:
            if "later_label" in st.session_state:
                st.subheader(f"Later Image ({st.session_state['later_label']})")
            else:
                st.subheader("Later Image")
                
            if "later_annotated" in st.session_state:
                st.image(st.session_state["later_annotated"], use_column_width=True)
                
                # Display metrics for later image (focusing on Tree and farmland)
                if "later_percentages" in st.session_state:
                    st.write("Land Coverage:")
                    
                    # Filter for Tree and farmland only
                    later_percentages = st.session_state["later_percentages"]
                    Tree_pct = later_percentages.get("Tree", 0)
                    farmland_pct = later_percentages.get("farmland", 0)
                    
                    # Display metrics
                    col2a, col2b = st.columns(2)
                    with col2a:
                        st.metric("Tree Coverage", f"{Tree_pct:.2f}%")
                    with col2b:
                        st.metric("Farmland Coverage", f"{farmland_pct:.2f}%")
    else:
        st.info("Please upload and process images in the 'Upload Images' tab first.")

# Tab 3: Compare changes
with tab3:
    st.header("Tree and Farmland Analysis")
    
    if "process_clicked" in st.session_state and st.session_state["process_clicked"]:
        if "comparison" in st.session_state:
            comparison = st.session_state["comparison"]
            
            # Create comparison metrics
            st.subheader("Coverage Changes")
            
            col1, col2 = st.columns(2)
            
            with col1:
                Tree_change = comparison["Tree"]["absolute_change"]
                change_color = "normal"
                if Tree_change < -5:
                    change_color = "inverse"  # Red for decrease in Trees
                
                st.metric(
                    label="Tree Coverage Change",
                    value=f"{Tree_change:.2f}%",
                    delta=f"{comparison['Tree']['relative_change']:.2f}%",
                    delta_color=change_color
                )
            
            with col2:
                farmland_change = comparison["farmland"]["absolute_change"]
                change_color = "normal"
                if farmland_change > 5:
                    change_color = "inverse"  # Red for increase in farmland (at expense of Trees)
                
                st.metric(
                    label="Farmland Coverage Change",
                    value=f"{farmland_change:.2f}%",
                    delta=f"{comparison['farmland']['relative_change']:.2f}%",
                    delta_color=change_color
                )
            
            # Tree to Farmland Ratio
            st.subheader("Tree to Farmland Ratio")
            
            earlier_ratio = comparison.get("Tree_to_farmland_earlier", 0)
            later_ratio = comparison.get("Tree_to_farmland_later", 0)
            
            col1, col2 = st.columns(2)
            with col1:
                if earlier_ratio == float('inf'):
                    st.metric("Earlier Ratio", "∞")
                else:
                    st.metric("Earlier Ratio", f"{earlier_ratio:.2f}")
            
            with col2:
                if later_ratio == float('inf'):
                    st.metric("Later Ratio", "∞")
                else:
                    st.metric("Later Ratio", f"{later_ratio:.2f}")
            
            # Create visualization
            st.subheader("Visual Comparison")
            
            # Bar chart for percentages
            fig, ax = plt.subplots(figsize=(10, 6))
            
            classes = ["Tree", "Farmland"]
            earlier_values = [comparison["Tree"]["earlier_percentage"], comparison["farmland"]["earlier_percentage"]]
            later_values = [comparison["Tree"]["later_percentage"], comparison["farmland"]["later_percentage"]]
            
            x = np.arange(len(classes))
            width = 0.35
            
            earlier_label = st.session_state.get("earlier_label", "Earlier")
            later_label = st.session_state.get("later_label", "Later")
            
            ax.bar(x - width/2, earlier_values, width, label=f'{earlier_label}')
            ax.bar(x + width/2, later_values, width, label=f'{later_label}')
            
            ax.set_ylabel('Coverage (%)')
            ax.set_title('Tree and Farmland Coverage Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(classes)
            ax.legend()
            
            st.pyplot(fig)
            
            # Analysis summary
            st.subheader("Analysis Summary")
            
            # Check for deforestation
            if comparison["Tree"]["absolute_change"] < -5 and comparison["farmland"]["absolute_change"] > 0:
                st.error("⚠️ Significant deforestation detected: Tree cover has decreased while farmland has increased.")
                
                # Calculate percent of Tree area converted
                Tree_loss = abs(comparison["Tree"]["absolute_change"])
                farmland_gain = comparison["farmland"]["absolute_change"] if comparison["farmland"]["absolute_change"] > 0 else 0
                
                if Tree_loss > 0:
                    conversion_estimate = min(Tree_loss, farmland_gain)
                    st.write(f"Approximately {conversion_estimate:.2f}% of the area appears to have been converted from Tree cover to farmland.")
            
            elif comparison["Tree"]["absolute_change"] < -5:
                st.warning("⚠️ Tree cover has decreased significantly, but farmland hasn't increased proportionally.")
            
            elif comparison["Tree"]["absolute_change"] > 5:
                st.success("✅ Tree cover has increased, indicating possible reforestation or natural growth.")
            
            else:
                st.info("No significant changes detected in the Tree cover or farmland distribution.")
            
            # Display recommendations based on analysis
            with st.expander("View Recommendations"):
                if comparison["Tree"]["absolute_change"] < -5:
                    st.write("### Recommendations for Addressing Deforestation:")
                    st.write("1. **Monitor** - Continue monitoring this area for further changes")
                    st.write("2. **Investigate** - Determine if the land use change was authorized")
                    st.write("3. **Remediate** - Consider reforestation efforts to balance agricultural expansion")
                    st.write("4. **Policies** - Evaluate existing policies on forest preservation and agricultural expansion")
                else:
                    st.write("### Recommendations:")
                    st.write("1. **Continue Monitoring** - Regularly check this area for future changes")
                    st.write("2. **Document** - Record current state as baseline for future comparisons")
    else:
        st.info("Please upload and process images in the 'Upload Images' tab first.")

# Add footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This app analyzes changes in Tree cover and farmland between two satellite images. "
    "The visualization focuses specifically on Tree-to-farmland conversion and provides "
    "metrics to evaluate potential deforestation."
)