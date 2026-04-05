import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import pytesseract
import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import numpy as np
import cv2
import io
import logging
import os
import re
from typing import Tuple, List, Optional, Any

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pytorch Image-to-Excel Pro", layout="wide", page_icon="📊")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stDataFrame { border: 1px solid #4a4a4a; border-radius: 10px; }
    h1, h2, h3 { color: #f0f2f6; font-family: 'Inter', sans-serif; }
    .stAlert { background-color: #1a1c24; border: 1px solid #3d414c; }
    </style>
""", unsafe_allow_html=True)

# --- MODEL CACHING ---
@st.cache_resource
def load_models():
    # Table Detection model
    det_processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    det_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    
    # Structure model
    str_processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")
    str_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")
    
    return det_processor, det_model, str_processor, str_model

# Helper for detection
def get_det_processor():
    return DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")

def get_det_model():
    return TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

# --- CORE PROCESSING ENGINES ---

def enhance_image(image: Image.Image) -> np.ndarray:
    """Hyper-Accuracy Enhancement: Scale 4x + Denoise + CLAHE + Adaptive Thresh"""
    # Convert PIL to CV2 (BGR)
    img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # 1. Extreme 4X Magnification for fine details
    h, w = img.shape[:2]
    img = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    
    # 2. Advanced Gray & Contrast (CLAHE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 3. Micro-Noise Removal (Non-Local Means)
    # Slow but extremely high quality cleaning for industrial scans
    gray = cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)
    
    # 4. Adaptive Thresholding to isolate ink clearly
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 25, 12
    )
    
    # 5. Fine Morphological Opening to clean character edges
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Return Inverted (Black text on White background)
    return 255 - processed

def get_structure_fallback(image: Image.Image) -> Tuple[List[List[str]], None]:
    # --- SUB-IMAGE CHUNKING FOR ULTRA-WIDE SCANS ---
    # This prevents the pixel truncation issue by processing in 5000px segments
    w, h = image.size
    chunk_width = 5000
    overlap = 200
    all_final_rows = []
    
    # Progress through width in chunks
    for x in range(0, w, chunk_width - overlap):
        right = min(x + chunk_width, w)
        chunk = image.crop((x, 0, right, h))
        
        # High-res enhancement for this specific chunk
        enhanced = enhance_image(chunk)
        
        # Configuration for Sparse Industrial Data
        # PSM 11 (Sparse text) + OEM 3 (Standard)
        data = pytesseract.image_to_data(enhanced, config='--psm 11 --oem 3', output_type=pytesseract.Output.DICT)
        
        blobs = []
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            # Only keep alphanumeric blobs or coordinate markers
            if txt and (any(c.isalnum() for c in txt) or "(" in txt or ")" in txt):
                # Adjust X coordinate relative to the original image scale (x4 in enhancement)
                blobs.append({
                    'text': txt, 
                    'x': data['left'][i] + (x * 4), 
                    'y': data['top'][i], 
                    'w': data['width'][i], 
                    'h': data['height'][i]
                })
        
        # Local row clustering for this chunk
        blobs.sort(key=lambda b: b['y'])
        chunk_rows = []
        if blobs:
            c_row = [blobs[0]]
            for b in blobs[1:]:
                # Tolerance based on character height
                if abs(b['y'] - c_row[-1]['y']) < (c_row[-1]['h'] or 40) * 0.7:
                    c_row.append(b)
                else:
                    chunk_rows.append(c_row)
                    c_row = [b]
            chunk_rows.append(c_row)
        
        all_final_rows.extend(chunk_rows)

    # Global row consolidation (joining chunks back into full-width rows)
    all_final_rows.sort(key=lambda r: r[0]['y'])
    consolidated = []
    if all_final_rows:
        cur_row = all_final_rows[0]
        for next_row in all_final_rows[1:]:
            # Vertically align based on Y overlap
            if abs(next_row[0]['y'] - cur_row[0]['y']) < 50:
                cur_row.extend(next_row)
            else:
                consolidated.append(cur_row)
                cur_row = next_row
        consolidated.append(cur_row)

    # Convert to parameter-value lists
    final_formatted = []
    for r_blobs in consolidated:
        r_blobs.sort(key=lambda b: b['x'])
        
        # Deduplicate overlaps from chunking transitions
        unique_blobs = []
        if r_blobs:
            unique_blobs.append(r_blobs[0])
            for b in r_blobs[1:]:
                # If blobs are within 10px X of each other, they are likely duplicates
                if b['x'] - unique_blobs[-1]['x'] > 10:
                    unique_blobs.append(b)
        
        # Merge blobs into column cells based on horizontal distance
        row_vals = []
        if unique_blobs:
            c_col = [unique_blobs[0]]
            for b in unique_blobs[1:]:
                # If gap is > 80px (scaled at 4x), it's a new parameter slot
                if b['x'] - (c_col[-1]['x'] + c_col[-1]['w']) > 80:
                    row_vals.append(" ".join([x['text'] for x in c_col]))
                    c_col = [b]
                else:
                    c_col.append(b)
            row_vals.append(" ".join([x['text'] for x in c_col]))
            final_formatted.append(row_vals)
            
    return final_formatted, None

def process_image(uploaded_file: Any, force_entire: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[Image.Image]]:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Industrial scans often need full area analysis
    table_box = [0, 0, image.size[0], image.size[1]]
    table_img = image.copy()
    
    # Extract structural data
    with st.spinner("Executing Hyper-Accuracy Scan..."):
        final_data, _ = get_structure_fallback(table_img)
        
    # Create detection visual
    visualized_img = image.copy()
    draw = ImageDraw.Draw(visualized_img)
    draw.rectangle(table_box, outline="red", width=5)
    
    # --- FINAL REFINED MAPPING ENGINE (UNIFICATION + NOISE FILTER) ---
    raw_data_rows = []
    max_params_found = 0
    
    for row_idx, row_cells in enumerate(final_data):
        row_id = f"Table{row_idx + 1}"
        
        # 1. UNIFY COORDINATES (Merge ( x y z ) into one cell)
        temp_vals = []
        is_merging = False
        buffer = []
        
        for cell in row_cells:
            val = str(cell).strip()
            if not val or val == "-/-/-": continue
            
            # Start coordinate grouping
            if "(" in val and not ")" in val:
                is_merging = True
                buffer.append(val)
            elif is_merging:
                buffer.append(val)
                if ")" in val:
                    temp_vals.append(" ".join(buffer))
                    buffer = []
                    is_merging = False
            else:
                temp_vals.append(val)
        
        if buffer: temp_vals.append(" ".join(buffer))

        # 2. FILTER HEADERS & FIX DIGIT CONFUSION
        filtered_vals = []
        for val in temp_vals:
            norm_val = val.upper()
            if "PARAMETER" in norm_val or "PARA" in norm_val: continue
            if "TABLE" in norm_val:
                row_id = val
                continue
            
            # Digit Correction Pass (0, 20 confusions)
            clean_val = val
            v_upper = val.upper()
            if v_upper == "O": clean_val = "0"
            elif v_upper in ["ZO", "ZU", "2O", "VV", "W", "VV.", "V V"]: clean_val = "20"
            
            # Placeholder handling: Preserve column slot
            if any(s in v_upper for s in ["-/-", "-I-", "-L-", "-/-/-"]):
                filtered_vals.append("")
                continue

            # Cull single character junk unless it's a digit
            if len(clean_val) == 1 and not clean_val.isdigit():
                continue
                
            filtered_vals.append(clean_val)
            
        if not filtered_vals: continue
        
        max_params_found = max(max_params_found, len(filtered_vals))
        raw_data_rows.append((row_id, filtered_vals))

    # Construct the headers dynamically based on data width
    headers = ["Table ID"] + [f"Parameter {i}" for i in range(1, max_params_found + 1)]
    final_rows = []
    
    for row_id, vals in raw_data_rows:
        matrix_row = ["NaN"] * (max_params_found + 1)
        matrix_row[0] = row_id
        
        for i, val in enumerate(vals):
            target_idx = i + 1
            if target_idx > max_params_found: break
            if matrix_row[target_idx] == "NaN":
                matrix_row[target_idx] = val
                    
        final_rows.append(matrix_row)

    if not final_rows:
        final_rows = [["No Data Found"] + ["NaN"] * max_params_found]

    df_matrix = pd.DataFrame(final_rows, columns=headers)
    return df_matrix, visualized_img

# --- UI LOGIC ---

def main():
    st.title("📊 Pytorch Image-to-Excel Pro")
    st.subheader("High-Fidelity Industrial Table Extraction")
    
    with st.sidebar:
        st.header("Upload Settings")
        uploaded_file = st.file_uploader("Upload Industrial Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])
        process_btn = st.button("🚀 Process Table", use_container_width=True)
        clear_btn = st.button("🧹 Clear All")
        
        if clear_btn:
            st.session_state.cumulative_df = pd.DataFrame()
            st.rerun()

    if uploaded_file and process_btn:
        df, vis_img = process_image(uploaded_file)
        
        if df is not None:
            st.success("Extraction Complete!")
            
            col1, col2 = st.tabs(["🚀 Extracted Matrix", "🖼️ Processed Output"])
            
            with col1:
                st.dataframe(df, use_container_width=True)
                
                # Excel Download with Styling
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                    workbook = writer.book
                    worksheet = writer.sheets['Sheet1']
                    
                    # Apply styling to headers
                    from openpyxl.styles import PatternFill, Font, Alignment
                    header_fill = PatternFill(start_color='800000', end_color='800000', fill_type='solid')
                    header_font = Font(color='FFFFFF', bold=True)
                    
                    for cell in worksheet[1]:
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = Alignment(horizontal='center')
                    
                    # Auto-adjust column widths
                    for col in worksheet.columns:
                        max_length = 0
                        column = col[0].column_letter
                        for cell in col:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except: pass
                        worksheet.column_dimensions[column].width = max_length + 2

                st.download_button(
                    label="📥 Download Structured Excel",
                    data=output.getvalue(),
                    file_name="industrial_extraction.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            with col2:
                st.image(vis_img, caption="Detection Region", use_container_width=True)
        else:
            st.error("Failed to process image. Please try another file.")

if __name__ == "__main__":
    main()
