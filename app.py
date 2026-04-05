import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
import pytesseract
import numpy as np
import cv2
import io
import logging
import re
from typing import Tuple, List, Optional, Any

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Industrial Table Extraction Pro", layout="wide", page_icon="📊")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stDataFrame { border: 1px solid #4a4a4a; border-radius: 10px; }
    h1, h2, h3 { color: #f0f2f6; font-family: 'Inter', sans-serif; }
    .stAlert { background-color: #1a1c24; border: 1px solid #3d414c; }
    </style>
""", unsafe_allow_html=True)

# --- CORE PROCESSING ENGINES ---

def enhance_image(image: Image.Image) -> np.ndarray:
    """Hyper-Accuracy Mode 2.0: Shadow Removal + 4X Scale + Denoise"""
    # 1. Convert to CV2
    img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # 2. Shadow Removal & Lighting Normalization (Critical)
    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    img = cv2.merge(result_planes)

    # 3. 4X Magnification
    h, w = img.shape[:2]
    img = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    
    # 4. Grayscale & CLAHE Enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 5. Advanced Cleaning
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
    
    # 6. Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 31, 15
    )
    
    # 7. Edge Sharpening
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return 255 - processed

def get_structure_fallback(image: Image.Image) -> Tuple[List[List[str]], None]:
    """Universal Chunking to maintain data fidelity in ultra-wide scans"""
    w, h = image.size
    chunk_width = 5000
    overlap = 200
    all_final_rows = []
    
    for x in range(0, w, chunk_width - overlap):
        right = min(x + chunk_width, w)
        chunk = image.crop((x, 0, right, h))
        enhanced = enhance_image(chunk)
        
        # High Res Industrial Scan
        data = pytesseract.image_to_data(enhanced, config='--psm 11 --oem 3', output_type=pytesseract.Output.DICT)
        
        blobs = []
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            if txt and (any(c.isalnum() for c in txt) or "(" in txt):
                blobs.append({
                    'text': txt, 
                    'x': data['left'][i] + (x * 4), 
                    'y': data['top'][i], 
                    'w': data['width'][i], 
                    'h': data['height'][i]
                })
        
        # Cluster per chunk
        blobs.sort(key=lambda b: b['y'])
        chunk_rows = []
        if blobs:
            c_row = [blobs[0]]
            for b in blobs[1:]:
                if abs(b['y'] - c_row[-1]['y']) < (c_row[-1]['h'] or 40) * 0.7:
                    c_row.append(b)
                else:
                    chunk_rows.append(c_row)
                    c_row = [b]
            chunk_rows.append(c_row)
        all_final_rows.extend(chunk_rows)

    # Vertical Consolidation
    all_final_rows.sort(key=lambda r: r[0]['y'])
    consolidated = []
    if all_final_rows:
        cur_row = all_final_rows[0]
        for next_row in all_final_rows[1:]:
            if abs(next_row[0]['y'] - cur_row[0]['y']) < 50:
                cur_row.extend(next_row)
            else:
                consolidated.append(cur_row)
                cur_row = next_row
        consolidated.append(cur_row)

    # Output List
    final_formatted = []
    for r_blobs in consolidated:
        r_blobs.sort(key=lambda b: b['x'])
        unique_blobs = []
        if r_blobs:
            unique_blobs.append(r_blobs[0])
            for b in r_blobs[1:]:
                if b['x'] - unique_blobs[-1]['x'] > 10:
                    unique_blobs.append(b)
        
        row_vals = []
        if unique_blobs:
            c_col = [unique_blobs[0]]
            for b in unique_blobs[1:]:
                if b['x'] - (c_col[-1]['x'] + c_col[-1]['w']) > 80:
                    row_vals.append(" ".join([x['text'] for x in c_col]))
                    c_col = [b]
                else:
                    c_col.append(b)
            row_vals.append(" ".join([x['text'] for x in c_col]))
            final_formatted.append(row_vals)
            
    return final_formatted, None

def process_image(uploaded_file: Any) -> Tuple[Optional[pd.DataFrame], Optional[Image.Image]]:
    image = Image.open(uploaded_file).convert("RGB")
    table_img = image.copy()
    
    with st.spinner("Executing Accuracy Scan..."):
        final_data, _ = get_structure_fallback(table_img)
        
    # --- MAPPING ENGINE ---
    raw_data_rows = []
    max_params = 0
    for row_idx, row_cells in enumerate(final_data):
        row_id = f"Parameter{row_idx + 1}"
        
        # 1. Unify Coordinates
        temp_vals = []
        buf = []
        merge = False
        for cell in row_cells:
            val = str(cell).strip()
            if not val or val == "-/-/-": continue
            if "(" in val and not ")" in val:
                merge = True; buf.append(val)
            elif merge:
                buf.append(val)
                if ")" in val: temp_vals.append(" ".join(buf)); buf = []; merge = False
            else: temp_vals.append(val)
        if buf: temp_vals.append(" ".join(buf))

        # 2. Cleanup & Dynamic Detection
        filtered = []
        for val in temp_vals:
            up = val.upper()
            if "PARAMETER" in up or "PARA" in up: continue
            if "PARAMETER" in up: row_id = val; continue
            
            # Numeric Correction
            clean = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', val)
            if up == "O": clean = "0"
            elif up in ["ZO", "ZU", "2O", "VV", "VV.", "W", "V V"]: clean = "20"
            
            # Slot Preservation
            if any(s in up for s in ["-/-", "-I-", "-L-", "-/-/-"]):
                filtered.append(""); continue
            if len(clean) == 1 and not clean.isdigit(): continue
            filtered.append(clean)
            
        if not filtered: continue
        max_params = max(max_params, len(filtered))
        raw_data_rows.append((row_id, filtered))

    # --- ORIENTATION REVERSAL (Vertical Headers, Horizontal Table Data) ---
    table_labels = [f"Table {i}" for i in range(1, max_params + 1)]
    all_headers = ["Parameter ID"] + table_labels
    
    # Initialize a matrix where each column is a Table
    data_matrix = []
    for header in all_headers:
        current_header_data = [header] # First cell in Excel row is the label
        for row_id, vals in raw_data_rows:
            # Map Table Data to this row
            if header == "Parameter ID":
                current_header_data.append(row_id)
            else:
                table_idx = all_headers.index(header) - 1
                if table_idx < len(vals):
                    current_header_data.append(vals[table_idx])
                else:
                    current_header_data.append("NaN")
        data_matrix.append(current_header_data)

    # Dynamic Column Names (Table1, Table2...)
    col_names = ["Identification"] + [r[0] for r in raw_data_rows]
    df_matrix = pd.DataFrame(data_matrix, columns=col_names)
    
    return df_matrix, image

# --- UI LOGIC ---

def main():
    st.title("📊 Pytorch Image-to-Excel Pro")
    st.subheader("High-Fidelity Industrial Table Extraction")
    
    with st.sidebar:
        st.header("Upload Settings")
        file = st.file_uploader("Upload Industrial Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])
        process_btn = st.button("🚀 Process Table", use_container_width=True)
        clear_btn = st.button("🧹 Clear All")
        if clear_btn: st.rerun()

    if file and process_btn:
        df, img = process_image(file)
        if df is not None:
            st.success("Extraction Complete!")
            col1, col2 = st.tabs(["🚀 Extracted Matrix", "🖼️ Processed Output"])
            
            with col1:
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Excel Export with Styling
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                    ws = writer.sheets['Sheet1']
                    from openpyxl.styles import PatternFill, Font, Alignment
                    # Styling for Transposed Matrix
                    maroon = PatternFill(start_color='800000', end_color='800000', fill_type='solid')
                    green = PatternFill(start_color='92D050', end_color='92D050', fill_type='solid')
                    font = Font(color='FFFFFF', bold=True)
                    
                    # Style the first column (Labels)
                    for row_idx in range(1, len(df) + 2):
                        cell = ws.cell(row=row_idx, column=1)
                        if row_idx == 1: cell.fill = maroon
                        else: cell.fill = green
                        cell.font = font
                        cell.alignment = Alignment(horizontal='center')
                    
                    # Style the first row (Parameter IDs)
                    for col_idx in range(2, len(df.columns) + 1):
                        cell = ws.cell(row=1, column=col_idx)
                        cell.fill = maroon
                        cell.font = font
                        cell.alignment = Alignment(horizontal='center')

                st.download_button("📥 Download Structured Excel", out.getvalue(), "industrial_data.xlsx")
            
            with col2: st.image(img, caption="Analyzed Scan")

if __name__ == "__main__":
    main()
