# Fix OpenMP library conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
from PIL import Image
from db import ImageVectorDB
import base64
import io
import pandas as pd
import re
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Fashion Visual Search",
    page_icon="👚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .product-card {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 15px;
        transition: transform 0.2s;
        height: 100%;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .product-image {
        border-radius: 8px;
        object-fit: contain;
        height: 300px;
        width: 100%;
        margin-bottom: 10px;
    }
    .product-title {
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 5px;
        height: 40px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .product-price {
        font-weight: 700;
        font-size: 18px;
        color: #d32f2f;
    }
    .product-mrp {
        text-decoration: line-through;
        color: #757575;
        font-size: 14px;
        margin-left: 8px;
    }
    .product-discount {
        color: #388e3c;
        font-weight: 500;
        font-size: 14px;
    }
    .similarity-badge {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    .stButton>button {
        width: 100%;
    }
    .view-product-btn {
        background: #ff5722 !important;
        color: white !important;
        border: none !important;
        padding: 5px 10px !important;
        border-radius: 4px !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    .disabled-btn {
        background: #9e9e9e !important;
        color: white !important;
        border: none !important;
        padding: 5px 10px !important;
        border-radius: 4px !important;
        cursor: not-allowed !important;
        width: 100% !important;
    }
    .similarity-score {
        font-size: 14px;
        font-weight: bold;
        color: #1976d2;
        margin-top: 5px;
    }
    .progress-bar {
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        margin-top: 5px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: #1976d2;
        border-radius: 3px;
    }
    .category-selector {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- App Title and Description ---
st.title("👗 Fashion Visual Search Engine")
st.markdown("""
Upload an image of a fashion item to find similar products from our collection. 
Get inspired and discover your perfect style match!
""")

# --- Robust Pathing ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Define paths for both categories
DRESSES_IMAGE_DIR = os.path.join(APP_DIR, "dresses_images")
DRESSES_METADATA_CSV = os.path.join(APP_DIR, "data", "dresses_with_image_paths.csv")
JEANS_IMAGE_DIR = os.path.join(APP_DIR, "jeans_images")
JEANS_METADATA_CSV = os.path.join(APP_DIR, "data", "jeans_with_image_paths.csv")

# --- Database Loading ---
@st.cache_resource
def load_database(category):
    db = ImageVectorDB()
    with st.spinner(f"🔍 Loading {category} database..."):
        db.build_database_from_sources(
            image_folder=DRESSES_IMAGE_DIR if category == "Dresses" else JEANS_IMAGE_DIR,
            metadata_csv=DRESSES_METADATA_CSV if category == "Dresses" else JEANS_METADATA_CSV,
            st_ui=st,
            cache_dir="image_db_cache"  # Custom cache directory
        )
    return db

# --- Helper functions ---
def clean_metadata_value(value):
    """Clean metadata values by removing brackets and unwanted characters"""
    if value is None or value == '' or str(value).lower() == 'nan':
        return 'N/A'
    
    # Convert to string
    value_str = str(value)
    
    # Remove square brackets and curly brackets
    value_str = re.sub(r'[\[\]{}]', '', value_str)
    
    # Remove quotes
    value_str = re.sub(r'["\']', '', value_str)
    
    # Clean up extra spaces
    value_str = re.sub(r'\s+', ' ', value_str).strip()
    
    # If it looks like a JSON string, try to parse and extract meaningful content
    try:
        if value_str.startswith('{') or value_str.startswith('['):
            parsed = json.loads(str(value))
            if isinstance(parsed, dict):
                # Extract meaningful values from dict
                meaningful_values = []
                for k, v in parsed.items():
                    if v and str(v).lower() not in ['nan', 'null', 'none', '']:
                        meaningful_values.append(f"{k}: {v}")
                return ', '.join(meaningful_values) if meaningful_values else 'N/A'
            elif isinstance(parsed, list):
                # Join list items
                return ', '.join([str(item) for item in parsed if item])
    except:
        pass
    
    return value_str if value_str and value_str != 'N/A' else 'N/A'

def safe_get_metadata(meta, possible_keys, default='N/A'):
    """Try multiple possible keys for a metadata field and clean the result"""
    if not isinstance(possible_keys, list):
        possible_keys = [possible_keys]
    
    for key in possible_keys:
        value = meta.get(key)
        cleaned_value = clean_metadata_value(value)
        if cleaned_value != 'N/A':
            return cleaned_value
    return default

def format_price(price_data):
    """Format price from dictionary or raw value"""
    if isinstance(price_data, dict):
        price = price_data.get('INR', 'N/A')
        try:
            return f"₹{float(price):,.2f}" if price != 'N/A' else 'N/A'
        except:
            return 'N/A'
    try:
        return f"₹{float(price_data):,.2f}" if price_data else 'N/A'
    except:
        return str(price_data) if price_data else 'N/A'

def format_discount(discount_data):
    """Format discount with 2 decimal places"""
    try:
        discount = float(discount_data)
        return f"{discount:.2f}%"
    except:
        return "N/A"

def format_similarity(similarity_score):
    """Format similarity score with color coding"""
    similarity_percent = similarity_score * 100
    if similarity_percent >= 80:
        color = "#4CAF50"  # Green
    elif similarity_percent >= 60:
        color = "#8BC34A"  # Light green
    elif similarity_percent >= 40:
        color = "#FFC107"  # Amber
    else:
        color = "#F44336"  # Red
    
    return f"""
    <div class="similarity-score" style="color: {color}">
        {similarity_percent:.2f}% match
    </div>
    <div class="progress-bar">
        <div class="progress-fill" style="width: {similarity_percent}%; background: {color};"></div>
    </div>
    """

def create_product_card(result):
    """Create a styled product card for display"""
    meta = result['metadata']
    pdp_url = safe_get_metadata(meta, ['pdp_url', 'pcip_url'], 'N/A')
    
    # Decode the base64 image data
    img_data = base64.b64decode(meta['image_data'])
    img = Image.open(io.BytesIO(img_data))
    
    # Get product information
    product_name = safe_get_metadata(meta, ['product_name', 'title'])
    brand = safe_get_metadata(meta, ['brand'])
    
    # Price information
    selling_price = format_price(meta.get('selling_price'))
    mrp = format_price(meta.get('mrp'))
    discount = format_discount(safe_get_metadata(meta, ['discount', 'discount_percentage', 'discount']))
    
    # Create card
    card = f"""
    <div class="product-card">
        <img src="data:image/png;base64,{meta['image_data']}" class="product-image">
        <div class="product-title">{product_name}</div>
        <div style="color: #616161; font-size: 14px; margin-bottom: 8px;">by {brand}</div>
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <span class="product-price">{selling_price}</span>
            <span class="product-mrp">{mrp}</span>
            {f'<span class="product-discount">{discount} off</span>' if discount != 'N/A' else ''}
        </div>
        {format_similarity(result['similarity'])}
    </div>
    <div style="margin-top: 10px;">
            {f'<a href="{pdp_url}" target="_blank" style="text-decoration: none;"><button class="view-product-btn">View Product</button></a>' 
             if pdp_url != 'N/A' else 
             '<button class="disabled-btn" disabled>No Link Available</button>'}
        </div>
    """
    return card

# --- Sidebar ---
with st.sidebar:
    st.header("🔍 Search Options")
    
    # Add category selection
    category = st.radio(
        "Choose Category",
        ["Dresses", "Jeans"],
        index=0,
        key="category_select"
    )
    
    search_k = st.slider("Number of results", 3, 12, 6, help="How many similar items to show")
    similarity_threshold = st.slider("Minimum similarity", 0.0, 1.0, 0.5, 0.05, 
                                   help="Adjust how similar items must be to your query")
    
    st.markdown("---")
    st.header("ℹ️ Database Info")
    
    # Initialize database based on selected category
    try:
        db = load_database(category)
        st.info(f"**Current Category:** {category}")
        st.info(f"**Total Items:** {db.index.ntotal}")
    except Exception as e:
        st.error(f"Failed to load database: {str(e)}")
        st.stop()
    
    if st.button("🔄 Refresh Database"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    **Features:**
    - Visual similarity search
    - Price comparison
    - Style matching
    - Instant product discovery
    """)

# --- Main Search Area ---
st.header(f"🔎 Upload Your {category} Inspiration")

# File uploader with improved styling
query_file = st.file_uploader(
    "Drag & drop an image here or click to browse",
    type=["png", "jpg", "jpeg"],
    help=f"Upload an image of {category.lower()} you like"
)

if query_file:
    st.markdown("---")
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Style Inspiration")
        query_image = Image.open(query_file).convert("RGB")
        st.image(query_image, use_container_width=True)
        
        # Add some space
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("**Quick Actions**")
        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("💖 Save to Favorites"):
                st.success("Saved to your favorites!")
        with col1b:
            if st.button("🛍 View Similar Outfits"):
                st.info("Feature coming soon!")

    with col2:
        st.subheader(f"Recommended For You ({search_k} items)")
        
        with st.spinner("✨ Finding your perfect fashion matches..."):
            results = db.search_similar_images(query_image, k=search_k, threshold=similarity_threshold)
        
        if not results:
            st.info("No similar items found. Try another image or adjust the similarity threshold.")
        else:
            # Display results in responsive grid
            cols = st.columns(3)  # 3 columns for the grid
            
            for i, result in enumerate(results):
                with cols[i % 3]:
                    # Use HTML/CSS for better product cards
                    st.markdown(create_product_card(result), unsafe_allow_html=True)
                    
                    # Quick details expander
                    with st.expander("📝 Product Details"):
                        meta = result['metadata']
                        
                        # Get all the metadata with cleaning
                        product_id = safe_get_metadata(meta, ['product_id'])
                        sku = safe_get_metadata(meta, ['sku'])
                        category = safe_get_metadata(meta, ['category_id', 'category'])
                        department = safe_get_metadata(meta, ['department_id', 'department'])
                        style_info = safe_get_metadata(meta, ['style_attributes', 'style'])
                        description = safe_get_metadata(meta, ['description'])
                        meta_info = safe_get_metadata(meta, ['meta_info'])
                        launch_date = safe_get_metadata(meta, ['launch_on', 'launch_date'])
                        last_seen = safe_get_metadata(meta, ['last_seen_date', 'last_seen'])
                        
                        # Display the information only if it's meaningful
                        details_shown = False
                        
                        if product_id != 'N/A':
                            st.markdown(f"**Product ID:** {product_id}")
                            details_shown = True
                            
                        if sku != 'N/A':
                            st.markdown(f"**SKU:** {sku}")
                            details_shown = True
                            
                        if category != 'N/A':
                            st.markdown(f"**Category:** {category}")
                            details_shown = True
                            
                        if department != 'N/A':
                            st.markdown(f"**Department:** {department}")
                            details_shown = True
                            
                        if style_info != 'N/A':
                            st.markdown(f"**Style:** {style_info}")
                            details_shown = True
                        
                        if launch_date != 'N/A':
                            st.markdown(f"**Launch Date:** {launch_date}")
                            details_shown = True
                            
                        if last_seen != 'N/A':
                            st.markdown(f"**Last Seen:** {last_seen}")
                            details_shown = True
                        
                        if description != 'N/A':
                            st.markdown(f"**Description:** {description}")
                            details_shown = True
                        
                        if meta_info != 'N/A':
                            st.markdown(f"**Additional Info:** {meta_info}")
                            details_shown = True
                        
                        if not details_shown:
                            st.markdown("*No additional product details available*")

# --- Featured Collections ---
st.markdown("---")
st.header("🌟 Trending Collections")

# Featured collections with images and redirect URLs
featured_collections = [
    {
        "name": "Summer Dresses",
        "image": "https://media.karenmillen.com/i/karenmillen/bkk24436_chartreuse_xl?$product_image_category_page_horizontal_filters_desktop_2x$&fmt=webp",
        "url": "https://www.karenmillen.com/womens/edits/summer"
    },
    {
        "name": "Occasion Dresses",
        "image": "https://media.karenmillen.com/i/karenmillen/bkk24328_blue_xl?$product_image_category_page_horizontal_filters_desktop_2x$&fmt=webp",
        "url": "https://www.karenmillen.com/womens/dresses/occasion-dresses"
    },
    {
        "name": "Denim Wear",
        "image": "https://media.karenmillen.com/i/karenmillen/bkk24148_light%20blue_xl?$product_image_category_page_horizontal_filters_desktop_2x$&fmt=webp",
        "url": "https://www.karenmillen.com/womens/denim"
    },
    {
        "name": "Work Dresses",
        "image": "https://media.karenmillen.com/i/karenmillen/bkk15686_ivory_xl?$product_image_category_page_horizontal_filters_desktop_2x$&fmt=webp",
        "url": "https://www.karenmillen.com/womens/work/dresses"
    } 
]

# Display featured collections in 4 columns
cols = st.columns(4)
for i, collection in enumerate(featured_collections):
    with cols[i]:
        st.image(collection["image"], caption=collection["name"], use_container_width=True)
        if st.button("Explore", key=f"explore_{i}"):
            st.markdown(f'<meta http-equiv="refresh" content="0; url={collection["url"]}">', unsafe_allow_html=True)