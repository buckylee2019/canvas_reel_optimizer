#!/usr/bin/env python3
"""
Streamlit interface for DPA (Dynamic Product Ads) Automation
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import tempfile
import zipfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import multiprocessing as mp
import os
import importlib
import sys

# Force reload of dpa_automation module to get latest changes
if 'dpa_automation' in sys.modules:
    importlib.reload(sys.modules['dpa_automation'])

from dpa_automation import DPAAutomator, ProductData, DPATemplate
from image_understanding import DPAImageAnalyzer, analyze_product_for_dpa, get_smart_template_suggestions

# Page configuration
st.set_page_config(
    page_title="DPA Automation Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'dpa_automator' not in st.session_state:
        st.session_state.dpa_automator = None
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = []
    if 'current_template' not in st.session_state:
        st.session_state.current_template = None

def load_automator():
    """Load DPA automator with configuration"""
    try:
        config_path = "dpa_config.json" if Path("dpa_config.json").exists() else None
        automator = DPAAutomator(config_path)
        return automator
    except Exception as e:
        st.error(f"Failed to initialize DPA Automator: {e}")
        return None

def render_sidebar():
    """Render sidebar with navigation and settings"""
    st.sidebar.title("🎯 DPA Studio")
    
    # Initialize current_page if not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📁 Catalog Upload"
    
    # Navigation - use session state as default, update session state when changed
    pages = ["📁 Catalog Upload", "🎨 Template Designer", "🚀 Batch Processing", "📈 Analytics"]
    
    # Find current index
    try:
        current_index = pages.index(st.session_state.current_page)
    except ValueError:
        current_index = 0
        st.session_state.current_page = pages[0]
    
    # Selectbox with current page as default
    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        pages,
        index=current_index
    )
    
    # Update session state if user changed selection
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
    
    st.sidebar.markdown("---")
    
    # Quick stats
    if st.session_state.processed_results:
        st.sidebar.markdown("### 📊 Quick Stats")
        total_products = len(st.session_state.processed_results)
        st.sidebar.metric("Products Processed", total_products)
        
        # Count successful generations
        successful = sum(1 for r in st.session_state.processed_results 
                        if r.get('assets', {}).get('images') or r.get('assets', {}).get('videos'))
        st.sidebar.metric("Successful Generations", successful)
    
    return st.session_state.current_page

def process_product_multiprocessing(product_data_dict):
    """
    Process a single product in a separate process (multiprocessing-safe)
    This function runs in isolation, avoiding shared memory issues
    """
    try:
        # Import inside function to avoid pickling issues
        import sys
        import os
        sys.path.append(os.getcwd())
        
        from dpa_automation import DPAAutomator, ProductData
        
        # Extract data from the dictionary
        row_dict = product_data_dict['row']
        template_id = product_data_dict['template_id']
        enable_optimization = product_data_dict['enable_optimization']
        config_path = product_data_dict['config_path']
        
        # Create fresh automator instance for this process
        automator = DPAAutomator(config_path=config_path)
        
        # Create product data
        product = ProductData(
            product_id=str(row_dict.get('product_id', '')),
            name=str(row_dict.get('name', '')),
            description=str(row_dict.get('description', '')),
            price=float(row_dict.get('price', 0)),
            currency=str(row_dict.get('currency', 'USD')),
            category=str(row_dict.get('category', '')),
            brand=str(row_dict.get('brand', '')),
            image_url=str(row_dict.get('image_url', '')),
            availability=str(row_dict.get('availability', 'in stock')),
            condition=str(row_dict.get('condition', 'new'))
        )
        
        # Find template
        template = next(t for t in automator.templates if t.template_id == template_id)
        
        # Create creative
        creative = automator.create_ad_creative(product, template)
        
        if enable_optimization:
            creative = automator.optimize_creative(creative, product)
        
        # Generate assets
        assets = automator.generate_ad_assets(product, creative, template)
        
        return {
            "success": True,
            "product": product.__dict__,
            "creative": creative.__dict__,
            "assets": assets,
            "error": None,
            "process_id": os.getpid()
        }
        
    except Exception as e:
        return {
            "success": False,
            "product": {"product_id": row_dict.get('product_id', 'unknown'), "name": row_dict.get('name', 'unknown')},
            "creative": None,
            "assets": None,
            "error": str(e),
            "process_id": os.getpid()
        }

def smart_column_mapping(df_columns):
    """Automatically detect column mappings based on common patterns"""
    mapping = {}
    
    # Common patterns for each field
    patterns = {
        'product_id': ['product_id', 'id', 'sku', 'asin', 'product_sku', 'item_id'],
        'name': ['name', 'title', 'product_name', 'item_name', 'product_title'],
        'description': ['description', 'desc', 'product_description', 'summary', 'details'],
        'price': ['price', 'cost', 'amount', 'value', 'unit_price'],
        'currency': ['currency', 'curr', 'currency_code'],
        'category': ['category', 'cat', 'product_category', 'type', 'class'],
        'brand': ['brand', 'manufacturer', 'make', 'company'],
        'image_url': ['image_url', 'image', 'img_url', 'picture', 'photo_url', 'thumbnail'],
        'availability': ['availability', 'stock', 'in_stock', 'available', 'status'],
        'condition': ['condition', 'state', 'quality', 'item_condition']
    }
    
    # Convert column names to lowercase for matching
    lower_columns = [col.lower() for col in df_columns]
    
    for field, possible_names in patterns.items():
        for pattern in possible_names:
            if pattern.lower() in lower_columns:
                # Find the actual column name (with original case)
                actual_col = df_columns[lower_columns.index(pattern.lower())]
                mapping[field] = actual_col
                break
        
        # If no exact match, try partial matching
        if field not in mapping:
            for pattern in possible_names:
                for col in df_columns:
                    if pattern.lower() in col.lower():
                        mapping[field] = col
                        break
                if field in mapping:
                    break
    
    return mapping

def render_catalog_upload():
    """Render catalog upload interface"""
    st.title("📁 Product Catalog Upload")
    
    # Initialize automator if not already done
    if st.session_state.dpa_automator is None:
        with st.spinner("Initializing DPA Automator..."):
            st.session_state.dpa_automator = load_automator()
    
    if st.session_state.dpa_automator is None:
        st.error("Failed to initialize DPA Automator. Please check your configuration.")
        return
    
    # Show initialization success
    st.success("✅ DPA Automator initialized successfully!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your product catalog",
        type=['csv', 'json', 'xlsx'],
        help="Supported formats: CSV, JSON, Excel"
    )
    
    if uploaded_file is not None:
        try:
            # Read file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} products from {uploaded_file.name}")
            
            # Display preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Smart column mapping with auto-detection
            st.subheader("🔗 Column Mapping")
            
            # Get smart mapping suggestions
            smart_mapping = smart_column_mapping(df.columns.tolist())
            
            # Show auto-detected mappings
            if smart_mapping:
                st.info("🤖 Auto-detected column mappings (you can modify if needed):")
                detected_fields = [f"**{field}** → {col}" for field, col in smart_mapping.items()]
                st.markdown("  \n".join(detected_fields))
            
            required_fields = ['product_id', 'name', 'description', 'price']
            optional_fields = ['currency', 'category', 'brand', 'image_url', 'availability', 'condition']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Required Fields**")
                field_mapping = {}
                
                for field in required_fields:
                    # Use smart mapping as default, or empty if not found
                    default_value = smart_mapping.get(field, '')
                    default_index = 0
                    
                    if default_value and default_value in df.columns:
                        default_index = list(df.columns).index(default_value) + 1  # +1 because of empty option
                    
                    field_mapping[field] = st.selectbox(
                        f"Map '{field}' to:",
                        options=[''] + list(df.columns),
                        index=default_index,
                        key=f"map_{field}"
                    )
            
            with col2:
                st.markdown("**Optional Fields**")
                
                for field in optional_fields:
                    # Use smart mapping as default, or empty if not found
                    default_value = smart_mapping.get(field, '')
                    default_index = 0
                    
                    if default_value and default_value in df.columns:
                        default_index = list(df.columns).index(default_value) + 1  # +1 because of empty option
                    
                    field_mapping[field] = st.selectbox(
                        f"Map '{field}' to:",
                        options=[''] + list(df.columns),
                        index=default_index,
                        key=f"map_{field}"
                    )
            
            # Validation
            missing_required = [field for field in required_fields if not field_mapping.get(field)]
            
            if missing_required:
                st.warning(f"⚠️ Missing required fields: {', '.join(missing_required)}")
            else:
                st.success("✅ All required fields mapped!")
                
                # Save mapped data
                if st.button("💾 Save Mapped Catalog", use_container_width=True):
                    # Create mapped dataframe
                    mapped_df = pd.DataFrame()
                    
                    for field, column in field_mapping.items():
                        if column:
                            mapped_df[field] = df[column]
                    
                    # Save to session state
                    st.session_state.catalog_data = mapped_df
                    
                    # Save to file
                    output_path = Path("mapped_catalog.csv")
                    mapped_df.to_csv(output_path, index=False)
                    
                    st.success(f"✅ Catalog saved as {output_path} - {len(mapped_df)} products ready!")
                    
                    # Show sample of mapped data
                    st.subheader("📋 Mapped Data Preview")
                    st.dataframe(mapped_df.head(), use_container_width=True)
                    
                    # Next step instructions
                    st.markdown("---")
                    st.info("""
                    ### 🚀 **Next Steps:**
                    
                    **Option 1 (Recommended):** Go to **🎨 Template Designer** in the sidebar to create custom templates for better results.
                    
                    **Option 2 (Quick Start):** Go directly to **🚀 Batch Processing** in the sidebar to start generating images with default templates.
                    
                    💡 **Tip:** Custom templates produce more targeted and effective product advertisements.
                    """)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

def render_template_designer():
    """Render template designer interface"""
    st.title("🎨 DPA Template Designer")
    
    if st.session_state.dpa_automator is None:
        st.error("🔧 DPA Automator not initialized yet.")
        st.info("💡 **Next Step:** Use the sidebar to navigate to **📁 Catalog Upload** and upload your product catalog first.")
        return
    
    # Template selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Existing Templates")
        
        template_options = {t.name: t for t in st.session_state.dpa_automator.templates}
        selected_template_name = st.selectbox(
            "Select template to edit:",
            options=['Create New'] + list(template_options.keys())
        )
        
        if selected_template_name != 'Create New':
            st.session_state.current_template = template_options[selected_template_name]
        else:
            st.session_state.current_template = None
    
    with col2:
        st.subheader("✏️ Template Editor")
        
        # Template form
        with st.form("template_form"):
            template_id = st.text_input(
                "Template ID",
                value=st.session_state.current_template.template_id if st.session_state.current_template else ""
            )
            
            name = st.text_input(
                "Template Name",
                value=st.session_state.current_template.name if st.session_state.current_template else ""
            )
            
            platform = st.selectbox(
                "Platform",
                options=['facebook', 'google', 'instagram', 'amazon'],
                index=['facebook', 'google', 'instagram', 'amazon'].index(
                    st.session_state.current_template.platform if st.session_state.current_template else 'facebook'
                )
            )
            
            ad_format = st.selectbox(
                "Ad Format",
                options=['single_image', 'carousel', 'video', 'collection'],
                index=['single_image', 'carousel', 'video', 'collection'].index(
                    st.session_state.current_template.ad_format if st.session_state.current_template else 'single_image'
                )
            )
            
            headline_template = st.text_area(
                "Headline Template",
                value=st.session_state.current_template.headline_template if st.session_state.current_template else "{brand} {name} - {price} {currency}",
                help="Use {field_name} for dynamic content"
            )
            
            description_template = st.text_area(
                "Description Template",
                value=st.session_state.current_template.description_template if st.session_state.current_template else "Discover {name}. {description} Shop now!",
                help="Use {field_name} for dynamic content"
            )
            
            image_style = st.text_area(
                "Image Style",
                value=getattr(st.session_state, 'suggested_image_style', '') or (st.session_state.current_template.image_style if st.session_state.current_template else "clean product shot on white background"),
                help="💡 Use 'Full Analysis' below to get AI-powered suggestions for optimal image styles"
            )
            
            image_negative_style = st.text_area(
                "Image Negative Prompts (Optional)",
                value=getattr(st.session_state.current_template, 'image_negative_style', '') if st.session_state.current_template else "blurry, low quality, distorted, watermark, text overlay",
                help="What to avoid in image generation (e.g., 'blurry, low quality, distorted')"
            )
            
            video_style = st.text_area(
                "Video Style",
                value=st.session_state.current_template.video_style if st.session_state.current_template else "product showcase with lifestyle context"
            )
            
            video_negative_style = st.text_area(
                "Video Negative Prompts (Optional)",
                value=getattr(st.session_state.current_template, 'video_negative_style', '') if st.session_state.current_template else "shaky, poor lighting, low resolution",
                help="What to avoid in video generation (e.g., 'shaky, poor lighting, low resolution')"
            )
            
            # Virtual Try-On Options
            st.markdown("### 🎭 Virtual Try-On Enhancement")
            
            use_virtual_try_on = st.checkbox(
                "Enable Virtual Try-On Enhancement",
                value=getattr(st.session_state, 'suggested_vto_enabled', False) or (getattr(st.session_state.current_template, 'use_virtual_try_on', False) if st.session_state.current_template else False),
                help="Use Nova Canvas Virtual Try-On to replace products in generated images for more realistic results. 💡 AI analysis can recommend optimal VTO settings."
            )
            
            vto_enhancement_type = st.selectbox(
                "Enhancement Type",
                options=["auto", "clothing", "placement"],
                index=["auto", "clothing", "placement"].index(getattr(st.session_state.current_template, 'vto_enhancement_type', 'auto') if st.session_state.current_template else 'auto'),
                help="Auto: Detect based on product category, Clothing: For apparel try-on, Placement: For product placement in scenes",
                disabled=not use_virtual_try_on
            )
            
            submitted = st.form_submit_button("💾 Save Template")
            
            if submitted and template_id and name:
                # Create new template with backward compatibility
                try:
                    # Try with negative style parameters (new version)
                    new_template = DPATemplate(
                        template_id=template_id,
                        name=name,
                        platform=platform,
                        ad_format=ad_format,
                        headline_template=headline_template,
                        description_template=description_template,
                        image_style=image_style,
                        video_style=video_style,
                        image_negative_style=image_negative_style,
                        video_negative_style=video_negative_style,
                        use_virtual_try_on=use_virtual_try_on,
                        vto_enhancement_type=vto_enhancement_type
                    )
                except TypeError:
                    # Fallback to old version without negative styles
                    new_template = DPATemplate(
                        template_id=template_id,
                        name=name,
                        platform=platform,
                        ad_format=ad_format,
                        headline_template=headline_template,
                        description_template=description_template,
                        image_style=image_style,
                        video_style=video_style
                    )
                    # Manually add negative style attributes
                    new_template.image_negative_style = image_negative_style
                    new_template.video_negative_style = video_negative_style
                    new_template.use_virtual_try_on = use_virtual_try_on
                    new_template.vto_enhancement_type = vto_enhancement_type
                
                # Add to automator (in real implementation, save to file)
                st.session_state.dpa_automator.templates.append(new_template)
                st.success(f"✅ Template '{name}' saved successfully!")
                
                # Next step instructions
                st.markdown("---")
                st.info("""
                ### 🚀 **Next Step:**
                
                Your custom template is now ready! Go to **🚀 Batch Processing** in the sidebar to start generating images using your template.
                
                💡 **Tip:** You can create multiple templates for different product categories or marketing campaigns.
                """)

def render_batch_processing():
    """Render batch processing interface"""
    st.title("🚀 Batch Processing")
    
    if st.session_state.dpa_automator is None:
        st.error("🔧 DPA Automator not initialized yet.")
        st.info("💡 **Next Step:** Use the sidebar to navigate to **📁 Catalog Upload** and upload your product catalog first.")
        return
    
    # Check for catalog data
    if not hasattr(st.session_state, 'catalog_data'):
        st.warning("⚠️ No catalog data found. Please upload a catalog first.")
        st.info("💡 **Next Step:** Use the sidebar to navigate to **📁 Catalog Upload** and upload your product catalog.")
        return
    
    # Processing configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Processing Configuration")
        
        # Template selection
        template_options = {t.name: t.template_id for t in st.session_state.dpa_automator.templates}
        selected_template = st.selectbox(
            "Select Template:",
            options=list(template_options.keys())
        )
        
        # Processing mode
        processing_mode = st.radio(
            "Processing Mode:",
            options=["Sequential (Safe)", "Parallel (Faster)"],
            index=1,  # Default to parallel
            help="Sequential is safer, Parallel is faster using separate processes"
        )
        
        # Parallel processing settings (only show if parallel selected)
        if processing_mode == "Parallel (Faster)":
            max_workers = st.slider("Parallel Workers", min_value=2, max_value=6, value=3, 
                                   help="Number of separate processes (2-6 recommended)")
            st.info(f"🚀 **Multiprocessing**: {max_workers} separate processes will run simultaneously")
        else:
            max_workers = 1
            st.info("🔄 **Sequential**: Products processed one at a time for maximum stability")
        
        # Progress update frequency
        progress_freq = st.slider("Progress Update Frequency", min_value=1, max_value=5, value=2,
                                 help="How often to update progress display")
        
        # Output format
        output_format = st.selectbox("Output Format", options=['json', 'csv'])
        
        # Processing options
        enable_optimization = st.checkbox("Enable AI Optimization", value=True)
        generate_images = st.checkbox("Generate Images", value=True)
        generate_videos = st.checkbox("Generate Videos", value=False)
    
    with col2:
        st.subheader("📊 Catalog Overview")
        
        df = st.session_state.catalog_data
        st.metric("Total Products", len(df))
        
        # Estimated processing time
        estimated_time_per_product = 90  # seconds (conservative estimate)
        sequential_time = len(df) * estimated_time_per_product
        
        if processing_mode == "Parallel (Faster)":
            parallel_time = (len(df) * estimated_time_per_product) / max_workers
            
            col_seq, col_par = st.columns(2)
            with col_seq:
                st.metric("Sequential Time", f"{sequential_time//60:.0f}m {sequential_time%60:.0f}s")
            with col_par:
                speedup_pct = ((sequential_time - parallel_time) / sequential_time * 100)
                st.metric("Parallel Time", f"{parallel_time//60:.0f}m {parallel_time%60:.0f}s", 
                         delta=f"-{speedup_pct:.0f}%")
        else:
            st.metric("Estimated Time", f"{sequential_time//60:.0f}m {sequential_time%60:.0f}s")
        
        # Category breakdown
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Products by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Start processing
    st.markdown("---")
    
    button_text = "🚀 Start Parallel Processing" if processing_mode == "Parallel (Faster)" else "🔄 Start Sequential Processing"
    
    if st.button(button_text, use_container_width=True, type="primary"):
        template_id = template_options[selected_template]
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        # Prepare data for processing
        total_products = len(st.session_state.catalog_data)
        processed_results = []
        failed_results = []
        
        start_time = time.time()
        
        if processing_mode == "Parallel (Faster)":
            # PARALLEL PROCESSING using multiprocessing
            status_text.text(f"🚀 Starting parallel processing with {max_workers} processes...")
            
            # Prepare data for multiprocessing (must be serializable)
            product_data_list = []
            for idx, row in st.session_state.catalog_data.iterrows():
                product_data_list.append({
                    'row': row.to_dict(),  # Convert to dict for serialization
                    'template_id': template_id,
                    'enable_optimization': enable_optimization,
                    'config_path': 'dpa_config.json'  # Pass config path instead of object
                })
            
            # Use multiprocessing Pool for parallel execution
            try:
                with mp.Pool(processes=max_workers) as pool:
                    # Submit all tasks and track progress
                    results_async = pool.map_async(process_product_multiprocessing, product_data_list)
                    
                    # Monitor progress
                    while not results_async.ready():
                        time.sleep(2)  # Check every 2 seconds
                        # Note: Can't get exact progress with Pool.map, but we show activity
                        status_text.text(f"🔄 Processing {total_products} products with {max_workers} processes...")
                    
                    # Get all results
                    all_results = results_async.get()
                    
                    # Process results
                    for idx, result in enumerate(all_results):
                        progress = (idx + 1) / total_products
                        progress_bar.progress(progress)
                        
                        if result["success"]:
                            processed_results.append(result)
                            status_text.text(f"✅ Completed {idx + 1}/{total_products}: {result['product']['name']} (Process {result['process_id']})")
                        else:
                            failed_results.append(result)
                            status_text.text(f"❌ Completed {idx + 1}/{total_products}: {result['product']['name']} (Failed)")
                        
                        # Update live results
                        if (idx + 1) % progress_freq == 0 or (idx + 1) == total_products:
                            with results_container.container():
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("✅ Successful", len(processed_results))
                                with col2:
                                    st.metric("❌ Failed", len(failed_results))
                                with col3:
                                    elapsed = time.time() - start_time
                                    st.metric("⏱️ Elapsed", f"{elapsed:.0f}s")
                    
            except Exception as e:
                st.error(f"Parallel processing failed: {e}")
                st.info("Falling back to sequential processing...")
                processing_mode = "Sequential (Safe)"  # Fallback
        
        if processing_mode == "Sequential (Safe)":
            # SEQUENTIAL PROCESSING (fallback or selected)
            status_text.text(f"🔄 Starting sequential processing...")
            
            for idx, (_, row) in enumerate(st.session_state.catalog_data.iterrows()):
                try:
                    # Update progress
                    progress = (idx + 1) / total_products
                    progress_bar.progress(progress)
                    
                    # Create product data
                    product = ProductData(
                        product_id=str(row.get('product_id', '')),
                        name=str(row.get('name', '')),
                        description=str(row.get('description', '')),
                        price=float(row.get('price', 0)),
                        currency=str(row.get('currency', 'USD')),
                        category=str(row.get('category', '')),
                        brand=str(row.get('brand', '')),
                        image_url=str(row.get('image_url', '')),
                        availability=str(row.get('availability', 'in stock')),
                        condition=str(row.get('condition', 'new'))
                    )
                    
                    status_text.text(f"🔄 Processing {idx + 1}/{total_products}: {product.name}")
                    
                    # Find template
                    template = next(t for t in st.session_state.dpa_automator.templates if t.template_id == template_id)
                    
                    # Create creative
                    creative = st.session_state.dpa_automator.create_ad_creative(product, template)
                    
                    if enable_optimization:
                        creative = st.session_state.dpa_automator.optimize_creative(creative, product)
                    
                    # Generate assets with progress callback
                    def progress_callback(message):
                        status_text.text(f"🔄 Processing {idx + 1}/{total_products}: {product.name} - {message}")
                    
                    assets = st.session_state.dpa_automator.generate_ad_assets(product, creative, template, progress_callback)
                    
                    result = {
                        "success": True,
                        "product": product.__dict__,
                        "creative": creative.__dict__,
                        "assets": assets,
                        "error": None
                    }
                    
                    processed_results.append(result)
                    status_text.text(f"✅ Completed {idx + 1}/{total_products}: {product.name} (Success)")
                    
                except Exception as e:
                    error_msg = str(e)
                    failed_result = {
                        "success": False,
                        "product": {"product_id": row.get('product_id', 'unknown'), "name": row.get('name', 'unknown')},
                        "creative": None,
                        "assets": None,
                        "error": error_msg
                    }
                    failed_results.append(failed_result)
                    status_text.text(f"❌ Completed {idx + 1}/{total_products}: {row.get('name', 'unknown')} (Failed)")
                    
                    # Show error but continue processing
                    st.error(f"Error processing {row.get('name', 'unknown')}: {error_msg}")
                
                # Update live results
                if (idx + 1) % progress_freq == 0 or (idx + 1) == total_products:
                    with results_container.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("✅ Successful", len(processed_results))
                        with col2:
                            st.metric("❌ Failed", len(failed_results))
                        with col3:
                            elapsed = time.time() - start_time
                            st.metric("⏱️ Elapsed", f"{elapsed:.0f}s")
        
        # Final results (common for both modes)
        total_time = time.time() - start_time
        
        # Save successful results
        st.session_state.processed_results = processed_results
        
        # Show completion summary
        progress_bar.progress(1.0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Successful", len(processed_results))
        with col2:
            st.metric("❌ Failed", len(failed_results))
        with col3:
            st.metric("⏱️ Total Time", f"{total_time:.0f}s")
        with col4:
            avg_time = total_time / total_products if total_products > 0 else 0
            st.metric("⚡ Avg per Product", f"{avg_time:.1f}s")
        
        if len(processed_results) > 0:
            mode_text = f"parallel processing with {max_workers} processes" if processing_mode == "Parallel (Faster)" else "sequential processing"
            st.success(f"🎉 Batch processing completed! Successfully generated assets for {len(processed_results)} products using {mode_text} in {total_time:.0f} seconds!")
            
            # Show performance improvement for parallel
            if processing_mode == "Parallel (Faster)" and max_workers > 1:
                sequential_estimate = total_products * (total_time / total_products) * max_workers
                speedup = sequential_estimate / total_time if total_time > 0 else 1
                st.info(f"🚀 **Performance**: ~{speedup:.1f}x faster than sequential processing!")
        
        if len(failed_results) > 0:
            st.warning(f"⚠️ {len(failed_results)} products failed to process.")
            
            # Show failed products
            with st.expander("❌ Failed Products Details"):
                for failed in failed_results:
                    st.error(f"**{failed['product']['name']}** (ID: {failed['product']['product_id']}): {failed['error']}")
        
        # Processing summary
        st.markdown("---")
        st.subheader("📊 Processing Summary")
        
        summary_data = {
            "Metric": ["Total Products", "Successful", "Failed", "Success Rate", "Total Time", "Processing Mode", "Workers/Processes", "Avg Time/Product"],
            "Value": [
                str(total_products),
                str(len(processed_results)),
                str(len(failed_results)),
                f"{(len(processed_results)/total_products*100):.1f}%" if total_products > 0 else "0%",
                f"{total_time:.0f}s",
                str(processing_mode),
                str(max_workers if processing_mode == "Parallel (Faster)" else 1),
                f"{avg_time:.1f}s"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Next step instructions
        if len(processed_results) > 0:
            st.markdown("---")
            st.success("""
            ### 🎉 **Processing Complete!**
            
            **Next Step:** Go to **📈 Analytics** in the sidebar to view your generated images in a beautiful gallery with download links.
            
            💡 **What you'll see:**
            - Visual gallery of all generated images
            - Performance metrics and success rates  
            - Presigned download links (24-hour expiration)
            - Detailed results table with all product information
            """)

def render_analytics():
    """Render analytics dashboard with generated images"""
    st.title("📈 Analytics Dashboard")
    
    # Try to load results from session state first, then from JSON files
    results = None
    
    if st.session_state.processed_results:
        results = st.session_state.processed_results
        st.info("📊 Showing results from current session")
    else:
        # Try to load from JSON files
        results_dir = Path("generated_ads")
        json_files = list(results_dir.glob("dpa_results_*.json"))
        
        if json_files:
            # Use the most recent JSON file
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            
            try:
                with open(latest_json, 'r') as f:
                    data = json.load(f)
                results = data.get('results', [])
                
                if results:
                    st.info(f"📁 Loaded results from: {latest_json.name} ({len(results)} products)")
                else:
                    st.warning(f"📁 JSON file {latest_json.name} contains no results")
            except Exception as e:
                st.error(f"❌ Error loading {latest_json.name}: {e}")
        
        if not results:
            st.info("No processed results available. Run batch processing first or check for JSON result files in generated_ads/")
            
            # Show available JSON files for debugging
            if json_files:
                with st.expander("🔍 Available Result Files"):
                    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
                        file_size = json_file.stat().st_size
                        mod_time = json_file.stat().st_mtime
                        st.write(f"• {json_file.name} ({file_size:,} bytes, modified: {mod_time})")
            return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", len(results))
    
    with col2:
        successful = sum(1 for r in results if r.get('assets', {}).get('images') or r.get('assets', {}).get('videos'))
        st.metric("Successful Generations", successful)
    
    with col3:
        total_images = sum(len(r.get('assets', {}).get('images', [])) for r in results)
        st.metric("Images Generated", total_images)
    
    with col4:
        # Count VTO-enhanced images
        vto_enhanced = sum(1 for r in results 
                          if r.get('assets', {}).get('metadata', {}).get('has_vto_enhancement', False))
        st.metric("VTO Enhanced", vto_enhanced)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Success rate by category
        category_data = {}
        for result in results:
            category = result.get('product', {}).get('category', 'Unknown')
            if category not in category_data:
                category_data[category] = {'total': 0, 'successful': 0}
            
            category_data[category]['total'] += 1
            if result.get('assets', {}).get('images') or result.get('assets', {}).get('videos'):
                category_data[category]['successful'] += 1
        
        categories = list(category_data.keys())
        success_rates = [category_data[cat]['successful'] / category_data[cat]['total'] * 100 for cat in categories]
        
        fig = px.bar(
            x=categories,
            y=success_rates,
            title="Success Rate by Category (%)",
            labels={'x': 'Category', 'y': 'Success Rate (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price distribution
        prices = [result.get('product', {}).get('price', 0) for result in results]
        
        fig = px.histogram(
            x=prices,
            title="Price Distribution",
            labels={'x': 'Price ($)', 'y': 'Count'},
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Generated Images Gallery
    st.subheader("🖼️ Generated Images Gallery")
    
    # Add presigned URL info
    if any(r.get('assets', {}).get('images', [{}])[0].get('s3_url') for r in results if r.get('assets', {}).get('images')):
        st.info("🔗 **Download Links**: Presigned URLs are generated for S3 images and expire in 24 hours for security.")
    
    # Filter and display options
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Category filter
        all_categories = list(set(r.get('product', {}).get('category', 'Unknown') for r in results))
        selected_category = st.selectbox("Filter by Category:", ["All"] + all_categories)
    
    with col2:
        # Product ID filter
        all_product_ids = list(set(r.get('product', {}).get('product_id', 'Unknown') for r in results))
        selected_product_id = st.selectbox("Filter by Product ID:", ["All"] + sorted(all_product_ids))
    
    with col3:
        # Generation method filter
        all_methods = list(set(
            img.get('method', 'unknown') 
            for r in results 
            for img in r.get('assets', {}).get('images', [])
        ))
        selected_method = st.selectbox("Filter by Method:", ["All"] + all_methods)
    
    with col4:
        # VTO comparison view
        show_vto_comparison = st.checkbox("Show VTO Comparison", value=False, 
                                         help="Show base and VTO-enhanced images side by side")
    
    with col5:
        # Images per row
        images_per_row = st.slider("Images per row:", 1, 6, 3)
        if show_vto_comparison and images_per_row < 2:
            st.warning("VTO comparison requires at least 2 columns")
            images_per_row = 2
    
    # Group images by product for VTO comparison
    if show_vto_comparison:
        st.markdown("### 🎭 Multi-Scene Generation Results")
        
        for result in results:
            product = result.get('product', {})
            assets = result.get('assets', {})
            
            # Apply category filter
            if selected_category != "All" and product.get('category') != selected_category:
                continue
            
            # Apply product ID filter
            if selected_product_id != "All" and product.get('product_id') != selected_product_id:
                continue
            
            # Group images by scene
            scene_groups = {}
            for img in assets.get('images', []):
                scene_name = img.get('scene_name', 'base_template')
                if scene_name not in scene_groups:
                    scene_groups[scene_name] = {'base': None, 'vto': None}
                
                if img.get('image_type') == 'scene_generated':
                    scene_groups[scene_name]['base'] = img
                elif img.get('image_type') == 'vto_scene_enhanced':
                    scene_groups[scene_name]['vto'] = img
            
            if scene_groups:
                st.markdown(f"**{product.get('name', 'Unknown Product')}** - {product.get('category', 'Unknown')}")
                
                # Show original product image if available
                original_image_path = assets.get('metadata', {}).get('original_product_image')
                if original_image_path and Path(original_image_path).exists():
                    with st.expander("📷 View Original Product Image", expanded=False):
                        col_orig, col_info = st.columns([1, 1])
                        with col_orig:
                            st.image(original_image_path, caption="Original Product Image", use_container_width=True)
                        with col_info:
                            st.markdown("**📊 Original Image Info:**")
                            try:
                                orig_img = Image.open(original_image_path)
                                st.write(f"• **Size:** {orig_img.size[0]} x {orig_img.size[1]} pixels")
                                st.write(f"• **Mode:** {orig_img.mode}")
                                file_size = Path(original_image_path).stat().st_size
                                st.write(f"• **File Size:** {file_size:,} bytes")
                            except Exception as e:
                                st.write(f"• **Error:** {e}")
                
                # Show AI analysis info if available
                total_scenes = assets.get('metadata', {}).get('total_scenes', len(scene_groups))
                if assets.get('metadata', {}).get('has_ai_analysis'):
                    st.info(f"🧠 AI Analysis Applied: Generated {total_scenes} separate scene images")
                
                # Display each scene
                for scene_name, scene_images in scene_groups.items():
                    base_img = scene_images['base']
                    vto_img = scene_images['vto']
                    
                    if base_img:
                        # Clean scene name for display
                        display_scene_name = scene_name.replace('ai_scene_', 'Scene ').replace('_', ' ').title()
                        if scene_name == 'base_template':
                            display_scene_name = 'Base Template Scene'
                        
                        st.markdown(f"#### 🎨 {display_scene_name}")
                        
                        # Show scene prompt
                        with st.expander(f"📝 View Scene Prompt: {display_scene_name}", expanded=False):
                            st.markdown("**🎨 Scene Prompt:**")
                            st.code(base_img.get('prompt', 'No prompt available'), language=None)
                            
                            if base_img.get('negative_prompt'):
                                st.markdown("**🚫 Negative Prompt:**")
                                st.code(base_img.get('negative_prompt'), language=None)
                            
                            # Show scene metadata
                            scene_idx = base_img.get('scene_index', 1)
                            total_scenes_img = base_img.get('total_scenes', 1)
                            st.caption(f"Scene {scene_idx} of {total_scenes_img}")
                        
                        # Display images side by side if VTO exists, otherwise single image
                        if vto_img:
                            col_base, col_vto = st.columns(2)
                            
                            with col_base:
                                st.markdown("**🎨 Base Scene**")
                                if Path(base_img.get('local_path', '')).exists():
                                    st.image(base_img['local_path'], use_container_width=True)
                                elif base_img.get('s3_url'):
                                    st.image(base_img['s3_url'], use_container_width=True)
                                
                                st.caption(f"Method: {base_img.get('method', 'unknown')}")
                                if base_img.get('s3_url'):
                                    st.markdown(f"[📥 Download Base]({base_img['s3_url']})")
                            
                            with col_vto:
                                st.markdown(f"**🎭 VTO Enhanced ({vto_img.get('vto_category', 'Unknown')})**")
                                if Path(vto_img.get('local_path', '')).exists():
                                    st.image(vto_img['local_path'], use_container_width=True)
                                elif vto_img.get('s3_url'):
                                    st.image(vto_img['s3_url'], use_container_width=True)
                                
                                st.caption(f"Method: {vto_img.get('method', 'unknown')}")
                                st.caption(f"Enhancement: {vto_img.get('enhancement', 'unknown')}")
                                if vto_img.get('s3_url'):
                                    st.markdown(f"[📥 Download VTO]({vto_img['s3_url']})")
                        else:
                            # Single scene image
                            if Path(base_img.get('local_path', '')).exists():
                                st.image(base_img['local_path'], use_container_width=True)
                            elif base_img.get('s3_url'):
                                st.image(base_img['s3_url'], use_container_width=True)
                            
                            st.caption(f"Method: {base_img.get('method', 'unknown')}")
                            if base_img.get('s3_url'):
                                st.markdown(f"[📥 Download Image]({base_img['s3_url']})")
                        
                        st.markdown("---")
                
                st.markdown("---")
            base_image = None
            vto_image = None
            
            for img in assets.get('images', []):
                if img.get('image_type') == 'base_generated':
                    base_image = img
                elif img.get('image_type') == 'vto_enhanced':
                    vto_image = img
            
            if base_image and vto_image:
                st.markdown(f"**{product.get('name', 'Unknown Product')}** - {product.get('category', 'Unknown')}")
                
                # Show prompts used for generation
                with st.expander("📝 View Generation Prompts", expanded=False):
                    col_prompt1, col_prompt2 = st.columns(2)
                    
                    with col_prompt1:
                        st.markdown("**🎨 Image Prompt:**")
                        image_prompt = base_image.get('prompt', 'No prompt available')
                        st.code(image_prompt, language=None)
                        
                        if base_image.get('negative_prompt'):
                            st.markdown("**🚫 Negative Prompt:**")
                            st.code(base_image.get('negative_prompt'), language=None)
                    
                    with col_prompt2:
                        st.markdown("**🎭 VTO Enhancement:**")
                        vto_category = vto_image.get('vto_category', 'Unknown')
                        enhancement_type = vto_image.get('enhancement', 'virtual_try_on')
                        st.write(f"• **Category:** {vto_category}")
                        st.write(f"• **Enhancement:** {enhancement_type}")
                        
                        # Show AI analysis if available
                        if 'ai_analysis' in assets.get('metadata', {}):
                            st.markdown("**🧠 AI Analysis Applied:**")
                            st.write("✅ Automatic background optimization")
                            st.write("✅ AI-generated mask prompts")
                            st.write("✅ Product-specific enhancement")
                
                col_base, col_vto = st.columns(2)
                
                with col_base:
                    st.markdown("**🎨 Base AI-Generated**")
                    if Path(base_image.get('local_path', '')).exists():
                        st.image(base_image['local_path'], use_container_width=True)
                    elif base_image.get('s3_url'):
                        st.image(base_image['s3_url'], use_container_width=True)
                    
                    # Show generation details
                    st.caption(f"Method: {base_image.get('method', 'unknown')}")
                    st.caption(f"Size: {base_image.get('size', 'unknown')}")
                    
                    # Download button for base image
                    if base_image.get('s3_url'):
                        st.markdown(f"[📥 Download Base Image]({base_image['s3_url']})")
                
                with col_vto:
                    st.markdown(f"**🎭 VTO-Enhanced ({vto_image.get('vto_category', 'Unknown')})**")
                    if Path(vto_image.get('local_path', '')).exists():
                        st.image(vto_image['local_path'], use_container_width=True)
                    elif vto_image.get('s3_url'):
                        st.image(vto_image['s3_url'], use_container_width=True)
                    
                    # Show VTO details
                    st.caption(f"Method: {vto_image.get('method', 'unknown')}")
                    st.caption(f"Enhancement: {vto_image.get('enhancement', 'unknown')}")
                    
                    # Download button for VTO image
                    if vto_image.get('s3_url'):
                        st.markdown(f"[📥 Download VTO Image]({vto_image['s3_url']})")
                
                st.markdown("---")
    
    else:
        # Regular gallery view
        st.markdown("### 🖼️ All Generated Images")
        
        # Collect all images with metadata
        all_images = []
        seen_images = set()  # Track unique images to avoid duplicates
        
        for result in results:
            product = result.get('product', {})
            assets = result.get('assets', {})
            
            # Apply category filter
            if selected_category != "All" and product.get('category') != selected_category:
                continue
            
            # Apply product ID filter
            if selected_product_id != "All" and product.get('product_id') != selected_product_id:
                continue
            
            for img in assets.get('images', []):
                # Apply method filter
                if selected_method != "All" and img.get('method') != selected_method:
                    continue
                
                # Check if we have either local file or S3 URL
                local_path = img.get('local_path')
                s3_url = img.get('s3_url')
                has_local_file = local_path and Path(local_path).exists()
                
                # Include image if we have either a local file or S3 URL
                if has_local_file or s3_url:
                    # Create unique identifier for this image
                    unique_id = f"{product.get('product_id', '')}_{img.get('scene_name', '')}_{img.get('size', '')}_{img.get('method', '')}"
                    
                    # Skip if we've already seen this image
                    if unique_id in seen_images:
                        continue
                    seen_images.add(unique_id)
                    
                    # Generate presigned URL if S3 URL exists
                    presigned_url = None
                    if s3_url and st.session_state.dpa_automator:
                        presigned_url = st.session_state.dpa_automator.generate_presigned_url(s3_url, expiration=86400)  # 24 hours
                    
                    # Try to find local file by pattern matching if no local_path
                    if not has_local_file and s3_url:
                        product_id = product.get('product_id', '')
                        scene_name = img.get('scene_name', '')
                        size = img.get('size', '')
                        
                        if product_id and scene_name:
                            # Try common filename patterns
                            possible_patterns = [
                                f"dpa_outpaint_{product_id}_{scene_name}_{size}_*.png",
                                f"dpa_outpaint_{product_id}_*_{scene_name}_{size}_*.png",
                                f"dpa_outpaint_{product_id}_*{scene_name}*_{size}_*.png",
                                f"dpa_outpaint_{product_id}_*{scene_name[:30]}*_{size}_*.png",  # Truncated scene name
                                f"dpa_outpaint_{product_id}_*{scene_name[:20]}*_{size}_*.png",  # More truncated
                                f"dpa_outpaint_{product_id}_*_{size}_*.png"  # Just product and size as fallback
                            ]
                            
                            results_dir = Path("generated_ads")
                            for pattern in possible_patterns:
                                matching_files = list(results_dir.glob(pattern))
                                if matching_files:
                                    # Use the first match
                                    local_path = str(matching_files[0])
                                    has_local_file = True
                                    break
                    
                    all_images.append({
                        'path': local_path,
                        'product_name': product.get('name', 'Unknown'),
                        'product_id': product.get('product_id', 'Unknown'),
                        'size': img.get('size', 'Unknown'),
                        'method': img.get('method', 'unknown'),
                        'prompt': img.get('prompt', 'No prompt available'),
                        'negative_prompt': img.get('negative_prompt', ''),
                        'image_type': img.get('image_type', 'unknown'),
                        'enhancement': img.get('enhancement', ''),
                        'vto_category': img.get('vto_category', ''),
                        's3_url': s3_url,
                        'presigned_url': presigned_url,
                        'price': product.get('price', 0),
                        'category': product.get('category', 'Unknown'),
                        'scene_name': img.get('scene_name', ''),
                        'has_local_file': has_local_file,
                        'ai_scoring': img.get('ai_scoring')  # Include AI scoring data
                    })
    
        if not all_images:
            st.info("No images found matching the selected filters.")
        else:
            st.info(f"Showing {len(all_images)} generated images")
            
            # Debug: Show processed all_images data
            with st.expander("🔍 Debug: Processed Images Data"):
                st.write(f"Total unique images found: {len(all_images)}")
                st.write(f"Seen images count: {len(seen_images)}")
                
                # Show first few images data for debugging
                for i, img_data in enumerate(all_images[:10]):  # Show first 10
                    st.write(f"**Image {i+1}:**")
                    st.json({
                        'product_id': img_data.get('product_id'),
                        'scene_name': img_data.get('scene_name'),
                        'size': img_data.get('size'),
                        'method': img_data.get('method'),
                        'path': img_data.get('path'),
                        's3_url': img_data.get('s3_url'),
                        'has_local_file': img_data.get('has_local_file')
                    })
                
                if len(all_images) > 10:
                    st.write(f"... and {len(all_images) - 10} more images")
            
            # Display images in grid
            for idx in range(0, len(all_images), images_per_row):
                cols = st.columns(images_per_row)
                print(f"i:{idx}")
                for j, col in enumerate(cols):
                    k = j + idx
                    if k < len(all_images):
                        img_data = all_images[k]

                        with col:
                            try:
                                image_displayed = False
                                
                                # Try to display image from multiple sources
                                # 1. Try presigned URL first (best for S3 images)
                                if img_data.get('presigned_url'):
                                    st.image(img_data['presigned_url'], use_container_width=True)
                                    image_displayed = True
                                
                                # 2. Try S3 URL if presigned URL failed
                                elif img_data.get('s3_url'):
                                    st.image(img_data['s3_url'], use_container_width=True)
                                    image_displayed = True
                                
                                # 3. Try local path as fallback
                                elif img_data.get('path') and Path(img_data['path']).exists():
                                    st.image(img_data['path'], use_container_width=True)
                                    image_displayed = True
                                
                                # 4. If no image could be displayed, show placeholder
                                if not image_displayed:
                                    st.error("🖼️ Image not available")
                                    st.caption("No valid image source found")
                                
                                # Image metadata (always show, even if image failed)
                                st.markdown(f"""
                                **{img_data['product_name']}**  
                                ID: `{img_data['product_id']}`  
                                Size: {img_data['size']}  
                                Method: {img_data['method']}  
                                Price: ${img_data['price']:.2f}  
                                Category: {img_data['category']}
                                """)
                        
                                # Show generation details
                                if img_data['image_type']:
                                    st.caption(f"Type: {img_data['image_type']}")
                                if img_data['enhancement']:
                                    st.caption(f"Enhancement: {img_data['enhancement']}")
                                if img_data['vto_category']:
                                    st.caption(f"VTO Category: {img_data['vto_category']}")
                                
                                # Show AI scoring if available
                                if img_data.get('ai_scoring'):
                                    scoring_results = img_data['ai_scoring']
                                    
                                    # Check if scoring was successful
                                    claude_success = scoring_results.get('claude_results', {}).get('success', False)
                                    nova_success = scoring_results.get('nova_results', {}).get('success', False)
                                    
                                    if claude_success or nova_success:
                                        # Calculate total scores
                                        def calculate_total_score(scores_dict):
                                            categories = ['visual_quality', 'product_presentation', 'advertising_effectiveness', 
                                                        'technical_execution', 'image_harmony', 'image_rationality',
                                                        'overall_commercial_appeal']
                                            total = 0
                                            for category in categories:
                                                if category in scores_dict:
                                                    score = scores_dict[category].get('score', 0)
                                                    total += score
                                            return total
                                        
                                        claude_total = 0
                                        nova_total = 0
                                        
                                        if claude_success:
                                            claude_scores = scoring_results['claude_results']['scores']
                                            claude_total = calculate_total_score(claude_scores)
                                        
                                        if nova_success:
                                            nova_scores = scoring_results['nova_results']['scores']
                                            nova_total = calculate_total_score(nova_scores)
                                        
                                        # Show summary scores as caption
                                        if claude_success and nova_success:
                                            avg_score = (claude_total + nova_total) / 2
                                            st.caption(f"🤖 AI Score: {avg_score:.1f}/70 (Claude: {claude_total}, Nova: {nova_total})")
                                        elif claude_success:
                                            st.caption(f"🤖 AI Score: {claude_total}/70 (Claude Sonnet 4)")
                                        elif nova_success:
                                            st.caption(f"🤖 AI Score: {nova_total}/70 (Amazon Nova Pro)")
                                    else:
                                        st.caption("🤖 AI scoring failed")
                                else:
                                    st.caption("🤖 AI scoring not available")
                                
                                # Show prompts in expander
                                with st.expander("📝 View Prompts", expanded=False):
                                    st.markdown("**🎨 Image Prompt:**")
                                    st.code(img_data['prompt'], language=None)
                                    
                                    if img_data['negative_prompt']:
                                        st.markdown("**🚫 Negative Prompt:**")
                                        st.code(img_data['negative_prompt'], language=None)
                                
                                # Show AI scoring details in expander
                                if img_data.get('ai_scoring'):
                                    scoring_results = img_data['ai_scoring']
                                    claude_success = scoring_results.get('claude_results', {}).get('success', False)
                                    nova_success = scoring_results.get('nova_results', {}).get('success', False)
                                    
                                    if claude_success or nova_success:
                                        with st.expander("🤖 View AI Scores", expanded=False):
                                            if claude_success:
                                                st.markdown("**🧠 Claude Sonnet 4 Analysis:**")
                                                claude_scores = scoring_results['claude_results']['scores']
                                                
                                                # Show scores in a more compact format
                                                col1, col2 = st.columns(2)
                                                
                                                for i, (category, data) in enumerate(claude_scores.items()):
                                                    if isinstance(data, dict) and 'score' in data:
                                                        col = col1 if i % 2 == 0 else col2
                                                        with col:
                                                            st.metric(
                                                                category.replace('_', ' ').title(),
                                                                f"{data['score']}/10",
                                                                help=data.get('feedback', '')
                                                            )
                                                
                                                claude_total = sum(data.get('score', 0) for data in claude_scores.values() if isinstance(data, dict))
                                                st.markdown(f"**Total: {claude_total}/70**")
                                                
                                                # Show summary if available
                                                if 'summary' in scoring_results['claude_results']:
                                                    st.markdown("**Summary:**")
                                                    st.write(scoring_results['claude_results']['summary'])
                                            
                                            if nova_success:
                                                if claude_success:
                                                    st.markdown("---")
                                                
                                                st.markdown("**🚀 Amazon Nova Pro Analysis:**")
                                                nova_scores = scoring_results['nova_results']['scores']
                                                
                                                # Show scores in a more compact format
                                                col1, col2 = st.columns(2)
                                                
                                                for i, (category, data) in enumerate(nova_scores.items()):
                                                    if isinstance(data, dict) and 'score' in data:
                                                        col = col1 if i % 2 == 0 else col2
                                                        with col:
                                                            st.metric(
                                                                category.replace('_', ' ').title(),
                                                                f"{data['score']}/10",
                                                                help=data.get('feedback', '')
                                                            )
                                                
                                                nova_total = sum(data.get('score', 0) for data in nova_scores.values() if isinstance(data, dict))
                                                st.markdown(f"**Total: {nova_total}/70**")
                                                
                                                # Show summary if available
                                                if 'summary' in scoring_results['nova_results']:
                                                    st.markdown("**Summary:**")
                                                    st.write(scoring_results['nova_results']['summary'])
                                
                                # Presigned URL if available
                                if img_data['presigned_url']:
                                    st.markdown(f"🔗 [Download Image (24h link)]({img_data['presigned_url']})")
                                elif img_data['s3_url'] and img_data['s3_url'] != 'N/A':
                                    st.caption("☁️ Stored in S3 (presigned URL generation failed)")
                                else:
                                    st.caption("📁 Local file only")
                                
                                # File info
                                if img_data.get('path'):
                                    file_size = Path(img_data['path']).stat().st_size
                                    st.caption(f"📁 {file_size:,} bytes")
                                
                            except Exception as e:
                                st.error(f"Error loading image: {e}")
            
            # Detailed results table
            st.subheader("📊 Detailed Results")
    
        # Add prompts analysis section
        st.subheader("📝 Generation Prompts Analysis")
    
        # Collect all prompts for analysis
        all_prompts = []
        for result in results:
            assets = result.get('assets', {})
            product = result.get('product', {})
        
            for img in assets.get('images', []):
                prompt_data = {
                    'product_name': product.get('name', 'Unknown'),
                    'product_category': product.get('category', 'Unknown'),
                    'method': img.get('method', 'unknown'),
                    'image_type': img.get('image_type', 'unknown'),
                    'prompt': img.get('prompt', 'No prompt available'),
                    'negative_prompt': img.get('negative_prompt', ''),
                    'enhancement': img.get('enhancement', ''),
                    'vto_category': img.get('vto_category', '')
                }
                all_prompts.append(prompt_data)
    
        if all_prompts:
            # Prompt statistics
            col1, col2, col3 = st.columns(3)
        
            with col1:
                total_prompts = len(all_prompts)
                st.metric("Total Prompts", total_prompts)
        
            with col2:
                ai_enhanced = len([p for p in all_prompts if 'vto' in p['method']])
                st.metric("AI Enhanced", ai_enhanced)
        
            with col3:
                avg_prompt_length = sum(len(p['prompt']) for p in all_prompts) / len(all_prompts)
                st.metric("Avg Prompt Length", f"{avg_prompt_length:.0f} chars")
        
            # Prompts by category
            st.markdown("#### 📊 Prompts by Product Category")
        
            category_prompts = {}
            for prompt_data in all_prompts:
                category = prompt_data['product_category']
                if category not in category_prompts:
                    category_prompts[category] = []
                category_prompts[category].append(prompt_data)
        
            for category, prompts in category_prompts.items():
                with st.expander(f"📁 {category.title()} ({len(prompts)} prompts)", expanded=False):
                    for i, prompt_data in enumerate(prompts[:3]):  # Show first 3 examples
                        st.markdown(f"**Example {i+1}: {prompt_data['product_name']}**")
                        st.markdown(f"*Method: {prompt_data['method']} | Type: {prompt_data['image_type']}*")
                    
                        col_pos, col_neg = st.columns(2)
                        with col_pos:
                            st.markdown("**🎨 Positive Prompt:**")
                            st.code(prompt_data['prompt'], language=None)
                    
                        with col_neg:
                            if prompt_data['negative_prompt']:
                                st.markdown("**🚫 Negative Prompt:**")
                                st.code(prompt_data['negative_prompt'], language=None)
                            else:
                                st.markdown("**🚫 Negative Prompt:**")
                                st.caption("No negative prompt used")
                    
                        if prompt_data['enhancement']:
                            st.caption(f"Enhancement: {prompt_data['enhancement']}")
                        if prompt_data['vto_category']:
                            st.caption(f"VTO Category: {prompt_data['vto_category']}")
                    
                        st.markdown("---")
                
                    if len(prompts) > 3:
                        st.caption(f"... and {len(prompts) - 3} more prompts in this category")
    
        # Create detailed dataframe with presigned URLs
        detailed_data = []
        for result in results:
            product = result.get('product', {})
            creative = result.get('creative', {})
            assets = result.get('assets', {})
        
            # Generate presigned URLs for this product's images
            presigned_urls = []
            for img in assets.get('images', []):
                s3_url = img.get('s3_url')
                if s3_url and st.session_state.dpa_automator:
                    presigned_url = st.session_state.dpa_automator.generate_presigned_url(s3_url, expiration=86400)
                    if presigned_url:
                        presigned_urls.append(f"[{img.get('size', 'Image')}]({presigned_url})")
        
            # Join presigned URLs or show alternative
            presigned_links = " | ".join(presigned_urls) if presigned_urls else "Local files only"
        
            detailed_data.append({
                'Product ID': product.get('product_id', 'N/A'),
                'Name': product.get('name', 'N/A'),
                'Category': product.get('category', 'N/A'),
                'Price': f"${product.get('price', 0):.2f}",
                'Headline': creative.get('headline', 'N/A'),
                'Images': len(assets.get('images', [])),
                'Videos': len(assets.get('videos', [])),
                'Generation Method': assets.get('metadata', {}).get('generation_method', 'N/A'),
                'Download Links (24h)': presigned_links,
                'Status': '✅ Success' if assets.get('images') or assets.get('videos') else '❌ Failed'
            })
    
        if detailed_data:
            df = pd.DataFrame(detailed_data)
            st.dataframe(df, use_container_width=True)
        
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"dpa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
            # Next step instructions
            st.markdown("---")
            st.info("""
            ### 🔄 **What's Next?**
        
            **Continue Processing:** Go to **📁 Catalog Upload** in the sidebar to upload a new product catalog and generate more images.
        
            **Create Templates:** Go to **🎨 Template Designer** in the sidebar to create custom templates for different product categories.
        
            **Process More:** Go to **🚀 Batch Processing** in the sidebar to process additional products with your existing templates.
        
            💡 **Tip:** You can download the CSV results for external analysis or reporting.
            """)

def main():
    """Main application"""
    initialize_session_state()
    
    # Render sidebar and get current page
    current_page = render_sidebar()
    
    # Render appropriate page
    if current_page == "📁 Catalog Upload":
        render_catalog_upload()
    elif current_page == "🎨 Template Designer":
        render_template_designer()
    elif current_page == "🚀 Batch Processing":
        render_batch_processing()
    elif current_page == "📈 Analytics":
        render_analytics()
    else:
        # Default to catalog upload
        render_catalog_upload()

if __name__ == "__main__":
    main()
