import streamlit as st, os
from PIL import Image
import base64
import time
from detector import YoloDetector
from utils import make_image_yolo_friendly, classes_from_prompt, draw_bounding_box

st.set_page_config(page_title="PromptVision", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
/* reduce main container padding */
div.block-container { padding-top: 1rem; }

/* fix image preview/result size */
img {
    max-height: 500px;   /* adjust as needed */
    object-fit: contain; /* keep ratio */
}
button[kind="primary"], .stButton > button { width: 100%; }

/* centered OR separator */
.or-sep{display:flex;align-items:center;gap:.5rem;margin:0.75rem 0 0.5rem;color:#888;font-weight:700;letter-spacing:.06em}
.or-sep:before,.or-sep:after{content:"";flex:1;border-bottom:1px solid #ddd}

/* Scanning overlay styles */
.pv-container {
    position: relative;
    display: inline-block;
    width: 100%;
}

.pv-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    display: none;
}

.pv-overlay.active {
    display: block;
}

.pv-overlay .bar {
    position:absolute; 
    left:5%; 
    width:90%; 
    height:16%;
    background:linear-gradient(to bottom,
        rgba(255,255,255,0) 0%,
        rgba(144, 143, 191,0.7) 50%,
        rgba(255,255,255,0) 100%);
    filter: blur(2px);
    animation: pv-scan 2.5s linear infinite; /* Slower animation */
    mix-blend-mode: screen; 
    pointer-events:none; 
    border-radius:12px;
    top: 0;
}

@keyframes pv-scan { 
    0% { top:0; transform: translateY(-100%); opacity: 0.5; } 
    50% { top:50%; transform: translateY(0); opacity: 1; }
    100% { top:100%; transform: translateY(100%); opacity: 0.5; } 
}

.pv-overlay .badge {
    position:absolute; 
    left:50%; 
    top:50%; 
    transform:translate(-50%,-50%);
    padding:6px 12px; 
    border-radius:999px;
    background:rgba(0,0,0,0.55); 
    color:#fff; 
    font-weight:700; 
    font-size:0.9rem;
    border:1px solid rgba(255,255,255,0.25); 
    backdrop-filter: blur(2px);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: translate(-50%, -50%) scale(1); }
    50% { transform: translate(-50%, -50%) scale(1.05); }
    100% { transform: translate(-50%, -50%) scale(1); }
}
</style>
""", unsafe_allow_html=True)

WEIGHTS = "yolov8m.pt"
IMAGE_SIZE = 640
CONFIDENCE = 0.25

# ---------------- Sidebar ----------------
st.sidebar.subheader("Choose a Sample Image")

sample_dir = "assets/sample_images"
sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg",".jpeg",".png",".webp"))]

# Initialize session state for image selection
if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'clear_uploader' not in st.session_state:
    st.session_state.clear_uploader = False

cols = st.sidebar.columns(2)
for i, file in enumerate(sample_files):
    img_path = os.path.join(sample_dir, file)
    thumb = Image.open(img_path).resize((150, 150))
    with cols[i % 2]:
        st.image(thumb, caption="", use_container_width=True)
        if st.button("Select", key=f"btn_{i}"):
            # Clear any uploaded file when selecting a sample
            st.session_state.uploaded_file = None
            st.session_state.selected_sample = file
            st.session_state.clear_uploader = True

# --- Display options ---
# show_labels = st.sidebar.toggle("Show detected object names", value=False)

st.sidebar.markdown('<div class="or-sep">OR</div>', unsafe_allow_html=True)

# uploader in sidebar
st.sidebar.markdown("**Upload an image**")

# Create a unique key for the file uploader
uploader_key = "file_uploader_" + str(hash(str(st.session_state.selected_sample)))

# Reset the uploader if needed
if st.session_state.clear_uploader:
    st.session_state.clear_uploader = False
    st.session_state.uploader_key = uploader_key

uploaded = st.sidebar.file_uploader(
    "Upload an image", 
    type=["jpg","jpeg","png","webp","heic"], 
    label_visibility="collapsed",
    key=uploader_key
)

# Update session state when a file is uploaded
if uploaded is not None:
    st.session_state.uploaded_file = uploaded
    # Clear sample selection when uploading a file
    st.session_state.selected_sample = None

# ---------------- Main ----------------
st.title("🎯 Object Detection with Prompt")

@st.cache_resource
def load_detector(w, c, s):
    return YoloDetector(weights=w, conf=c, imgsz=s)

# Initialize session state for overlay
if 'show_overlay' not in st.session_state:
    st.session_state.show_overlay = False
if 'processing_start_time' not in st.session_state:
    st.session_state.processing_start_time = 0

# Determine which image to use
file_bytes, filename = None, None

# Priority 1: Use uploaded file if available
if st.session_state.uploaded_file is not None:
    uploaded = st.session_state.uploaded_file
    file_bytes = uploaded.read()
    filename = uploaded.name
    # Reset file pointer for potential reuse
    uploaded.seek(0)

# Priority 2: Use selected sample if no uploaded file
elif st.session_state.selected_sample:
    sample_path = os.path.join(sample_dir, st.session_state.selected_sample)
    with open(sample_path, "rb") as f:
        file_bytes = f.read()
    filename = st.session_state.selected_sample

# ---- prompt input + Detect Button ----
with st.form("detect_form", clear_on_submit=False):
    c1, c2 = st.columns([5, 1])
    with c1:
        prompt = st.text_input(
            label=" ",
            placeholder="Find all vehicles and people",
            value="Find all objects",
            label_visibility="collapsed"
        )
    with c2:
        run = st.form_submit_button("🔎 Detect", use_container_width=True)

# image slot (preview or result)
img_slot = st.empty()

# Function to display image with overlay
def display_image_with_overlay(image_bytes, show_overlay=False):
    # Convert image bytes to base64 for HTML embedding
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    overlay_class = "active" if show_overlay else ""
    
    html = f"""
    <div class="pv-container">
        <img src="data:image/jpeg;base64,{encoded_image}" style="width:100%">
        <div class="pv-overlay {overlay_class}">
            <div class="bar"></div>
            <div class="badge">Scanning...</div>
        </div>
    </div>
    """
    img_slot.markdown(html, unsafe_allow_html=True)

# preview
if file_bytes and not run:
    img_slot.image(file_bytes, use_container_width=True)

# detection
if run:
    if not file_bytes:
        st.warning("Please choose a sample or upload an image first.")
    else:
        # Record start time and show overlay
        st.session_state.processing_start_time = time.time()
        display_image_with_overlay(file_bytes, show_overlay=True)
        
        # Process the image
        img_rgb = make_image_yolo_friendly(file_bytes)
        detector = load_detector(WEIGHTS, CONFIDENCE, IMAGE_SIZE)
        res = detector.predict(img_rgb)
        target_labels = set(classes_from_prompt(prompt))
        vis_bgr = draw_bounding_box(res, prompt, target_labels, show_labels=False)
        
        # Calculate elapsed time and ensure minimum 5 seconds
        elapsed_time = time.time() - st.session_state.processing_start_time
        if elapsed_time < 5:
            # Wait for the remaining time to reach 5 seconds
            time.sleep(5 - elapsed_time)
        
        # Display the result without overlay
        img_slot.image(vis_bgr[:, :, ::-1], use_container_width=True)