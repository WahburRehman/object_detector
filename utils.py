import io
import re
import cv2
import random
import pillow_heif
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer, util


# YOLO model load
det = YOLO('yolov8s.pt')

# Class name mappings
ID_2_NAME = det.names
ALL_CLASS_NAMES = [ID_2_NAME[i] for i in range(len(ID_2_NAME))]

# Embedding model (for semantic matching)
nlp = SentenceTransformer('all-MiniLM-L6-v2')

# Seed for consistent colors



# COCO class embeddings
# Precompute embeddings for all COCO classes.
# To allow semantic matching (e.g., "sedan" → "car", "puppy" → "dog").
# Each class name is converted into a vector, so we can find closest matches using cosine similarity.
def get_class_embeddings(all_class_names):
    random.seed(42)
    class_embs = nlp.encode(all_class_names, convert_to_tensor=True)
    # also build a color map
    color_map = {n: tuple(int(x) for x in np.random.randint(0, 255, 3))
                 for n in all_class_names}
    return class_embs, color_map

# --- Initialization (run once at import time) ---
CLASS_EMBS, COLOR_MAP = get_class_embeddings(ALL_CLASS_NAMES)



# Very small plural → singular helper (persons→person, cars→car)
def singularize(word: str):
    if word.endswith("s") and word[:-1] in ALL_CLASS_NAMES:
        return word[:-1]
    return word


# expand_group:
# - Maps higher-level group names (e.g., "vehicles", "animals", "round") to
#   a predefined set of COCO classes.
# - Useful when user writes a category instead of a specific object.
# Example: "vehicles" → ["car", "bus", "truck", "bicycle"]
def expand_group(token):

    ROUND_OBJECTS = ["sports ball", "apple", "orange",
                     "donut", "frisbee", "clock", "pizza", "stop sign"]
    RECTANGULAR_OBJECTS = ["book", "laptop", "tv", "cell phone",
                           "microwave", "refrigerator", "toaster", "keyboard", "remote"]

    GROUPS = {
        # ------ Transportation ------
        "vehicles": ["car", "bus", "truck", "train", "motorcycle", "bicycle", "airplane", "boat"],
        "road vehicles": ["car", "bus", "truck", "motorcycle", "bicycle"],
        "air vehicles": ["airplane"],
        "water vehicles": ["boat"],

        # ------ Animals ------
        "animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
        "pets": ["dog", "cat", "bird"],
        "wild animals": ["elephant", "bear", "zebra", "giraffe"],

        # ------ Food/Kitchen ------
        "food": ["apple", "banana", "orange", "sandwich", "pizza", "donut", "cake"],
        "kitchenware": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],

        # ------ Furniture ------
        "furniture": ["chair", "couch", "bed", "dining table", "toilet"],

        # ------ Electronics ------
        "electronics": ["tv", "laptop", "cell phone", "remote", "keyboard", "microwave", "oven", "toaster"],

        # ------ Outdoor Objects ------
        "outdoor": ["traffic light", "fire hydrant", "stop sign", "bench", "umbrella"],

        # ------ Sports ------
        "sports": ["sports ball", "frisbee", "skateboard", "surfboard", "tennis racket"],

        # ------ People ------
        "people": ["person"],
        "kids": ["person"],  # Special case (filter by size later)

        # ====== SHAPE-BASED GROUPS ======
        # Round/Circular/Spherical objects
        "round": ROUND_OBJECTS,
        "circular": ROUND_OBJECTS,
        "round objects": ROUND_OBJECTS,
        "sphere": ["sports ball", "apple", "orange"],


        # Rectangular/Square objects
        "rectangular": RECTANGULAR_OBJECTS,
        "rectangle": RECTANGULAR_OBJECTS,
        "rectangular objects": RECTANGULAR_OBJECTS,
        # Some clocks are square
        "square": ["picture frame", "pillow", "clock"],


        # Flat/Thin objects
        "flat": ["picture frame", "credit card", "paper", "book"],
        "thin": ["knife", "spoon", "fork", "credit card"],

        # Cylindrical objects
        "cylinder": ["bottle", "cup", "wine glass", "toilet paper"],
    }

    matches = GROUPS.get(token, [])
    return list(set(matches))  # Remove duplicates


# resolve_alias:
# - Converts a user token into its embedding.
# - Compares with embeddings of all COCO classes using cosine similarity.
# - If similarity >= threshold, maps token to closest COCO class(es).
# Example: "sedan" → "car"
def resolve_alias(token, threshold=0.5):

    token_emb = nlp.encode(token, convert_to_tensor=True)
    sims = util.cos_sim(token_emb, CLASS_EMBS).flatten()

    # Get indices of all matches above threshold
    matches = [i for i, sim in enumerate(sims) if sim >= threshold]

    # Return all matching class names
    return [ALL_CLASS_NAMES[i] for i in matches]


# classes_from_prompt:
# - Main pipeline that processes the user prompt.
# - Steps:
#   1. Tokenize prompt (split on commas, spaces, "and").
#   2. Remove stopwords (e.g., "find", "all").
#   3. If token matches COCO class → keep it.
#   4. Else, check group expansion (e.g., "vehicles" → ["car","bus",...]).
#   5. Else, try alias resolution (e.g., "sedan" → "car").
# - Returns: sorted list of matching COCO class names.
def classes_from_prompt(prompt):

    STOPWORDS = {
        "find", "detect", "locate", "identify", "show", "highlight", "mark", "label",
        "all", "every", "any", "some", "the", "a", "an", "in", "on", "at", "and", "or", "but",
        "please", "object", "objects", "item", "items", "things", "stuff"
    }

    # Step 1: Tokenize (split on commas/&/and)
    tokens = [t.strip() for t in re.split(
        r'[,&\s]| and ', prompt.lower()) if t.strip()]

    keep = set()
    for token in tokens:
        # Skip stopwords
        if token in STOPWORDS:
            continue

        token = singularize(token)

        # Case 1: Exact COCO class match
        if token in ALL_CLASS_NAMES:
            keep.add(token)
            continue

        # Case 2: Group expansion
        # This Funtion helps to return the all COCO classes if token is a collective category e.g. vehicles -> car, bus, truck, bus etc.
        group_classes = expand_group(token)
        if group_classes:
            keep.update(group_classes)
            continue

        # Case 3: Alias resolution (embeddings)
        # This Funtion helps to return the all COCO classes if token is an alias e.g. sedan → car, puppy → dog
        alias_match = resolve_alias(token)
        if alias_match:
            keep.update(alias_match)
            continue

        # (Optional) Case 4: LLM fallback for complex terms
        # llm_match = query_llm(f"Map '{token}' to a COCO class")
        # if llm_match in ALL_CLASS_NAMES:
        #     keep.add(llm_match)

    return sorted(keep)


# Converts ANY input image into a YOLO-compatible RGB JPEG.
# Supports JPG, JPEG, PNG, HEIC (if pillow-heif installed), WEBP, etc.
def make_image_yolo_friendly(file_bytes, filename="input.jpg"):

    try:
        # Convert bytes to a file-like object
        img = Image.open(io.BytesIO(file_bytes))
    except Exception:
        # If it's HEIC, need pillow-heif
        try:
            pillow_heif.register_heif_opener()
            img = Image.open(io.BytesIO(file_bytes))
        except Exception as e:
            raise RuntimeError(f"Could not read image: {e}")

    # Fix orientation, convert to RGB
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)  # RGB (H,W,3)




# Draw only those detections whose class name are in target_labels.
def draw_bounding_box(res, user_prompt: str, target_labels: set,
                  show_labels: bool = False, kid_threshold: float = 0.9):

    # res.orig_img is RGB -> convert to BGR for cv2 drawing because OpenCV uses BGR instead of RGB
    img_bgr = cv2.cvtColor(res.orig_img.copy(), cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]
    TH = max(2, int(round(min(H, W) * 0.005)))  # ~0.5% of shorter side

    if res.boxes is None or len(res.boxes) == 0:
        return img_bgr

    # zip() is a Python built-in Function to combine multiple lists/arrays element-wise.
    # Here, for each detection it gives (box, class_id, confidence) together,
    # so the loop iterates once per detection with all three values aligned.
    for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        coco_class_name = ID_2_NAME[int(cls)] if isinstance(ID_2_NAME, dict) else ID_2_NAME[int(cls)]

        # class filter
        if coco_class_name not in target_labels and len(target_labels) > 0:
            continue

        # kids heuristic
        if ("kid" in user_prompt.lower() or "child" in user_prompt.lower()) and coco_class_name == "person":
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            if (y2 - y1) / H >= kid_threshold:
                continue

        # draw box
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        color = COLOR_MAP.get(coco_class_name, (0, 255, 0))
        
        # 1) black outline (contrast), 2) colored box (AA = smoother)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 0), TH + 2, lineType=cv2.LINE_AA)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, TH, lineType=cv2.LINE_AA)

        if show_labels:
            fs = max(0.6, TH / 6.0)
            cv2.putText(img_bgr, coco_class_name, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                max(1, TH // 2), lineType=cv2.LINE_AA)

    return img_bgr
