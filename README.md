# PromptScope — Prompt-Based Object Detection (YOLOv8 + NLP)

Type a natural prompt (e.g., **“Find all cars”**, **“highlight cats”**, **“person, bicycle”**) → upload or pick a sample image → get clean detections.  

> **Live demo:** _add Streamlit link after deploy_  

---

## ✨ Features
- **Prompt-based filtering** — works with **natural language** or **class names**:
  - “Find all cars and people”, “highlight cats”, or `person, car, laptop`
- **Fast UI** — Streamlit app with sample images (compressed for speed)
- **Clean visuals** — resolution-aware box thickness
- **“scanning” animation** while detection runs

---

## 🧠 Tech 
- **Ultralytics YOLOv8** (`yolov8m.pt`)
- **NLP**: Cosine similarity and Semantic Matching
- **Streamlit** (frontend)  
- **OpenCV** + **Pillow** (image processing)  

---

## 🚀 Quickstart (Local)
# 1) Install dependencies
pip install -r requirements.txt

# 2) Run the app
streamlit run app.py

