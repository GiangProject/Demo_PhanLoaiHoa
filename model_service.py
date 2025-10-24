# ======================================================
# model_service.py — BẢN TỐI ƯU CÓ CACHE WIKIPEDIA
# ======================================================

import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import json
import torch.nn.functional as F
import os
import sys
import wikipediaapi
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

# --- CẤU HÌNH ---
MODEL_PATH = 'models/exp_vit.pth'
CLASS_MAPPING_PATH = 'models/cat_to_name.json'
COLOR_MAP_PATH = 'models/flower_color_map.json'
WIKI_CACHE_PATH = 'models/wiki_cache.json'  # 🔥 file cache wiki
INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Khởi tạo Wikipedia ---
WIKI_VI = wikipediaapi.Wikipedia('FlowerApp (example@example.com)', 'vi')
WIKI_EN = wikipediaapi.Wikipedia('FlowerApp (example@example.com)', 'en')

# ======================================================
# 1️⃣ HỖ TRỢ: HÀM TÓM TẮT & CACHE WIKIPEDIA
# ======================================================

def smart_summarize(text, sentence_count=2):
    """Tóm tắt nội dung bằng Sumy."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("vietnamese"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, sentence_count)
        return " ".join(str(s) for s in summary_sentences)
    except Exception:
        return text[:500] + "..."

# --- Tải cache Wikipedia (nếu có) ---
if os.path.exists(WIKI_CACHE_PATH):
    with open(WIKI_CACHE_PATH, 'r', encoding='utf-8') as f:
        WIKI_CACHE = json.load(f)
else:
    WIKI_CACHE = {}

def get_wiki_summary_cached(flower_name):
    """Tra Wikipedia có cache để tăng tốc."""
    if flower_name in WIKI_CACHE:
        return WIKI_CACHE[flower_name]

    try:
        page_vi = WIKI_VI.page(flower_name)
        if page_vi.exists():
            summary = smart_summarize(page_vi.summary)
        else:
            page_en = WIKI_EN.page(flower_name)
            summary = smart_summarize(page_en.summary) if page_en.exists() else "Không có thông tin mô tả."
    except Exception as e:
        print(f"[⚠️] Lỗi Wikipedia: {e}", file=sys.stderr)
        summary = "Không thể tải thông tin Wikipedia."

    # Lưu vào cache để lần sau nhanh hơn
    WIKI_CACHE[flower_name] = summary
    with open(WIKI_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(WIKI_CACHE, f, ensure_ascii=False, indent=2)

    return summary

# ======================================================
# 2️⃣ XÂY DỰNG & TẢI MODEL
# ======================================================

def build_model_vit(num_classes=102):
    model = models.vit_b_16(weights=None)
    num_ftrs = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
    return model

def load_model_and_classes():
    print("🔧 Đang tải mô hình AI...")

    model = build_model_vit(num_classes=102).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with open(CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[str(i)] for i in range(1, 103)]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    return model, class_names, preprocess

# ======================================================
# 3️⃣ KHỞI ĐỘNG MODEL KHI SERVER BẬT
# ======================================================

try:
    MODEL, CLASS_NAMES, PREPROCESS = load_model_and_classes()
    print("✅ Mô hình đã sẵn sàng.")
except Exception as e:
    print(f"❌ Lỗi khởi động mô hình: {e}", file=sys.stderr)
    MODEL, CLASS_NAMES, PREPROCESS = None, [], None

# ======================================================
# 4️⃣ DỰ ĐOÁN
# ======================================================

def predict(image_path, top_k=3):
    """Dự đoán và trả kết quả nhanh (đã có cache Wikipedia)."""
    if MODEL is None:
        return [{
            "label": "Lỗi Model",
            "probability": "0.00%",
            "color": "Đen",
            "summary": "Không tải được mô hình."
        }]

    try:
        pil_img = Image.open(image_path).convert('RGB')
        input_tensor = PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = MODEL(input_tensor)
            probs = F.softmax(output, dim=1)

        top_probs, top_indices = torch.topk(probs, top_k)
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()

        with open(COLOR_MAP_PATH, 'r', encoding='utf-8') as f:
            color_map = json.load(f)

        results = []
        for i in range(len(top_indices)):
            label_name = CLASS_NAMES[top_indices[i]]
            probability = top_probs[i]
            color = color_map.get(label_name, "Không rõ")
            summary = get_wiki_summary_cached(label_name)

            results.append({
                "label": label_name,
                "probability": f"{probability * 100:.2f}%",
                "color": color,
                "summary": summary
            })

        return results

    except Exception as e:
        print(f"Lỗi dự đoán: {e}", file=sys.stderr)
        return [{
            "label": "Lỗi xử lý ảnh",
            "probability": "0.00%",
            "color": "Đen",
            "summary": "Không thể xử lý ảnh đầu vào."
        }]
