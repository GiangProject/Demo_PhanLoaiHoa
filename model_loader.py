# model_loader.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import json
import torch.nn.functional as F

# --- CÁC THAM SỐ CỐ ĐỊNH ---
MODEL_PATH = 'models/exp_vit.pth'
CLASS_MAPPING_PATH = 'cat_to_name.json'
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HÀM XÂY DỰNG MODEL ---
def build_model_vit(num_classes=102):
    """Xây dựng lại cấu trúc model ViT-B/16."""
    model = models.vit_b_16(weights=None)
    num_ftrs = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
    return model

# --- HÀM TẢI MODEL VÀ MAPPING ---
def load_model_and_classes():
    """Tải model và danh sách tên lớp một lần duy nhất."""
    # Tải model
    model = build_model_vit().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Chuyển model sang chế độ đánh giá
    
    # Tải mapping tên hoa
    with open(CLASS_MAPPING_PATH, 'r') as f:
        cat_to_name = json.load(f)
    # Tạo danh sách tên lớp theo đúng thứ tự (từ 1 đến 102)
    class_names = [cat_to_name[str(i)] for i in range(1, 103)]
    
    # Định nghĩa pipeline tiền xử lý
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    return model, class_names, preprocess

# --- TẢI MỌI THỨ VÀO BỘ NHỚ NGAY KHI APP KHỞI CHẠY ---
print("--- Đang tải mô hình AI vào bộ nhớ, vui lòng đợi... ---")
MODEL, CLASS_NAMES, PREPROCESS = load_model_and_classes()
print("✅ Mô hình đã sẵn sàng!")


# --- HÀM DỰ ĐOÁN CHÍNH ---
def predict(image_path, top_k=3):
    """
    Hàm nhận đường dẫn ảnh, tiền xử lý và trả về top K dự đoán.
    Sử dụng các biến MODEL, CLASS_NAMES, PREPROCESS đã được tải sẵn.
    """
    try:
        pil_img = Image.open(image_path).convert('RGB')
        input_tensor = PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = MODEL(input_tensor)
            probs = F.softmax(output, dim=1)
        
        top_probs, top_indices = torch.topk(probs, top_k)
        
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()
        
        results = []
        # Xử lý trường hợp top_k=1
        if top_k == 1:
            top_indices = [top_indices.item()]
            top_probs = [top_probs.item()]

        for i in range(len(top_indices)):
            label = CLASS_NAMES[top_indices[i]]
            prob = top_probs[i]
            results.append({"label": label, "probability": f"{prob*100:.2f}%"})
            
        return results
    except Exception as e:
        print(f"Lỗi khi dự đoán: {e}")
        return {"status": "error", "message": f"Lỗi xử lý ảnh: {e}"}