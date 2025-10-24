# Ứng dụng Phân loại 102 Loài hoa (Python/Flask)

Đây là một ứng dụng web demo sử dụng mô hình học sâu **Vision Transformer (ViT-B/16)** để phân loại 102 loài hoa từ bộ dữ liệu Oxford 102 Flowers. Mô hình đã được huấn luyện và tối ưu hóa để đạt độ chính xác **~91.9%** trên tập kiểm tra.

Ứng dụng này được xây dựng hoàn toàn bằng **Python** sử dụng framework **Flask**, giúp đảm bảo tốc độ dự đoán nhanh bằng cách tải mô hình vào bộ nhớ một lần duy nhất.

## ## Tính năng Chính

* **Giao diện Đơn giản:** Giao diện web thân thiện cho phép người dùng dễ dàng tải ảnh lên.
* **Mô hình Hiệu suất cao (SOTA):** Sử dụng kiến trúc `ViT-B/16` đã được tinh chỉnh sâu (fine-tuned) để cho kết quả chính xác vượt trội.
* **Tốc độ Dự đoán Nhanh:** Mô hình được tải một lần duy nhất khi máy chủ khởi động, cho phép phản hồi dự đoán gần như tức thì (loại bỏ "khởi động lạnh").
* **Kết quả Chi tiết:** Trả về Top 3 loài hoa có khả năng cao nhất cùng với phần trăm xác suất.

---

## ## Yêu cầu Hệ thống (Prerequisites)

Trước khi bắt đầu, hãy đảm bảo máy tính của bạn đã cài đặt:

* **Python** (phiên bản 3.9 trở lên) và **pip**.
    * **Lưu ý quan trọng:** Trong quá trình cài đặt Python trên Windows, hãy đảm bảo bạn đã **đánh dấu vào ô "Add Python to PATH"**.

---

## ## Hướng dẫn Cài đặt và Chạy

### **Bước 1: Tải Mã nguồn và File Model**

1.  **Clone repository này về máy:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
    *(Thay `your-username/your-repository-name` bằng thông tin GitHub của bạn)*

2.  **Tải File Model:** Do file model ViT rất lớn (~330MB), bạn cần tải nó về thủ công.
    * Tải file **`exp_vit.pth`** từ link sau: **(https://drive.google.com/file/d/1ACAMxI0iTu3Y8NRFKWo64lczgRPbvxif/view?usp=drive_link)**
    * Sau khi tải về, đặt file này vào đúng thư mục: `models/`.

### **Bước 2: Cài đặt Môi trường**

1.  **(Khuyến khích) Tạo và Kích hoạt Môi trường ảo:**
    Mở terminal tại thư mục gốc của dự án và chạy:
    ```bash
    # Tạo môi trường ảo
    python -m venv venv
    
    # Kích hoạt môi trường ảo
    # Trên Windows:
    .\venv\Scripts\activate
    # Trên MacOS/Linux:
    source venv/bin/activate
    ```

2.  **Cài đặt các Thư viện:**
    Vẫn trong terminal đó (với môi trường ảo đã được kích hoạt), chạy lệnh sau:
    ```bash
    pip install -r requirements.txt
    ```
    *(Quá trình này có thể mất vài phút vì cần tải về PyTorch.)*

### **Bước 3: Khởi động Ứng dụng**

1.  **Chạy server:**
    ```bash
    python app.py
    ```

2.  **Xem kết quả:**
    Đợi terminal hiển thị thông báo "Đang tải mô hình..." và "Mô hình đã sẵn sàng!", sau đó:
    * Mở trình duyệt web của bạn và truy cập vào địa chỉ: `http://localhost:5000`

Bây giờ bạn đã có thể tải ảnh lên và trải nghiệm mô hình phân loại hoa mạnh nhất của mình!

---

## ## Cấu trúc Dự án

```
flower_app_flask/
├── models/
│   ├── exp_vit.pth              # File trọng số mô hình ViT (phải tải)
│   ├── cat_to_name.json         # Ánh xạ nhãn → tên loài hoa
│   ├── flower_color_map.json    # Dữ liệu màu chủ đạo cho từng loài hoa
│   └── wiki_cache.json          # Cache mô tả Wikipedia (tùy chọn)
│
├── static/
│   ├── uploads/                 # Ảnh người dùng upload
│   └── style.css                # CSS giao diện
│
├── templates/
│   └── index.html               # Giao diện web chính
│
├── app.py                       # Ứng dụng Flask (entry point)
├── model_service.py             # Xử lý dự đoán, màu sắc và wiki
├── requirements.txt             # Thư viện Python cần thiết
└── README.md                    # Tài liệu hướng dẫn (bạn đang đọc)

```