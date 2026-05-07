# Báo Cáo Thực Hành MLOps: Từ Thực Nghiệm Cục Bộ Đến Triển Khai Liên Tục

**Họ và tên:** Hồ Sỹ Minh Hà
**MSSV**: 2A20260006
**Cohort:** A20-Track2

---

## 1. Thông Tin Repo GitHub
- **URL Repo:** [https://github.com/cafesuada24/2A202600060-HoSyMinhHa-Track2-Day21]
---

## 3. Báo Cáo Phân Tích

### 3.1. Lựa Chọn Siêu Tham Số (Kết quả Bước 1)
Sau khi thực hiện thí nghiệm trên MLflow, tôi đã chọn bộ siêu tham số sau cho mô hình cuối cùng:
- `n_estimators`: 500
- `max_depth`: 15
- `min_samples_split`: 5

**Lý do chọn:**
Qua quá trình so sánh trên MLflow UI, bộ tham số này mang lại độ chính xác (`accuracy`) và chỉ số `f1_score` cao nhất (đạt trên ngưỡng 0.70 yêu cầu). Việc tăng `n_estimators` lên 500 giúp mô hình ổn định hơn, và `max_depth: 15` cho phép mô hình học được các đặc trưng phức tạp của dữ liệu Wine Quality mà không bị quá khớp (overfitting) quá mức so với các giá trị thấp hơn. Hơn nữa, độ chính xác bằng với n_estimators cao hơn, cho thấy đây là giới hạn trên của model.

### 3.2. Khó Khăn Và Giải Pháp
1. **Lỗi xác thực DVC trên GitHub Actions:**
   - *Vấn đề:* Job `train` không thể `dvc pull` do thiếu quyền truy cập Cloud Storage.
   - *Giải pháp:* Đã thêm secret `CLOUD_CREDENTIALS` vào GitHub và cấu hình file key tạm thời trong workflow `.yml` để DVC có thể nhận diện.
2. **Thứ tự Push dữ liệu và code ở Bước 3:**
   - *Vấn đề:* Nếu push code trước khi `dvc push` dữ liệu mới, pipeline sẽ bị lỗi vì không tìm thấy hash dữ liệu trên remote.
   - *Giải pháp:* Luôn tuân thủ quy trình: `dvc push` (đẩy dữ liệu lên cloud) trước khi `git push` (kích hoạt pipeline).
3. **Cấu hình Firewall cho VM:**
   - *Vấn đề:* Không thể truy cập endpoint `:8000` từ bên ngoài.
   - *Giải pháp:* Kiểm tra lại và thêm firewall rule trên GCP Console để cho phép traffic TCP qua cổng 8000 với tag tương ứng của VM.

---

*Hết báo cáo.*
