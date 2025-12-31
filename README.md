# Dự đoán nhiệt độ theo ngày

## 1. Giới thiệu đề tài

Bài toán dự đoán nhiệt độ theo ngày đóng vai trò quan trọng trong việc hỗ trợ các lĩnh vực như nông nghiệp, năng lượng, du lịch và đời sống hàng ngày. Mục tiêu của bài toán là áp dụng các kỹ thuật học máy và phân tích chuỗi thời gian để dự báo chính xác nhiệt độ trong tương lai dựa trên dữ liệu khí tượng lịch sử thu thập được. Từ đó, các đơn vị và cá nhân có thể chủ động lên kế hoạch sản xuất, tối ưu hóa việc tiêu thụ năng lượng và chuẩn bị ứng phó với các biến đổi thời tiết.

### Mục tiêu đề tài

* Xây dựng pipeline thu thập và xử lý dữ liệu thời tiết (làm sạch, xử lý dữ liệu chuỗi thời gian).
* Huấn luyện và đánh giá mô hình Machine Learning/Deep Learning (ví dụ: Linear Regression, LSTM, ARIMA).
* Triển khai demo dự đoán (inference) nhiệt độ cho các ngày tiếp theo dựa trên dữ liệu mới.
* So sánh và đánh giá hiệu quả mô hình bằng các metric phù hợp (MAE, MSE, RMSE).

## 2. Dataset

* Bộ dữ liệu thời tiết của Hà Nội từ năm 2015-2025
Kaggle: https://www.kaggle.com/datasets/thor1407/weather-daily-data-1102015-2025

## 3. Pipline

* Dataset → EDA → Clean → Encode → Train → Evaluate → Inference

## 4. Mô hình sử dụng

* Linear Regression
* Random Forest
* Long Shot-Term Memory 

### Kết quả

* Long Short-Term Memory
    * RMSE: 1.5854356651939232
    * MAE: 1.235079885908395
* Random Forest
    * RMSE: 1.7122
    * MAE: 1.2351
* Linear Regression
    * RMSE: 1.5849
    * MAE: 1.2351
    
## 5. Cách chạy chương trình
* B1: Tạo môi trường venv python 3.10.x
* B2: activate môi trường
* B3: chạy lệnh "pip install -r requirment.txt"


### Demo 
* streamlit run app/app1.py


## Tác giả
* Lê Tuấn Minh
* Mã sinh viên: 12523054
* Mã lớp: 12423TN