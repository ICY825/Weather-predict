import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự báo Thời tiết", page_icon="🔮", layout="wide")

# --- HÀM HỖ TRỢ ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if 'datetime' in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# --- GIAO DIỆN CHÍNH ---
st.title("🔮 Phần mềm Dự báo Nhiệt độ Tương lai")
st.markdown("Huấn luyện mô hình AI trên dữ liệu lịch sử để dự đoán nhiệt độ cho **N ngày tiếp theo**.")

# 1. SIDEBAR - CẤU HÌNH
with st.sidebar:
    st.header("1. Dữ liệu & Tham số")
    uploaded_file = st.file_uploader("Tải file CSV (Hanoi_weather.csv)", type=['csv'])
    
    st.divider()
    st.subheader("Cấu hình Mô hình")
    seq_length = st.slider("Số ngày dùng để đoán (Window)", 10, 60, 30, help="Dùng bao nhiêu ngày quá khứ để đoán 1 ngày tương lai")
    epochs = st.number_input("Số lần học (Epochs)", 10, 200, 20)
    
    # Reset Session nếu đổi file
    if uploaded_file:
        if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.clear()
            st.session_state.last_file = uploaded_file.name

# 2. XỬ LÝ DỮ LIỆU
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Chọn cột mục tiêu
    target_col = 'temp'
    if target_col not in df.columns:
        st.error(f"Trong file CSV phải có cột '{target_col}'")
        st.stop()

    # Hiển thị dữ liệu gốc
    with st.expander("Xem dữ liệu lịch sử", expanded=False):
        st.dataframe(df.tail(10))

    # Chuẩn bị dữ liệu
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[[target_col]])
    X, y = create_sequences(data_scaled, seq_length)

    # 3. HUẤN LUYỆN MÔ HÌNH
    col_train, col_status = st.columns([1, 3])
    
    with col_train:
        train_btn = st.button("🚀 Huấn luyện Mô hình", type="primary", use_container_width=True)

    if train_btn:
        with st.status("Đang huấn luyện AI...", expanded=True) as status:
            st.write("🧠 Đang xây dựng mạng LSTM...")
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            
            st.write("🔄 Đang học dữ liệu lịch sử...")
            model.fit(X, y, batch_size=32, epochs=epochs, verbose=0)
            
            # Lưu vào Session State
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['last_sequence'] = data_scaled[-seq_length:] # Lấy chuỗi cuối cùng của file
            st.session_state['last_date'] = df['datetime'].iloc[-1]
            st.session_state['is_trained'] = True
            
            status.update(label="Huấn luyện hoàn tất! Bạn có thể dự báo ngay.", state="complete", expanded=False)

    # 4. DỰ BÁO TƯƠNG LAI (Chỉ hiện khi đã train)
    if st.session_state.get('is_trained'):
        st.divider()
        st.header("2. Dự báo Tương lai")
        
        c1, c2 = st.columns([1, 3])
        with c1:
            days_to_predict = st.number_input("Số ngày muốn dự báo:", min_value=1, max_value=365, value=7)
            predict_btn = st.button("🔮 Dự báo ngay")
        
        if predict_btn:
            model = st.session_state['model']
            scaler = st.session_state['scaler']
            curr_seq = st.session_state['last_sequence'].copy()
            last_date = st.session_state['last_date']
            
            future_preds = []
            future_dates = []
            
            # Vòng lặp dự báo từng ngày
            with st.spinner(f"Đang tính toán nhiệt độ cho {days_to_predict} ngày tới..."):
                for i in range(days_to_predict):
                    # 1. Dự báo bước tiếp theo
                    # curr_seq shape đang là (seq_length, 1) -> cần reshape thành (1, seq_length, 1) cho model
                    input_seq = curr_seq.reshape(1, seq_length, 1)
                    pred_value = model.predict(input_seq, verbose=0) # Kết quả dạng scale
                    
                    # 2. Lưu kết quả
                    future_preds.append(pred_value[0, 0])
                    
                    # 3. Tạo ngày tương lai
                    next_date = last_date + datetime.timedelta(days=i+1)
                    future_dates.append(next_date)
                    
                    # 4. Cập nhật chuỗi đầu vào: Bỏ giá trị cũ nhất, thêm giá trị vừa dự báo vào cuối
                    curr_seq = np.append(curr_seq[1:], pred_value, axis=0)

            # Đảo ngược scale để ra nhiệt độ thật
            future_temps = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
            
            # Tạo DataFrame kết quả
            df_future = pd.DataFrame({
                'Ngày': future_dates,
                'Nhiệt độ dự báo (°C)': future_temps.flatten()
            })
            
            # --- HIỂN THỊ KẾT QUẢ ---
            st.success(f"Đã dự báo xong {days_to_predict} ngày!")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.subheader("Bảng chi tiết")
                st.dataframe(df_future.style.format({"Nhiệt độ dự báo (°C)": "{:.2f}"}), use_container_width=True)
                
                # Nút tải về
                csv = df_future.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải kết quả CSV", data=csv, file_name="du_bao_tuong_lai.csv", mime="text/csv")

            with col_res2:
                st.subheader("Biểu đồ Xu hướng")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Lấy một ít dữ liệu cũ để nối biểu đồ cho đẹp (30 ngày cuối)
                history_df = df.tail(60)
                
                ax.plot(history_df['datetime'], history_df[target_col], label='Lịch sử gần đây', color='gray', alpha=0.5)
                ax.plot(df_future['Ngày'], df_future['Nhiệt độ dự báo (°C)'], label='Dự báo tương lai', color='red', marker='o', markersize=4)
                
                ax.set_title(f"Dự báo nhiệt độ {days_to_predict} ngày tới")
                ax.set_xlabel("Ngày")
                ax.set_ylabel("Nhiệt độ (°C)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # Format ngày
                ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
                plt.xticks(rotation=45)
                
                st.pyplot(fig)

                

else:
    st.info("👋 Vui lòng tải file dữ liệu ở thanh bên trái để bắt đầu.")
    # Demo data generator
    if st.checkbox("Dùng dữ liệu mẫu (Demo)"):
        dates = pd.date_range(start="2023-01-01", periods=365)
        temp = 25 + 8 * np.sin(np.linspace(0, 10, 365)) + np.random.normal(0, 2, 365)
        df_demo = pd.DataFrame({"datetime": dates, "temp": temp})
        st.download_button("Tải file mẫu CSV", df_demo.to_csv(index=False).encode('utf-8'), "sample.csv")