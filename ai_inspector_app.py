import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Inspector", layout="wide")
st.title("🤖 AI ตรวจสอบคุณภาพสินค้า (จำลอง)")
st.write("อัปโหลดรูปภาพเพื่อทดสอบว่า 'ดี' หรือ 'ชำรุด' (โมเดลนี้ฝึกจากชุดข้อมูลถั่ว)")

# --- 2. ตั้งชื่อคลาส (จากชุดข้อมูล 'beans') ---
# (0: angular_leaf_spot, 1: bean_rust, 2: healthy)
class_names = ['Angular Leaf Spot (ชำรุด)', 'Bean Rust (ชำรุด)', 'Healthy (ดี)']
MODEL_FILE = 'my_ai_model.h5'
IMG_SIZE = 150 # ต้องตรงกับตอนที่ Train

# --- 3. ฟังก์ชันโหลดโมเดล (ใช้ Cache เพื่อความเร็ว) ---
# @st.cache_resource จะเก็บโมเดลใน RAM ทำให้โหลดครั้งเดียว
@st.cache_resource
def load_ai_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        return None

# --- 4. ฟังก์ชันประมวลผลรูปภาพ ---
def process_image(image_file):
    try:
        # เปิดรูปภาพ
        image = Image.open(image_file).convert('RGB')
        # ย่อ/ขยายรูปภาพ
        image = image.resize((IMG_SIZE, IMG_SIZE))
        # แปลงเป็น Array
        image_array = np.array(image)
        # Normalize (เหมือนตอน Train)
        image_array = image_array / 255.0
        # ขยายมิติ (ให้เป็น 4D เพื่อป้อนเข้าโมเดล)
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลรูปภาพ: {e}")
        return None

# --- 5. ส่วนหลักของแอป ---

# โหลดโมเดล
model = load_ai_model(MODEL_FILE)

if model is None:
    st.error(f"ไม่พบไฟล์โมเดล '{MODEL_FILE}'")
    st.info("กรุณาดาวน์โหลดไฟล์ 'my_ai_model.h5' จาก Google Colab มาไว้ในโฟลเดอร์เดียวกับแอปนี้")
else:
    # สร้างที่อัปโหลดไฟล์
    uploaded_file = st.file_uploader(
        "อัปโหลดรูปภาพที่ต้องการตรวจสอบ (JPG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # แสดงรูปภาพที่อัปโหลด
        st.image(uploaded_file, caption="รูปภาพที่อัปโหลด", use_column_width=True)
        
        # สร้างปุ่มสำหรับเริ่มตรวจสอบ
        if st.button("เริ่มตรวจสอบ (Predict)"):
            with st.spinner("AI กำลังวิเคราะห์..."):
                # 1. ประมวลผลรูปภาพ
                processed_image = process_image(uploaded_file)
                
                if processed_image is not None:
                    # 2. ป้อนเข้าโมเดล
                    prediction = model.predict(processed_image)
                    
                    # 3. หาผลลัพธ์
                    score = np.max(prediction[0])
                    class_index = np.argmax(prediction[0])
                    result_class = class_names[class_index]
                    
                    # 4. แสดงผลลัพธ์
                    st.success(f"การวิเคราะห์เสร็จสิ้น!")
                    
                    if result_class == 'Healthy (ดี)':
                        st.balloons()
                        st.subheader(f"ผลลัพธ์: {result_class}")
                        st.subheader(f"ความมั่นใจ: {score * 100:.2f}%")
                    else:
                        st.error(f"ผลลัพธ์: {result_class}")
                        st.subheader(f"ความมั่นใจ: {score * 100:.2f}%")