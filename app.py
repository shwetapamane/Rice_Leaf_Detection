import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="rice_leaf_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Bacterial Leaf Blight", "Leaf Smut", "Brown Spot"]

st.title("🌱 Rice Leaf Disease Detection")
uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    
