import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load class labels
with open("class_labels.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = tf.lite.Interpreter(
    model_path="skin_cancer_quantized.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit UI
st.title("Lightweight Skin Cancer Detection System")

st.write("Upload a dermoscopic skin image for prediction.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    image = image.resize((224, 224))

    # Convert image to numpy array
    img_array = np.array(image, dtype=np.float32)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Set tensor
    interpreter.set_tensor(
        input_details[0]['index'],
        img_array
    )

    # Run prediction
    interpreter.invoke()

    # Get output
    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )

    # Get predicted class
    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction)

    # Display results
    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence Score: {confidence:.2f}")
