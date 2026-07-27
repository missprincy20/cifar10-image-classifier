import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image

st.title("CIFAR-10 Image Classifier")
st.write("Hello from Streamlit!")

# Configure Streamlit Page

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"

)

# Header

st.markdown(

'<p class="main-title">🖼️ CIFAR-10 Image Classification System</p>',

unsafe_allow_html=True

)

st.markdown(

'<p class="sub-title">Deep Learning using CNN + TensorFlow + Streamlit</p>',

unsafe_allow_html=True

)

st.divider()

# Sidebar

st.sidebar.title("Project Information")
st.sidebar.markdown("---")
st.sidebar.write("### Dataset")
st.sidebar.info(

"""

Dataset Used:

CIFAR-10
Images : 60,000
Classes : 10
Image Size : 32×32 RGB

"""

)

st.sidebar.write("Technologies")
st.sidebar.success("""

- Python
- TensorFlow
- Keras
- Streamlit

""")


# Supported Classes

st.sidebar.write("### Supported Classes")

classes = [

"✈ Airplane",

"🚗 Automobile",

"🐦 Bird",

"🐱 Cat",

"🦌 Deer",

"🐶 Dog",

"🐸 Frog",

"🐴 Horse",

"🚢 Ship",

"🚚 Truck"

]

for item in classes:

    st.sidebar.write(item)


# Load Trained CNN Model

@st.cache_resource
def load_cnn_model():

    model = tf.keras.models.load_model("cifar10_cnn_model.keras")

    return model

model = load_cnn_model()


# Load Class Names

@st.cache_data
def load_class_names():

    with open("class_names.json", "r") as file:

        class_names = json.load(file)

    return class_names

class_names = load_class_names()

# Image Upload

st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(

    "Choose an image",

    type=["jpg", "jpeg", "png"]

)

# Image Preprocessing Function

def preprocess_image(image):
    # Convert image to RGB
    image = image.convert("RGB")
    # Resize image to CIFAR-10 size
    image = image.resize((32, 32))
    # Convert image to NumPy array
    image = np.array(image)
    # Normalize pixel values
    image = image.astype("float32") / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

# Image Preview

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(

        image,

        caption="Uploaded Image",

        use_container_width=True

    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(image, use_container_width=True)


# Image Information

if uploaded_file is not None:

    st.write("### Image Information")

    st.write(f"Filename : {uploaded_file.name}")

    st.write(f"Size : {image.size}")

    st.write(f"Mode : {image.mode}")

# Prediction Button

predict_button = st.button(

    "Predict Image"

)

# Prediction

if predict_button:

    if uploaded_file is None:

        st.error("⚠ Please upload an image first.")

    else:
        try:

            processed_image = preprocess_image(image)

            prediction = model.predict(processed_image, verbose=0)

            predicted_index = np.argmax(prediction)

            predicted_class = class_names[predicted_index]

            confidence = prediction[0][predicted_index] * 100

            st.success("Prediction Completed Successfully!")

            st.subheader("Prediction Result")

            st.metric(

                label="Predicted Class",

                value=predicted_class

            )

            st.metric(

                label="Confidence",

                value=f"{confidence:.2f}%"

            )

            st.subheader("Top 3 Predictions")

            top3 = np.argsort(prediction[0])[::-1][:3]

            for i in top3:

                st.write(

                    f"**{class_names[i]}** : {prediction[0][i]*100:.2f}%"

                )

            st.subheader("Confidence Score")

            st.progress(float(confidence/100))

        # Result Layout

            if confidence >= 90:

                st.success("🟢 Very High Confidence")

            elif confidence >= 75:

                st.info("🔵 High Confidence")

            elif confidence >= 60:

                st.warning("🟡 Medium Confidence")

            else:

                st.error("🔴 Low Confidence")


            st.subheader("📊 Prediction Probabilities")

            probabilities = prediction[0] * 100

            chart_data = {
                class_names[i]: probabilities[i]
                for i in range(len(class_names))
            }

            st.bar_chart(chart_data)

            st.subheader("Top 3 Predictions")

            top3 = np.argsort(prediction[0])[::-1][:3]

            for rank, index in enumerate(top3, start=1):

                st.write(
                    f"**{rank}. {class_names[index]}** "
                    f"({prediction[0][index]*100:.2f}%)"
                )

            result = f"""
            CIFAR-10 Image Classification

            Predicted Class : {predicted_class}

            Confidence : {confidence:.2f}%

            Top 3 Predictions

            """

            for index in top3:

                result += (
                    f"{class_names[index]} : "
                    f"{prediction[0][index]*100:.2f}%\n"
                )

            st.download_button(

                label="📥 Download Result",

                data=result,

                file_name="prediction_result.txt",

                mime="text/plain"

            )

            st.divider()

            st.info("Upload another image to make a new prediction.")


        except Exception as e:

            st.error("Prediction Failed!")

            st.exception(e)


# Instructions

st.subheader("📖 Instructions")

st.info(

"""

1. Upload an image.

2. Click Predict.

3. The CNN model will classify the image.

4. Prediction confidence will also be displayed.

"""
)


# About Project

st.subheader("📚 About This Project")

st.write(

"""

This application uses a Convolutional Neural Network (CNN)

trained on the CIFAR-10 dataset.

The model can classify images into one of the following

10 categories:

Airplane, Automobile, Bird, Cat, Deer,

Dog, Frog, Horse, Ship and Truck.

"""
)

