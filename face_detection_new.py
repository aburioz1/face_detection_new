import cv2
import streamlit as st
import numpy as np

# Function to detect faces and draw rectangles
def detect_faces(image, min_neighbors, scale_factor, rectangle_color):
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, minNeighbors=min_neighbors, scaleFactor=scale_factor)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image


# Function to save the image with detected faces
def save_image_with_faces(image, output_path):
    cv2.imwrite(output_path, image)


# Streamlit app
def main():
    st.title("Face Detection using Viola-Jones Algorithm")

    # Instructions
    st.write("## Instructions")
    st.write("1. Upload an image using the file uploader below.")
    st.write("2. Adjust the parameters for face detection using the sliders on the left sidebar.")
    st.write("3. Choose the color for the rectangles drawn around the detected faces using the 'Rectangle Color' button.")
    st.write("4. Click the 'Detect Faces' button to apply face detection.")
    st.write("5. The image with detected faces will be displayed, and you can save it using the 'Save Image' button.")

    st.sidebar.title("Settings")

    # Slider for minNeighbors parameter
    min_neighbors = st.sidebar.slider("minNeighbors", min_value=1, max_value=10, value=5)

    # Slider for scaleFactor parameter
    scale_factor = st.sidebar.slider("scaleFactor", min_value=1.1, max_value=1.5, step=0.1, value=1.2)

    # Color picker for rectangle color
    rectangle_color = st.sidebar.color_picker("Rectangle Color", "#FF0000")

    # File uploader for image upload
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display the original image
        st.image(image, channels="BGR", caption="Original Image", use_column_width=True)

        if st.button("Detect Faces"):
            # Detect faces and draw rectangles
            image_with_faces = detect_faces(image, min_neighbors, scale_factor, rectangle_color)

            # Display the image with detected faces
            st.image(image_with_faces, channels="BGR", caption="Image with Detected Face(s)", use_column_width=True)

            if st.button("Save Image"):
                # Save the image with detected faces
                save_image_with_faces(image_with_faces, "image_with_faces.jpg")
                st.write("Image saved successfully!")


if __name__ == "__main__":
    main()
