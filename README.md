------
# Plant Disease Detection System for Sustainable Agriculture 🌱🚀

This project aims to empower farmers with an AI-driven tool for early and accurate plant disease detection. By leveraging advanced machine learning techniques and the Streamlit framework, this system provides real-time disease diagnosis and actionable treatment recommendations to promote sustainable farming practices.

## Features
- **AI-Powered Disease Detection**: Utilizes Convolutional Neural Networks (CNNs) built with TensorFlow and Keras to analyze images of plant leaves and detect various diseases with high precision.
- **User-Friendly Web Application**: Developed using Streamlit, offering an intuitive interface accessible via smartphones and other devices.
- **Real-Time Diagnostics**: Provides instant disease diagnosis and actionable treatment recommendations, helping farmers make informed decisions to protect their crops.
- **Data Visualization**: Utilizes libraries such as NumPy, Seaborn, and Matplotlib for preprocessing, data augmentation, and visualization to enhance model performance and understanding.

## Tools and Technologies
- **Python**: Core programming language for the project.
- **TensorFlow/Keras**: Libraries for building and training the deep learning model.
- **Streamlit**: Framework for creating the interactive web application.
- **Jupyter Notebook**: Environment for developing and testing the model.
- **NumPy**: Library for numerical computations.
- **Seaborn & Matplotlib**: Libraries for data visualization and analysis.

## Getting Started
To get a local copy up and running, follow these steps:

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- Streamlit
- Jupyter Notebook
- NumPy
- Seaborn
- Matplotlib

### Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/plant-disease-detection.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd plant-disease-detection
    ```
3. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
1. **Open the Jupyter Notebook** to explore the model training and evaluation:
    ```bash
    jupyter notebook
    ```
2. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

## Usage
1. **Upload an image** of a plant leaf via the web application.
2. The model analyzes the image and provides a **disease diagnosis**.
3. View **actionable treatment recommendations** to address the diagnosed disease.

## Diagram of the Proposed Solution
![Diagram](diagram.png)

### Explanation of the Diagram
1. **Data Collection**: Images of plant leaves are collected to form the dataset.
2. **Data Preprocessing**: Images are preprocessed to enhance quality and uniformity.
3. **Model Training**: A CNN is trained using TensorFlow and Keras.
4. **Evaluation & Optimization**: The model is evaluated and optimized for accuracy.
5. **Web App Development**: Streamlit is used to develop the user-friendly web application.
6. **User Interface**: The UI allows farmers to upload images and receive diagnoses and treatments.

## Acknowledgments
This project is part of the AICTE Internship on "AI: Transformative Learning" with TechSaksham – a joint CSR initiative of Microsoft & SAP. Special thanks to my supervisor, Pavan Kumar U, and project guide, P. Raja, Master Trainer, Edunet Foundation, for their invaluable support and guidance.

## License
None
.........
