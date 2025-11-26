Breast Cancer Malignancy Classifier (Streamlit App)



This is a simple web application built with Streamlit and Python to predict the malignancy of a breast mass (Benign or Malignant) using a Support Vector Machine (SVM) model trained on the Breast Cancer Wisconsin (Diagnostic) Dataset.



The application loads a pre-trained model (svm\_cancer\_model.pkl) and a fitted data scaler (scaler\_cancer.pkl) to ensure predictions are made correctly on unscaled input data.



📁 Project Structure



Your final directory structure should look like this:



cancer-classifier/

├── app.py                  # The main Streamlit application

├── requirements.txt        # Python dependencies

├── README.md               # This file

├── svm\_cancer\_model.pkl    # The trained SVM model (Output of train\_model.py)

└── scaler\_cancer.pkl       # The fitted StandardScaler (Output of train\_model.py)





🚀 Setup and Run



1\. Prerequisites



You need Python 3 installed on your system.



2\. Install Dependencies



Navigate to the project directory (cancer-classifier/) in your terminal and install the necessary libraries:



pip install -r requirements.txt





3\. Run the Application



Start the Streamlit server from the project directory:



streamlit run app.py





4\. Usage



The application will open in your web browser (usually on http://localhost:8501).



Use the sidebar inputs to enter the 30 features (Mean, SE, and Worst values) for the patient data.



Click the "Analyze Patient Data" button to get the malignancy prediction (Benign or Malignant) and the associated probability.

