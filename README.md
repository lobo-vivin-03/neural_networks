# Neural Networks Model for Prediction 


This project predicts students' **final exam scores** based on input features such as quiz scores and time spent studying. The system uses **machine learning** with a **neural network model** built using Keras and TensorFlow.

## **Project Structure**

The repository contains the following files:

1. **`data.py`**  
   - This script generates synthetic data, including:
     - **Quiz scores** (input feature)
     - **Time spent** on quizzes (input feature)
     - **Final exam score** (output target variable)  
   - The generated data mimics real-world patterns and is saved into a CSV file named **`enhanced_data.csv`**.  
   - **Purpose**: To create training data for the neural network.

2. **`enhanced_data.csv`**  
   - A CSV file containing the generated data.  
   - **Structure**:
     - `quiz_scores`: Numeric values representing quiz marks.
     - `time_spent`: Hours spent studying or completing quizzes.
     - `final_score`: The actual exam score (target variable).  
   - **Purpose**: Serves as input data for model training.

3. **`training_model.py`**  
   - This script trains a **feedforward neural network** using Keras and TensorFlow.  
   - **Key steps**:
     - **Data Preprocessing**:  
       - Input (`quiz_scores`, `time_spent`) and target (`final_score`) are **normalized** using **MinMaxScaler**.
       - Scalers are saved as **`scaler_X.pkl`** and **`scaler_y.pkl`** for reuse during predictions.
     - **Model Architecture**:
       - **Input Layer**: Accepts scaled features.
       - **Hidden Layers**: Include **ReLU activation functions** to introduce non-linearity.
       - **Output Layer**: A single neuron with **linear activation** for continuous output prediction.
     - **Loss Function**:  
       - **Mean Squared Error (MSE)** is used as the loss function for regression tasks.
     - **Optimizer**:  
       - Adam optimizer minimizes the loss during training.
     - **Model Saving**:
       - The trained model is saved as **`final_exam_model.h5`**.
       - Scalers for input and output data are saved as **`scaler_X.pkl`** and **`scaler_y.pkl`**.  
   - **Purpose**: Train and save the prediction model.

4. **`final_exam_model.h5`**  
   - The trained neural network model saved in HDF5 format.  
   - **Purpose**: Used for making predictions on new input data.

5. **`scaler_X.pkl` and `scaler_y.pkl`**  
   - These files store the **MinMaxScaler** objects used during training.  
   - **Purpose**: Ensure input features and target variables are scaled consistently when making predictions.

6. **`model.py`**  
   - This script loads the trained model (`final_exam_model.h5`) and the scalers (`scaler_X.pkl` and `scaler_y.pkl`) to **predict final exam scores** for new data.  
   - **Key steps**:
     - Load the pre-trained neural network model.
     - Load the scalers for input and output normalization.
     - Scale the new input data (`quiz_scores` and `time_spent`).
     - Predict the final exam score using the model.
     - Reverse the scaling on the output to get the prediction in the original scale.  
   - **Purpose**: Perform predictions on unseen data using the trained model.

7. **`README.md`**  
   - This documentation file explaining the project structure, technical details, and file responsibilities.

---

## **Technical Details**

### **1. Machine Learning Model**
- **Type**: Feedforward Neural Network  
- **Framework**: Keras with TensorFlow backend  
- **Loss Function**: Mean Squared Error (MSE)  
- **Optimizer**: Adam Optimizer  
- **Activation Functions**:
  - **ReLU** in hidden layers for non-linearity.
  - **Linear Activation** in the output layer for regression tasks.

### **2. Data Preprocessing**
- **Feature Scaling**: MinMaxScaler  
- Input (`quiz_scores` and `time_spent`) and target (`final_score`) are scaled to a range [0, 1] for efficient model training.  
- Scalers are saved and reused during prediction.

### **3. Training Process**
- **Input Features**:  
  - `quiz_scores` (Quiz marks)  
  - `time_spent` (Hours spent)  
- **Output Variable**:  
  - `final_score` (Final exam score)  
- The model is trained to predict `final_score` based on the input features.

---

## **How to Use the Project**

1. **Generate Data**  
   Run `data1.py` to generate synthetic data and save it as `enhanced_data.csv`.  
   ```bash
   python data1.py
2. **Train the Model**
Run `training_model.py` to train the neural network and save the model and scalers.
    ```bash
    python training_model.py

3. **Predict Final Exam Scores**
Use `model.py` to predict final scores based on new input data.
    ```bash
    python model.py

## Requirements
Ensure you have the following Python libraries installed:

`tensorflow (2.18.0 or higher)`

`keras (3.7.0)`

`numpy`

`pandas`

`scikit-learn`

`pickle`

Install dependencies using:
  ```bash
  pip install tensorflow keras numpy pandas scikit-learn


