
# Text Sentiment: Inferring given text's emotion

This project aims to analyze a given text and infer the emotion it conveys. I've used the DistilBERT model which achieves an accuracy of about 0.889. 

**The Challenge**

I had a problem with the available size of datasets. Since DistilBERT is based on the "Transformer" architecture which is very data-hungry, the model would always overfit on the first epoch while using smaller datasets (5-10k samples). 

**The Solution**

To solve this problem I created a custom dataset by concatenating 3 different smaller datasets. It is a very balanced dataset where each category (except one) has about 10k samples to ensure that the model learns how to identify all the emotions equally. You can access the dataset on [Kaggle](https://www.kaggle.com/datasets/prajwalnayakat/text-emotion).
# Files

## DistilBERT.ipynb

- Python Notebook to train the DistilBERT model.
- Model was trained on local hardware (GPU), can be trained on CPU but expect way longer training times.

## LR_RFC.ipynb

- Python Notebook where I trained Logistic Regression model and a Random Forest Classifier ensemble on the same dataset.
- Used these results as a baseline for DistilBERT.
- This approach used the TF-IDF vectorizer.
- Note: Please read through the file for a better understanding. 

## app.py 

- Flask API file.
- Renders the HTML templates and accepts the given text and returns the produced results.
- Handles the backend working of the website.

## convert_to_onnx.py

- File to convert the trained model from PyTorch to ONNX. 
- ONNX stands for Open Neural Network Exchange.
- It is a universal standard to share trained models in.
- Allows for a lightweight and efficient model which is crucial for deployment.

## creating_dataset.ipynb

- Python Notebook for creating the custom dataset.
- Includes cleaning, concatenating and feature engineering. 
- Note: Please read though the file for a better understanding.

## inference.py

- Python Notebook to use the trained model.
- Imports the trained model from Hugging Face.
- Accepts the text given to it by app.py (Flask API) and gives it to the trained model.
- Returns the results and the inference time to app.py (Flask API).

## manual_test.py

- Python Notebook to test and interact with the model without the Flask API.
- Allows you to use the model in your IDE's terminal. 
## Run it locally

If you want this project on your local machine:


1. Clone the repo
```
   git clone https://github.com/prajwalnayaka/Text_Sentiment.git
   ```

2. Install necessary libraries
```
   pip install -r requirements.txt
   ```

3. Run the Flask API file via your IDE's terminal.
```
   flask run
   ```

4. Open the localhost URL to view the website in your browser: http://127.0.0.1:5000/

## Website

Try the website [here.](https://text-emotion.nayaka.space/)

## How it works?

Copious amounts of sheer luck and a hint of magic :)
