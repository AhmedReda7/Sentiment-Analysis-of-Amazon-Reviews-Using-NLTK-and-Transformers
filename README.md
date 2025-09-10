# Amazon Reviews Sentiment Analysis

## Project Overview
This project analyzes Amazon Fine Food Reviews using both **traditional NLP (VADER, Logistic Regression, SVM, Random Forest)** and **state-of-the-art deep learning models (RoBERTa, Transformers pipeline)**.  

It includes:  
- **Exploratory Data Analysis (EDA)** of Amazon reviews dataset.  
- **Text preprocessing** (tokenization, stopword removal, stemming).  
- **Sentiment scoring** with **VADER** and **RoBERTa**.  
- **Model comparison** using Logistic Regression, SVM, and Random Forest.  
- A **Streamlit Web App** where users can input reviews and instantly see sentiment predictions.  

---

## Features
- Download and process **Amazon Fine Food Reviews dataset** (Kaggle).  
- Perform **EDA** and visualize review distributions.  
- Apply **VADER sentiment analysis** for lexicon-based scoring.  
- Apply **RoBERTa transformer model** for context-based sentiment classification.  
- Compare **classical ML classifiers**: Logistic Regression, SVM, Random Forest.  
- Deploy an **interactive Streamlit application** (`app.py`) for real-time review analysis.  

---

## Dataset
Dataset: [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  

- **Score** → Rating (1–5 stars).  
- **Text** → Review content.  
- **Id** → Review ID.  
- Subset of 500 reviews used for faster experimentation.  

---

## Tech Stack
- **Python**  
- **Libraries:** NLTK, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, Transformers (Hugging Face), SciPy  
- **Models:** VADER, RoBERTa, Logistic Regression, SVM, Random Forest  
- **Web App:** Streamlit  

---

## Installation & Usage

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/amazon-sentiment-analysis.git
   cd amazon-sentiment-analysis
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
3. **Run Jupyter Notebook (for EDA & model comparison)**
    ```bash
    jupyter notebook
4. **Run Streamlit Web App**
   ```bash
   streamlit run app.py
5. Open in browser → Enter a review → Get sentiment predictions from **VADER and RoBERTa**.
6. 
## 📊 Example Results

### VADER Example
```json
{
  "neg": 0.2,
  "neu": 0.6,
  "pos": 0.2,
  "compound": 0.0
}
### RoBERTa Example
```json
{
  "RoBERTa Negative": 0.15,
  "RoBERTa Neutral": 0.10,
  "RoBERTa Positive": 0.75
}
