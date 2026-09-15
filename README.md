# AI Resume & Career Intelligence System

An AI-powered resume analysis and career intelligence platform that analyzes resumes against job requirements, identifies skill gaps, predicts suitable job roles, and provides personalized career recommendations.

## 🚀 Live Demo

[AI Resume & Career Intelligence](https://ai-resume-job-matcher-ts7hzkgsue2vqqge6lpapt.streamlit.app/)

## 📌 Project Overview

The AI Resume & Career Intelligence System helps candidates understand how well their resume matches a target job role.

The system takes a resume in PDF format and analyzes:

* Resume content
* Technical skills
* Job-description requirements
* Resume-job similarity
* Skill match percentage
* Missing skills
* Suitable job roles
* Career recommendations
* Mock interview questions

## ✨ Key Features

### 📄 Resume Parsing

Extracts text automatically from uploaded PDF resumes using PyPDF.

### 🧠 Skill Extraction

Identifies technical skills such as Python, SQL, Machine Learning, Power BI, Pandas, NumPy, PostgreSQL, Git, APIs, and more.

### 🎯 Resume-Job Matching

Uses TF-IDF and cosine similarity to compare resume content with job descriptions.

### 📊 Skill Gap Analysis

Identifies:

* Matching skills
* Missing skills
* Skill match percentage

### 🤖 AI Job Role Prediction

Predicts suitable job roles using machine-learning classification.

Supported roles include:

* Data Analyst
* Business Analyst
* Machine Learning Engineer
* AI Engineer
* Software Engineer

### 🔎 Semantic Matching

Uses the `all-MiniLM-L6-v2` Sentence Transformer model to understand semantic similarity between resumes and job descriptions.

### 📚 Career Recommendations

Provides learning recommendations based on missing skills.

### 🎤 Mock Interview

Generates role-specific technical interview questions and evaluates answers using structured feedback.

## 🧪 Machine Learning

The project compares three classification algorithms:

* Multinomial Naive Bayes
* Logistic Regression
* Linear SVM

Text is converted into numerical features using **TF-IDF with unigram and bigram features**.

### Model Evaluation

The final model was selected using 5-fold stratified cross-validation.

| Model               | CV Accuracy | CV Macro F1 |
| ------------------- | ----------: | ----------: |
| Naive Bayes         |      99.00% |      98.98% |
| Logistic Regression |      99.00% |      98.98% |
| Linear SVM          |      99.00% |      98.98% |

Naive Bayes was selected as the final model based on cross-validation Macro F1 and saved using Joblib.

> Note: The training dataset currently contains 100 labeled examples. The reported evaluation results are based on this dataset and should not be interpreted as real-world production accuracy.

## 🛠️ Tech Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* TF-IDF
* Multinomial Naive Bayes
* Logistic Regression
* Linear SVM
* Sentence Transformers

### Data Processing

* Pandas
* NumPy

### Application

* Streamlit
* Plotly

### NLP

* TF-IDF
* Cosine Similarity
* Sentence Transformers

### Resume Processing

* PyPDF

### Model Persistence

* Joblib

## 📂 Project Structure

```text
AI-Resume-Job-Matcher/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── jobs.csv
│   └── training_data.csv
│
├── models/
│   └── job_role_model.pkl
│
└── src/
    ├── __init__.py
    ├── resume_parser.py
    ├── preprocessing.py
    ├── skill_extractor.py
    ├── matcher.py
    ├── recommender.py
    ├── ml_model.py
    ├── semantic_matcher.py
    └── interview.py
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ashikaas/AI-Resume-Job-Matcher.git
cd AI-Resume-Job-Matcher
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🧠 Train the Machine Learning Model

To retrain the job-role classification model:

```bash
python train_model.py
```

The trained model is saved to:

```text
models/job_role_model.pkl
```

## 🔄 Application Workflow

```text
Resume PDF
     ↓
Text Extraction
     ↓
Text Preprocessing
     ↓
Skill Extraction
     ↓
Resume–Job Matching
     ↓
Skill Gap Analysis
     ↓
ML Job Role Prediction
     ↓
Semantic Similarity
     ↓
Career Recommendations
     ↓
Mock Interview
```

## 🎯 Use Cases

* Resume evaluation
* Career planning
* Job-role discovery
* Skill-gap identification
* Interview preparation
* Personalized learning recommendations

## ⚠️ Limitations

The current system is a portfolio/research project rather than a production recruitment system.

The ML model is trained on a relatively small labeled dataset, so its evaluation results may not generalize to all real-world resumes and job descriptions.

The skill extraction component currently relies on a predefined technical-skill vocabulary.

## 🔮 Future Improvements

Potential improvements include:

* Larger real-world training datasets
* Advanced NLP-based skill extraction
* LLM-powered resume analysis
* Personalized learning paths
* Real-time job-market integration
* Improved interview evaluation
* Database-backed candidate profiles
* Deployment optimization

## 👩‍💻 Author

**Ashika Srivastava**

B.Tech Computer Science Engineering
Specialization: Artificial Intelligence and Analytics

## 📄 License

This project is intended for educational and portfolio purposes.
