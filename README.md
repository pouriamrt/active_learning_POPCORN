# POPCORN: Active Learning for Population Health Modelling Consensus Reporting

**POPCORN** (Population Health Modelling Consensus Reporting Network) presents an interactive **Streamlit-based active learning tool** for classifying scientific papers. This lightweight, human-in-the-loop system enables researchers to iteratively label papers as *included* or *excluded*, with real-time model updates. 

Ideal for tasks such as:
- Systematic reviews and evidence synthesis
- Population health modeling
- Inclusion/exclusion screening in research workflows

Live System: [POPCORN Active Learning App](http://ec2-52-60-155-21.ca-central-1.compute.amazonaws.com/popcorn_al)

---

## 🚀 Features

- ✅ Label papers interactively through a web interface
- 🔁 Active learning with real-time model retraining
- 📊 Live stats on labeled data
- 💾 Local persistence of model and labels (auto-generated)
- 🧠 Lightweight ML pipeline: TF-IDF + Logistic Regression

---

## 🗂 Project Structure

```
├── app.py              # Streamlit application
├── papers.csv          # Input dataset (title + abstract)
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
```

> ⚠️ `model.pkl` is generated on first run and **not included** in the repository.

---

## ⚙️ How It Works

1. **Load papers** from `papers.csv`
2. **Initialize or load a model** (`LogisticRegression` + `TfidfVectorizer`)
3. **Label papers** (Include / Exclude) one at a time
4. **Retrain the model** as new labels are added
5. **Save model + vectorizer + labeled data** to `model.pkl`

---

## 📄 Input Format

Ensure `papers.csv` follows this structure:

```csv
title,abstract
"Understanding Cardiovascular Risks","This paper explores the relationship between..."
"AI in Population Health","We propose a machine learning approach for..."
```

---

## 🧪 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/popcorn-active-learning.git
cd popcorn-active-learning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## 📊 Interface Overview

- One randomly selected **unseen paper**
- Buttons to **✅ Include** or **❌ Exclude**
- Expandable abstract view
- Model stats: labeled count, class distribution

---

## 📌 Notes

- Binary classification only (include/exclude)
- Model improves with more labeled data
- To reset the session, delete `model.pkl`

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit pull requests or open issues.

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
