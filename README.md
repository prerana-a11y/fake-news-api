# 📰 Fake News Detection System

An AI-powered web application that detects whether a news article is **REAL** or **FAKE** using Machine Learning.

---

## 🚀 Features

* 🔍 Detects fake vs real news instantly
* ⚡ Fast API response using Flask
* 🧠 Trained on Kaggle Fake & Real News Dataset
* 🎨 Interactive UI (chatbot-style / typing animation supported)
* 📊 Scalable for future improvements (deep learning, APIs)

---

## 🧠 Model Details

* **Algorithm Used:** Logistic Regression / Passive Aggressive Classifier
* **Vectorization:** TF-IDF (Term Frequency - Inverse Document Frequency)
* **Dataset:** Kaggle Fake & Real News Dataset
* **Labels:**

  * `1 → Real News`
  * `0 → Fake News`

---

## 📂 Project Structure

```
fake-news-detector/
│
├── app.py                # Flask backend
├── model.pkl             # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
├── templates/
│   └── index.html        # Frontend UI
├── static/
│   ├── style.css
│   └── script.js
├── dataset/
│   ├── True.csv
│   └── Fake.csv
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the application:

```
python app.py
```

4. Open in browser:

```
http://127.0.0.1:5000/
```

---

## 🧪 How It Works

1. User enters a news headline or article
2. Text is cleaned and processed
3. TF-IDF converts text → numerical features
4. Model predicts:

   * Fake ❌
   * Real ✅

---

## 📊 Dataset Info

* Source: Kaggle
* Files:

  * `True.csv` → Real News
  * `Fake.csv` → Fake News

Combined and labeled before training.

---

## ⚠️ Known Issues

* Model may predict **everything as fake** if dataset is unbalanced
* Needs better generalization for short inputs
* Accuracy depends on training data quality

---

## 🔧 Future Improvements

* 🤖 Use Deep Learning (LSTM / BERT)
* 🌐 Integrate live news APIs
* 📊 Show confidence score (% real vs fake)
* 🧾 Highlight suspicious words
* 📱 Mobile responsive UI

---

## 👨‍💻 Author

**Prerana Dash**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
