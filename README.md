# 📚 Exam Preparation Assistant  
AI-powered study helper for summarizing PDFs/PPTs/YouTube videos, generating practice questions, and creating study plans using **RAG + Gemini**.

---

## 🚀 Features  
✔ Upload **PDF** or **PPTX**  
✔ Enter a **YouTube video link**  
✔ Automatic **text extraction + chunking**  
✔ SentenceTransformer embeddings + **FAISS** vector store  
✔ **Gemini 2.5 Flash** for summary, practice questions, and study plan  
✔ Built-in **safety filter**  
✔ Gradio interface (Runs locally, in Colab, and on Hugging Face Spaces)  
✔ Production-ready `app.py`  

---

## 🔧 Tech Stack  
- **Python 3.10+**  
- **LangChain**  
- **Sentence Transformers**  
- **FAISS**  
- **Gradio**  
- **Gemini API**  
- **PyPDF / python-pptx / YouTube Transcript API**

---

# 📁 Project Structure

```

Exam-Preparation-Assistant/
│── app.py
│── requirements.txt
│── README.md
└── assets/

````

---

# 🔑 Environment Setup

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
````

---

### 2️⃣ Set your Gemini API key

Create a **.env** file:

```
GEMINI_API_KEY=your_key_here
```

Or export directly:

```bash
export GEMINI_API_KEY="your_key_here"
```

Windows (PowerShell):

```powershell
setx GEMINI_API_KEY "your_key_here"
```

---

# ▶ Run the App (Locally)

```bash
python app.py
```

Gradio will open at:

```
http://127.0.0.1:7860
```

---

# ▶ Run on Google Colab

Add this at the top of your Colab notebook:

```python
!git clone https://github.com/abaidurerehman/-Exam-Preparation-Assistant.git
%cd -Exam-Preparation-Assistant
!pip install -r requirements.txt
```

Run the app:

```python
!python app.py
```

Use the public Gradio link that appears in output.

---

# 🚀 Deploy on Hugging Face Spaces

### 1️⃣ Create a new HF Space

Choose:

* **SDK: Gradio**
* **Runtime: Python 3.10**

### 2️⃣ Upload these files:

* `app.py`
* `requirements.txt`
* `README.md`

### 3️⃣ Add your HF secret key

Go to:

**Settings → Repository Secrets → New Secret**

```
Key: GEMINI_API_KEY
Value: your_key_here
```

### 4️⃣ Auto-build starts

Demo:

👉 **[https://abaidurerehman-exam-preparation-assistant-7ea98e0.hf.space](https://abaidurerehman-exam-preparation-assistant-7ea98e0.hf.space)**

---

# 🛠 GitHub Commands

### Initialize git & push project

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin git@github.com:abaidurerehman/-Exam-Preparation-Assistant.git
git push -u origin main
```

---

# 🧠 How It Works (Architecture)

### 1. Upload File or Paste YouTube Link

↓

### 2. Text Extraction

* PDF → pdfplumber
* PPTX → python-pptx
* YouTube → YouTubeTranscriptAPI

↓

### 3. Preprocessing

* Cleaning
* Chunking (1500–2000 tokens)

↓

### 4. Embeddings

Using **SentenceTransformer**
Stored in **FAISS vector DB**

↓

### 5. Gemini LLM

* Summary
* Practice Questions
* Study Plan

↓

### 6. Gradio UI

Displays all generated outputs.

---

# ⚖️ Ethical Considerations

* Built-in **safety filter** blocks harmful content
* No bias-prone prompts sent to Gemini
* User data is **never stored**
* Only content-based tutoring, no cheating/answer-revealing

---

# 🤝 Contributing

Pull requests are welcome! Feel free to improve extraction, add OCR, or enhance the UI.



Do you want me to do that next?
```
