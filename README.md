# 📄 Conversational PDF Chatbot

This project is a web-based chatbot that allows users to upload a PDF and ask questions about its content. The chatbot reads the PDF, understands the content using AI, and responds to user queries in a conversational way.

---

## 🔍 What It Does

- Upload any PDF file.
- Ask questions about the content inside the PDF.
- Get context-aware answers using a large language model.
- Keeps a history of your chat during the session.

---

## 🛠️ Technologies Used

- **Streamlit** – Web interface
- **LangChain** – Chain and retrieval logic
- **Groq API** – For high-speed language model responses
- **HuggingFace Embeddings** – For converting text to vectors
- **ChromaDB** – For vector storage and retrieval
- **PyPDFLoader** – For reading PDF files
- **dotenv** – For managing environment variables

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/conversational-pdf-chatbot.git
cd conversational-pdf-chatbot
