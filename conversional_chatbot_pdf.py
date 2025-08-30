import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from chromadb.config import Settings
import tempfile
import os
from dotenv import load_dotenv

# load the environment variables
load_dotenv()

# Add this function at the top of your file, after imports
def validate_file_size(uploaded_file, max_size_mb=10):
    """Validate if the uploaded file is within size limit"""
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
    return file_size <= max_size_mb


os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")
st.title("Conversational PDF Chatbot")

# sidebar for api key and LLM Models
with st.sidebar:
    st.header("Authentication")
    api_key = st.text_input("Enter your Groq API Key:", type="password")
    session_id = st.text_input("Session ID", value="Default Session")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

if api_key and not api_key.startswith("gsk_"):
    st.error("Invalid Groq API key format. Please check your API key.")
    st.stop()
elif api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192", temperature=temperature)
    uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Add a progress bar for overall processing
        progress_bar = st.progress(0)
        processed_files = []
        all_documents = []

        for idx, uploaded_file in enumerate(uploaded_files):

            if not validate_file_size(uploaded_file):
                st.error(f"File {uploaded_file.name} exceeds 10MB limit. Skipping.")
                st.stop()

            with st.spinner(f"Processing {uploaded_file.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_pdf_path = tmp_file.name

                # Load the PDF content using pdf loader
                loader = PyPDFLoader(tmp_pdf_path)
                documents = loader.load()

                # Add source metadata to each document
                for doc in documents:
                    doc.metadata["source_file"] = uploaded_file.name
                    doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1

                all_documents.extend(documents)
                processed_files.append(tmp_pdf_path)
                
                # Update progress bar
                progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # Chunk all documents together
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        splits = text_splitter.split_documents(all_documents)

        
        with st.sidebar:
            st.subheader(" Document Statistics")
            st.write(f"Number of PDFs: {len(uploaded_files)}")
            
            # Group documents by source file
            file_stats = {}
            for doc in all_documents:
                filename = doc.metadata["source_file"]
                if filename not in file_stats:
                    file_stats[filename] = {"pages": set()}
                file_stats[filename]["pages"].add(doc.metadata["page_number"])
            
            # Display stats for each file
            for filename, stats in file_stats.items():
                with st.expander(f"📄 {filename}"):
                    st.write(f"Pages: {len(stats['pages'])}")

        


        # embedding the context and store in vector database
        @st.cache_resource
        def get_vectorstore(_splits):
            return Chroma.from_documents(
                documents=_splits,
                embedding=embeddings,
                client_settings=Settings(anonymized_telemetry=False)
            )

        vectorstore = get_vectorstore(splits)
        
        # Add this after vectorstore creation
        
            
        retriever = vectorstore.as_retriever()

        # Prompt for context
        context_prompt = """
        Given a chat history and the latest user question,
        which might reference context in the chat history,
        formulate a standalone question which can be understood
        without the chat history. Do not answer the question,
        just reformulate it if needed and otherwise return it as is.
        """

        standalone_prompt = ChatPromptTemplate.from_messages([
            ("system", context_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, standalone_prompt)

        # System prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are a helpful AI assistant for answering questions strictly based on the provided PDF context.
                Rules:
                1. Use ONLY the information from {context}.
                2. If the answer is not present in {context}, respond exactly:
                "Your query is not related to the PDF context."
                3. If the user asks something irrelevant to the PDF, respond exactly:
                "Your query is not related to the PDF context."
                4. If the user asks anything violent, harmful, hateful, illegal, sexual, or unsafe, respond exactly:
                "I cannot respond to that type of request."
                5. Never guess, assume, or use outside knowledge — stay within the provided context.
                6. If multiple pieces of context are provided, combine them logically but without adding new facts.
                """)
,
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # create RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        if "store" not in st.session_state:
            st.session_state.store = {}

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        st.subheader(" Ask a Question from PDF")
        user_input = st.text_input("Enter your query below 👇", key="query_input")

        if user_input:
            with st.spinner("Generating response..."):
                try:
                    session_history = get_session_history(session_id)
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}},
                    )

                    st.markdown("#### Assistant:")
                    st.write(response["answer"])

                    with st.expander("🧾 Chat History"):
                        for msg in session_history.messages:
                            role = "User" if msg.type == "human" else "🤖 Assistant"
                            st.markdown(f"**{role}:** {msg.content}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        if "tmp_pdf_path" in locals():
                    try:
                        for tmp_path in processed_files:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                    except Exception as e:
                        st.warning(f"Error cleaning up temporary files: {e}")
else:
    st.info("Please enter your Groq API Key in the sidebar to continue.")
