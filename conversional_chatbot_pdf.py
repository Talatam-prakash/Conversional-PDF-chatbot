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
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")
st.title("Conversational PDF Chatbot")

# sidebar for api key and LLM Models
with st.sidebar:
    st.header("Authentication")
    api_key = st.text_input("Enter your Groq API Key:", type="password")
    session_id = st.text_input("Session ID", value="Default Session")
    model_name = st.selectbox(
    "Select Groq Model",
    ["llama3-8b-8192", "llama3-70b-8192", "Gemma2-9b-It"]
)


if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name=model_name)
    uploaded_file = st.file_uploader(" Upload a PDF File", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_pdf_path = tmp_file.name

        # load the PDF content using pdf loader
        loader = PyPDFLoader(tmp_pdf_path)
        documents = loader.load()

        # chunking the context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        splits = text_splitter.split_documents(documents)

        # embedding the context and store in vector database
        @st.cache_resource
        def get_vectorstore(_splits):
            return Chroma.from_documents(
                documents=_splits,
                embedding=embeddings,
                client_settings=Settings(anonymized_telemetry=False)
            )

        vectorstore = get_vectorstore(splits)

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
                You are an assistant for a question-answering task.
                Use the following context to answer the question:
                {context}
                If the question is not answerable based on the context, respond:
                'Your query is not related to the PDF context.'
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

        # Clean up temporary file
        os.remove(tmp_pdf_path)
else:
    st.info("Please enter your Groq API Key in the sidebar to continue.")
