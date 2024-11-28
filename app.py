
import os
import torch
import streamlit as st
import logging
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device to "cuda" if available, else fallback to "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Selected device: {device}")

# Set PyTorch CUDA memory management environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# URLs for company information
urls = {
    "Exl Services": "https://www.exlservice.com/about-exl",
    "Genpact": "https://www.genpact.com/about-us",
    "Wipro": "https://www.wipro.com/about-us/",
    "Cognizant": "https://www.cognizant.com/us/en/about-cognizant"
}

# Check if FAISS vector store exists
faiss_store_path = "./faiss_store"
if not os.path.exists(faiss_store_path):
    # Load documents from URLs
    docs = [WebBaseLoader(url).load() for url in urls.values()]
    documents = [item for sublist in docs for item in sublist]

    # Split documents for better context management
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    # Initialize embeddings with dynamic device handling
    model_path = "./models/sentence-transformers/all-mpnet-base-v2"
    if device.type == "cuda":
        logging.info("Using GPU for embeddings")
    else:
        logging.info("Using CPU for embeddings")
    embeddings = HuggingFaceEmbeddings(model_name=model_path)

    # Create and save FAISS vector store
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(faiss_store_path)
    logging.info("FAISS store created and saved.")
else:
    # Load FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="./models/sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.load_local(faiss_store_path, embeddings, allow_dangerous_deserialization=True)
    logging.warning("Ensure that the FAISS store was created by a trusted source before enabling deserialization.")
    logging.info("FAISS store loaded from local storage.")

retriever = vector_store.as_retriever()

# Initialize ChatGroq
llm = ChatGroq(temperature=0.2, verbose=True)

# Prompts
ci_prompt = PromptTemplate(
    template="""
    You are a competitive intelligence assistant. Your task is to analyze and compare two companies: EXL Services and the given competitor.
    Use the provided documents to perform the comparison and extract actionable insights.
    Highlight key trends, competitor strategies, opportunities, and threats in three concise sentences.

    Competitor Company: {input}
    <context>
    {context}
    </context>

    Answer
    """
)

swot_prompt = PromptTemplate(
    template="""
    You are a SWOT analysis expert. Your task is to analyze the strengths, weaknesses, opportunities, and threats for the given company based on the provided documents.

    Company: {input}
    <context>
    {context}
    </context>

    Answer
    """
)

evaluation_prompt = PromptTemplate(
    template="""
    You are an evaluator. Your task is to assess the given result against the provided context based on the following criteria:

    1. Correctness: Is the result factually accurate and aligned with the context?
    2. Truthfulness: Does the result avoid introducing any false or misleading information?
    3. Helpfulness: Is the result clear, actionable, and relevant to the context?

    Based on the context below:
    {context}

    Evaluate the following result:
    {result}

    Provide a score (1-10) for each criterion along with a brief explanation for your assessment.
    """
)

# Create chains for Competitive Intelligence, SWOT, and Evaluation
ci_chain = create_stuff_documents_chain(llm, ci_prompt)
swot_chain = create_stuff_documents_chain(llm, swot_prompt)
evaluation_chain = create_stuff_documents_chain(llm, evaluation_prompt)

# Create retrieval chains for each type
ci_retrieval_chain = create_retrieval_chain(retriever, ci_chain)
swot_retrieval_chain = create_retrieval_chain(retriever, swot_chain)
evaluation_retrieval_chain = create_retrieval_chain(retriever, evaluation_chain)

# Streamlit app
st.title("Company Analysis App")
st.write("Perform Competitive Intelligence or SWOT analysis for various companies.")

# User inputs
company_name = st.selectbox("Select the company for analysis:", options=list(urls.keys()))
analysis_type = st.radio("Choose the type of analysis:", ("Competitive Intelligence", "SWOT"))

# Validate user input
if analysis_type.lower() not in ["competitive intelligence", "swot"]:
    st.error("Invalid analysis type selected. Please choose 'Competitive Intelligence' or 'SWOT'.")
    st.stop()

def truncate_context(context, limit=4000):
    """Truncate context to fit within token limits."""
    tokens = context.split()
    if len(tokens) > limit:
        truncated_context = " ".join(tokens[:limit])
        logging.warning("Context truncated to fit token limit.")
        return truncated_context
    return context

if st.button("Analyze"):
    try:
        # Construct input for the retrieval chain
        retrieval_input = {"input": company_name}

        if analysis_type.lower() == "competitive intelligence":
            response = ci_retrieval_chain.invoke(retrieval_input)
        elif analysis_type.lower() == "swot":
            response = swot_retrieval_chain.invoke(retrieval_input)

        # Get the answer from the chain
        analysis_result = response["answer"]
        st.subheader(f"{analysis_type} Analysis")
        st.write(analysis_result)

        # Store the analysis result for evaluation
        st.session_state["analysis_result"] = analysis_result
        st.session_state["context"] = response.get("context", "")

    except Exception as e:
        logging.error(f"Error during execution: {e}")
        st.error(f"An error occurred: {e}")

# Perform evaluation on the analysis result
if "analysis_result" in st.session_state and st.button("Evaluate"):
    try:
        evaluation_input = {
            "context": truncate_context(st.session_state["context"], limit=4000),
            "result": st.session_state["analysis_result"],
            "input": company_name
        }
        evaluation_response = evaluation_retrieval_chain.invoke(evaluation_input)

        st.subheader("Evaluation")
        st.write(evaluation_response["answer"])

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        st.error(f"An error occurred during evaluation: {e}")
