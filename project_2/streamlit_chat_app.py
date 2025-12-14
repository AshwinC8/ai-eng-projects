import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

# Set page configuration
st.set_page_config(
    page_title="Everstorm Outfitters Support",
    page_icon="💬",
    layout="centered"
)

# System prompt for the RAG chatbot
SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot** for Everstorm Outfitters. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with "I'm not sure from the docs."

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) When possible, cite sources as [source: source] using the metadata.

CONTEXT:
{context}

USER:
{question}
"""

@st.cache_resource
def load_rag_chain():
    """Load the RAG chain components (cached for performance)"""
    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

    # Load FAISS vector store
    vector_store = FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    # Create prompt template
    prompt = PromptTemplate(
        template=SYSTEM_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Initialize LLM
    llm = OllamaLLM(model="gemma3:1b", temperature=0.1)

    # Build RAG chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain

# Initialize chat history and chat_history for the chain
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display title
st.title("💬 Everstorm Outfitters Support")
st.markdown("Ask questions about our policies, shipping, returns, and more!")

# Load RAG chain
try:
    chain = load_rag_chain()
    chain_loaded = True
except Exception as e:
    st.error(f"Error loading RAG chain: {e}")
    st.info("Make sure Ollama is running with 'ollama serve' and the gemma3:1b model is installed.")
    chain_loaded = False

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your support question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response using RAG chain
    if chain_loaded:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Query the RAG chain
                    result = chain.invoke({
                        "question": prompt,
                        "chat_history": st.session_state.chat_history
                    })

                    response = result["answer"]

                    # Update chat history for the chain
                    st.session_state.chat_history.append((prompt, response))

                    # Display the response
                    st.markdown(response)

                    # Optionally display source documents
                    if result.get("source_documents"):
                        with st.expander("View Sources"):
                            for i, doc in enumerate(result["source_documents"][:3]):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(doc.page_content[:200] + "...")
                                st.markdown("---")

                except Exception as e:
                    response = f"Sorry, I encountered an error: {str(e)}"
                    st.error(response)
    else:
        response = "RAG chain is not loaded. Please check the error message above."
        st.error(response)

    # Add assistant response to message history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a sidebar with options
with st.sidebar:
    st.header("Options")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions about Everstorm Outfitters policies.")
    st.markdown("**Powered by:**")
    st.markdown("- FAISS vector store")
    st.markdown("- HuggingFace embeddings (gte-small)")
    st.markdown("- Gemma3:1b via Ollama")

    st.markdown("---")
    st.markdown("### Status")
    if chain_loaded:
        st.success("✅ RAG chain loaded")
    else:
        st.error("❌ RAG chain not loaded")
