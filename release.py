import os
import tempfile
from datetime import datetime
from typing import List, Union
import streamlit as st
import bs4
from agno.agent import Agent
from agno.models.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus.vectorstores import Milvus  # 使用正确的导入路径
from langchain_core.embeddings import Embeddings
from agno.tools.exa import ExaTools
from agno.embedder.ollama import OllamaEmbedder as AgnoOllamaEmbedder # Renamed to avoid confusion

# --- Add Logging ---
import logging
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for more verbosity if needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Output to console
    ]
)
logger = logging.getLogger(__name__)
# You can also set agno's logger level if it uses Python's logging
# logging.getLogger("agno").setLevel(logging.DEBUG) # Uncomment if agno has its own logger

# --- End Logging ---


class OllamaEmbedderr(Embeddings):
    def __init__(self, model_name="snowflake-arctic-embed", base_url="http://localhost:11434"):
        """
        Initialize the OllamaEmbedderr with a specific model.

        Args:
            model_name (str): The name of the model to use for embedding.
            base_url (str): The base URL for the Ollama API server.
        """
        self.model_name = model_name
        self.base_url = base_url
        logger.info(f"Initializing OllamaEmbedderr with model: {self.model_name} at {self.base_url}")
        try:
            # Verify model availability with Ollama (optional, but good for debugging)
            # Note: This requires the 'ollama' Python package, not just agno
            # import ollama as official_ollama_client
            # official_ollama_client.list() # This would show available models
            self.embedder = AgnoOllamaEmbedder(id=model_name, dimensions=1024, host=base_url)
            logger.info(f"AgnoOllamaEmbedder initialized successfully for model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize AgnoOllamaEmbedder for model {self.model_name}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Embedding {len(texts)} documents with model {self.model_name}")
        embeddings = []
        for i, text in enumerate(texts):
            try:
                # logger.debug(f"Embedding document {i+1}/{len(texts)}: '{text[:100]}...'")
                embeddings.append(self.embed_query(text))
            except Exception as e:
                logger.error(f"Error embedding document {i+1} (length: {len(text)}). Text snippet: '{text[:100]}...'", exc_info=True)
                # 当嵌入失败时，使用零向量代替，避免整个过程失败
                logger.warning(f"Using zero vector as fallback for document {i+1}")
                embeddings.append([0.0] * 1024)  # 使用1024维的零向量
        logger.info(f"Successfully embedded {len(embeddings)} documents.")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        logger.info(f"Embedding query (length: {len(text)}) with model {self.model_name}. Query snippet: '{text[:100]}...'")
        if not text or not text.strip(): # Check for empty or whitespace-only text
            logger.warning(f"Attempting to embed an empty or whitespace-only text string with model {self.model_name}. Length: {len(text)}. Content: '{text}'. Returning zero vector as fallback.")
            return [0.0] * 1024  # Assuming 1024 dimensions as per AgnoOllamaEmbedder init in __init__

        # For more detailed debugging, uncomment the line below to see more of the text
        # logger.debug(f"Full text for embedding with {self.model_name} (first 500 chars): '{text[:500]}'")
        try:
            embedding = self.embedder.get_embedding(text)
            if not embedding or not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
                logger.error(f"Received invalid embedding from agno embedder for text '{text[:100]}...'. Embedding type: {type(embedding)}, Value: {str(embedding)[:200]}")
                logger.warning(f"Using zero vector as fallback due to invalid embedding received for text: '{text[:100]}...'" )
                return [0.0] * 1024  # Fallback for invalid embedding format
            # logger.debug(f"Embedding successful for query: '{text[:100]}...'. Dimensions: {len(embedding)}")
            return embedding
        except Exception as e:
            # This is where the 400 error from Ollama (via agno) is likely to be caught
            logger.error(f"Failed to get embedding from AgnoOllamaEmbedder for text (length: {len(text)}): '{text[:100]}...'", exc_info=True)
            # AT THIS POINT, CHECK YOUR OLLAMA SERVER LOGS FOR THE DETAILED ERROR MESSAGE FROM OLLAMA ITSELF
            # The exception will be caught by embed_documents and handled with a zero vector.
            raise  # Re-raise the exception so it propagates up to embed_documents


# Constants
COLLECTION_NAME = "test_qwen_test"


# Streamlit App Initialization
st.title("🐋 Qwen 3 Local RAG Reasoning Agent")

# --- Add Model Info Boxes ---
st.info("**Qwen3:** The latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models.")
st.info("**Gemma 3:** These models are multimodal—processing text and images—and feature a 128K context window with support for over 140 languages.")
# -------------------------

# Session State Initialization
if 'model_version' not in st.session_state:
    st.session_state.model_version = "qwen3:1.7b"
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'exa_api_key' not in st.session_state:
    st.session_state.exa_api_key = ""
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False
if 'force_web_search' not in st.session_state:
    st.session_state.force_web_search = False
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.7
if 'rag_enabled' not in st.session_state:
    st.session_state.rag_enabled = True


# Sidebar Configuration
st.sidebar.header("⚙️ Settings")

# Model Selection
st.sidebar.header("🧠 Model Choice")
model_help = """
- qwen3:1.7b: Lighter model (MoE)
- gemma3:1b: More capable but requires better GPU/RAM(32k context window)
- gemma3:4b: More capable and MultiModal (Vision)(128k context window)
- deepseek-r1:1.5b
- qwen3:8b: More capable but requires better GPU/RAM

Choose based on your hardware capabilities.
"""
st.session_state.model_version = st.sidebar.radio(
    "Select Model Version",
    options=["qwen3:1.7b", "gemma3:1b", "gemma3:4b", "deepseek-r1:1.5b", "qwen3:8b"],
    help=model_help
)

st.sidebar.info("Run ollama pull qwen3:1.7b")
st.sidebar.info("Also run: ollama pull snowflake-arctic-embed:latest (for embeddings)") # Added this hint
st.sidebar.info("确保Ollama服务运行在 http://localhost:11434 (默认地址)")

# RAG Mode Toggle
st.sidebar.header("📚 RAG Mode")
st.session_state.rag_enabled = st.sidebar.toggle("Enable RAG", value=st.session_state.rag_enabled)

# Clear Chat Button
if st.sidebar.button("✨ Clear Chat"):
    st.session_state.history = []
    st.rerun()

# Show API Configuration only if RAG is enabled
if st.session_state.rag_enabled:
    st.sidebar.header("🔬 Search Tuning")
    st.session_state.similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Lower values will return more documents but might be less relevant. Higher values are more strict."
    )

# Add in the sidebar configuration section, after the existing API inputs
st.sidebar.header("🌍 Web Search")
st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback", value=st.session_state.use_web_search)

if st.session_state.use_web_search:
    exa_api_key = st.sidebar.text_input(
        "Exa AI API Key",
        type="password",
        value=st.session_state.exa_api_key,
        help="Required for web search fallback when no relevant documents are found"
    )
    st.session_state.exa_api_key = exa_api_key

    default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
    custom_domains = st.sidebar.text_input(
        "Custom domains (comma-separated)",
        value=",".join(default_domains),
        help="Enter domains to search from, e.g.: arxiv.org,wikipedia.org"
    )
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

# Document Processing Functions
def process_pdf(file) -> List:
    logger.info(f"Processing PDF file: {file.name}")
    try:
        file_size_mb = len(file.getvalue()) / (1024 * 1024)
        if file_size_mb > 100:
            logger.warning(f"📄 File too large ({file_size_mb:.2f}MB): {file.name}")
            st.error(f"📄 文件过大 ({file_size_mb:.2f}MB)，请上传小于100MB的PDF文件")
            return []

        # 记录文件信息以便调试
        logger.info(f"PDF file info - Name: {file.name}, Size: {file_size_mb:.2f}MB, Type: {file.type if hasattr(file, 'type') else 'unknown'}")
        
        try:
            # 使用Flask中成功的文件处理方式
            temp_path = None
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                # 保存文件内容
                tmp.write(file.getvalue())
                temp_path = tmp.name
            
            logger.info(f"PDF '{file.name}' saved to temporary file: {temp_path}")
            # 使用临时文件路径加载PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            # 处理完成后删除临时文件
            os.unlink(temp_path)
            logger.info(f"Temporary file {temp_path} deleted.")
        except Exception as inner_e:
            logger.error(f"Error during file handling or PDF loading: {str(inner_e)}", exc_info=True)
            st.error(f"📄 文件处理错误: {str(inner_e)}")
            return []

        if not documents:
            logger.warning(f"⚠️ PDF file {file.name} did not yield any documents.")
            st.warning(f"⚠️ PDF文件 {file.name} 未包含可提取的文本内容")
            return []

        for doc in documents:
            doc.metadata = {
                "source_type": "pdf",
                "file_name": file.name,
                "timestamp": datetime.now().isoformat()
            }

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        if not split_docs:
            logger.warning(f"⚠️ PDF file {file.name} yielded no split documents.")
            st.warning(f"⚠️ PDF文件 {file.name} 分割后未产生有效文档片段")
        else:
            logger.info(f"✅ Successfully processed PDF {file.name}, {len(split_docs)} chunks.")
            st.success(f"✅ 成功处理PDF文件 {file.name}，生成了 {len(split_docs)} 个文档片段")
        return split_docs

    except Exception as e:
        logger.error(f"📄 PDF processing error for {file.name}: {str(e)}", exc_info=True)
        st.error(f"📄 PDF加载或处理错误: {str(e)}")
        return []


def process_web(url: str) -> List:
    logger.info(f"Processing URL: {url}")
    st.info(f"🔄 开始处理URL: {url}")
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header", "content", "main")))
        )
        documents = loader.load()

        if not documents:
            logger.warning(f"⚠️ URL {url} did not yield any documents.")
            st.warning(f"⚠️ URL {url} 未包含可提取的文本内容")
            return []

        for doc in documents:
            doc.metadata = {
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            }
            if len(doc.page_content) > 65000:
                logger.warning(f"⚠️ URL content from {url} too long, truncating to 65000 chars.")
                doc.page_content = doc.page_content[:65000]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        if not split_docs:
            logger.warning(f"⚠️ URL {url} yielded no split documents.")
            st.warning(f"⚠️ URL {url} 分割后未产生有效文档片段")
        else:
            logger.info(f"✅ Successfully processed URL {url}, {len(split_docs)} chunks.")
            st.success(f"✅ 成功处理URL {url}，生成了 {len(split_docs)} 个文档片段")
        return split_docs

    except Exception as e:
        logger.error(f"🌐 URL processing error for {url}", exc_info=True)
        st.error(f"🌐 URL加载或处理错误: {str(e)}")
        return []


# Vector Store Management
def create_vector_store(client, texts):
    logger.info(f"Attempting to create/update vector store '{COLLECTION_NAME}' with {len(texts)} documents.")
    try:
        # 只用langchain_milvus，不再用pymilvus
        try:
            logger.info("Initializing embedding function...")
            embedding_function = OllamaEmbedderr(base_url="http://localhost:11434") # 显式指定Ollama API服务器地址
            logger.info("Embedding function initialized successfully.")
        except Exception as emb_e:
            logger.error(f"Failed to initialize embedding function: {str(emb_e)}", exc_info=True)
            st.error(f"嵌入模型初始化失败: {str(emb_e)}")
            return None
        
        try:
            logger.info("Initializing Milvus vector store...")
            vector_store = Milvus(
                embedding_function=embedding_function,
                collection_name=COLLECTION_NAME,
                connection_args={
                    "uri": "tcp://192.168.7.147:19530"
                },
                auto_id=True,
            )
            logger.info("Milvus Langchain vector store object initialized.")
        except Exception as vs_e:
            logger.error(f"Failed to initialize vector store: {str(vs_e)}", exc_info=True)
            st.error(f"向量存储初始化失败: {str(vs_e)}")
            return None
        
        with st.spinner('📤 上传文档到Milvus...'):
            try:
                for i, text in enumerate(texts[:3]):
                    logger.info(f"Sample document {i+1} - Length: {len(text.page_content)}, Metadata: {text.metadata}")
                logger.info(f"Adding {len(texts)} documents to Milvus collection '{COLLECTION_NAME}'.")
                vector_store.add_documents(texts)
                st.success("✅ 文档存储成功!")
                logger.info("✅ Documents added to Milvus successfully.")
                return vector_store
            except Exception as add_e:
                logger.error(f"Error adding documents to vector store: {str(add_e)}", exc_info=True)
                st.error(f"文档添加失败: {str(add_e)}")
                return None
    except Exception as e:
        logger.error(f"🔴 Vector store creation/update error: {str(e)}", exc_info=True)
        st.error(f"🔴 向量存储错误: {str(e)}")
        return None


# ... (rest of your Streamlit app code, get_web_search_agent, get_rag_agent, check_document_relevance, etc.)
# Ensure all st.error calls also have a logger.error(..., exc_info=True) counterpart if they represent actual errors.

def get_web_search_agent() -> Agent:
    """Initialize a web search agent."""
    logger.info("Initializing Web Search Agent")
    return Agent(
        name="Web Search Agent",
        model=Ollama(id="llama3.2"), # Consider making this model configurable
        tools=[ExaTools(
            api_key=st.session_state.exa_api_key,
            include_domains=search_domains if 'search_domains' in globals() else [], # Ensure search_domains is defined
            num_results=5
        )],
        instructions="""You are a web search expert. Your task is to:
        1. Search the web for relevant information about the query
        2. Compile and summarize the most relevant information
        3. Include sources in your response
        """,
        show_tool_calls=True,
        markdown=True,
    )


def get_rag_agent() -> Agent:
    """Initialize the main RAG agent."""
    logger.info(f"Initializing RAG Agent with model: {st.session_state.model_version}")
    return Agent(
        name="Qwen 3 RAG Agent",
        model=Ollama(id=st.session_state.model_version),
        instructions="""You are an Intelligent Agent specializing in providing accurate answers.

        When asked a question:
        - Analyze the question and answer the question with what you know.
        
        When given context from documents:
        - Focus on information from the provided documents
        - Be precise and cite specific details
        
        When given web search results:
        - Clearly indicate that the information comes from web search
        - Synthesize the information clearly
        
        Always maintain high accuracy and clarity in your responses.
        """,
        show_tool_calls=True,
        markdown=True,
    )


def check_document_relevance(query: str, vector_store, threshold: float = 0.7) -> tuple[bool, List]:
    if not vector_store:
        logger.warning("check_document_relevance called with no vector_store.")
        return False, []
    logger.info(f"Checking document relevance for query: '{query[:100]}...' with threshold: {threshold}")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold}
    )
    try:
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents for query '{query[:100]}...'")
        return bool(docs), docs
    except Exception as e:
        logger.error(f"Error during document retrieval for query '{query[:100]}...'", exc_info=True)
        st.error(f"Error searching documents: {e}")
        return False, []


chat_col, toggle_col = st.columns([0.9, 0.1])

with chat_col:
    prompt = st.chat_input("Ask about your documents..." if st.session_state.rag_enabled else "Ask me anything...")

with toggle_col:
    st.session_state.force_web_search = st.toggle('🌐', help="Force web search")

# Main application logic
if st.session_state.rag_enabled:
    with st.expander("📁 Upload Documents or URLs for RAG", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            accept_multiple_files=True, 
            type='pdf'
        )
        url_input = st.text_input("Enter URL to scrape")

        if uploaded_files:
            # st.write(f"Processing {len(uploaded_files)} PDF file(s)...") # Less verbose
            all_texts = []
            for file in uploaded_files:
                if file.name not in st.session_state.processed_documents:
                    with st.spinner(f"Processing {file.name}... "): 
                        logger.info(f"Starting PDF processing for: {file.name}")
                        texts = process_pdf(file) # Already logs inside
                        if texts: 
                            logger.info(f"Extracted {len(texts)} text chunks from {file.name}")
                            # st.info(f"📊 从 {file.name} 中提取了 {len(texts)} 个文本片段") # Already in process_pdf
                            all_texts.extend(texts)
                            st.session_state.processed_documents.append(file.name)
                        else:
                            logger.warning(f"No text extracted from {file.name}")
                            # st.warning(f"⚠️ 无法从 {file.name} 中提取文本，请检查PDF文件是否有效") # Already in process_pdf
                else:
                    st.write(f"📄 {file.name} already processed.")
            
            if all_texts:
                with st.spinner("Creating vector store..."):
                    logger.info(f"Creating vector store with {len(all_texts)} total text chunks.")
                    st.session_state.vector_store = create_vector_store(None, all_texts) # Already logs inside
                    if not st.session_state.vector_store:
                        st.error("❌ 创建向量存储失败，请查看上方错误信息") # Already in create_vector_store
            elif uploaded_files: # if files were uploaded but all_texts is empty
                 st.warning("No processable text found in the uploaded PDF(s).")


        if url_input:
            if url_input not in st.session_state.processed_documents:
                with st.spinner(f"Scraping and processing {url_input}..."):
                    logger.info(f"Starting URL processing for: {url_input}")
                    texts = process_web(url_input) # Already logs inside
                    if texts:
                        logger.info(f"Creating vector store with {len(texts)} text chunks from URL.")
                        st.session_state.vector_store = create_vector_store(None, texts) # Already logs inside
                        st.session_state.processed_documents.append(url_input)
                    else:
                        logger.warning(f"No text extracted from URL: {url_input}")
            else:
                st.write(f"🔗 {url_input} already processed.")
            
        if st.session_state.vector_store:
            st.success("Vector store is ready.")
        elif not uploaded_files and not url_input:
             st.info("Upload PDFs or enter a URL to populate the vector store.")

    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Sources")
        for source in st.session_state.processed_documents:
            if source.endswith('.pdf'):
                st.sidebar.text(f"📄 {source}")
            else:
                st.sidebar.text(f"🌐 {source}")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    logger.info(f"User prompt: {prompt}")

    if st.session_state.rag_enabled:
        with st.spinner("🤔Evaluating the Query..."):
            rewritten_query = prompt # Simplified, assuming no rewrite for now
            # logger.info(f"Rewritten query: {rewritten_query}") # If you add query rewriting

        context = ""
        docs = []
        used_web_search = False

        if not st.session_state.force_web_search and st.session_state.vector_store:
            logger.info("Attempting document search from vector store.")
            relevant, docs = check_document_relevance(rewritten_query, st.session_state.vector_store, st.session_state.similarity_threshold)
            if relevant:
                context = "\n\n".join([d.page_content for d in docs])
                st.info(f"📊 Found {len(docs)} relevant documents (similarity > {st.session_state.similarity_threshold})")
                logger.info(f"Found {len(docs)} relevant documents from vector store.")
            elif st.session_state.use_web_search:
                st.info("🔄 No relevant documents found in database, falling back to web search...")
                logger.info("No relevant documents in vector store, will try web search if enabled.")
        elif st.session_state.force_web_search:
            logger.info("Web search is forced by toggle.")
        elif not st.session_state.vector_store:
            logger.info("No vector store available to search.")


        if (st.session_state.force_web_search or (not context and st.session_state.use_web_search)) and st.session_state.exa_api_key:
            with st.spinner("🔍 Searching the web..."):
                logger.info("Attempting web search.")
                used_web_search = True
                try:
                    web_search_agent = get_web_search_agent()
                    web_results = web_search_agent.run(rewritten_query).content
                    if web_results:
                        context = f"Web Search Results:\n{web_results}"
                        if st.session_state.force_web_search:
                            st.info("ℹ️ Using web search as requested via toggle.")
                        else: # Fallback case
                            st.info("ℹ️ Using web search as fallback since no relevant documents were found.")
                        logger.info("Web search successful.")
                    else:
                        logger.warning("Web search returned no results.")
                except Exception as e:
                    st.error(f"❌ Web search error: {str(e)}")
                    logger.error("Web search error", exc_info=True)

        with st.spinner("🤖 Thinking..."):
            try:
                rag_agent = get_rag_agent()
                
                if context:
                    full_prompt = f"""Context: {context}

Original Question: {prompt}
Please provide a comprehensive answer based on the available information."""
                else:
                    full_prompt = f"Original Question: {prompt}\n"
                    st.info("ℹ️ No relevant information found in documents or web search.")
                    logger.info("No context available from RAG or web search for the agent.")

                response = rag_agent.run(full_prompt)
                st.session_state.history.append({"role": "assistant", "content": response.content})
                
                with st.chat_message("assistant"):
                    st.write(response.content)
                    if not used_web_search and docs: # Show doc sources only if not from web search and docs exist
                        with st.expander("🔍 See document sources"):
                            for i, doc in enumerate(docs, 1):
                                source_type = doc.metadata.get("source_type", "unknown")
                                source_icon = "📄" if source_type == "pdf" else "🌐"
                                source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url", "unknown")
                                st.write(f"{source_icon} Source {i} from {source_name}:")
                                st.write(f"{doc.page_content[:200]}...")
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                logger.error("Error generating RAG agent response", exc_info=True)

    else: # Simple mode (no RAG)
        with st.spinner("🤖 Thinking..."):
            try:
                agent = get_rag_agent() # Using rag_agent setup even for non-RAG for consistency
                context_for_non_rag = ""

                if st.session_state.force_web_search and st.session_state.use_web_search and st.session_state.exa_api_key:
                    logger.info("Non-RAG mode: Web search forced.")
                    with st.spinner("🔍 Searching the web..."):
                        try:
                            web_search_agent = get_web_search_agent()
                            web_results = web_search_agent.run(prompt).content
                            if web_results:
                                context_for_non_rag = f"Web Search Results:\n{web_results}"
                                st.info("ℹ️ Using web search as requested.")
                                logger.info("Non-RAG mode: Web search successful.")
                        except Exception as e:
                            st.error(f"❌ Web search error: {str(e)}")
                            logger.error("Non-RAG mode: Web search error", exc_info=True)
                
                if context_for_non_rag:
                    full_prompt = f"""Context: {context_for_non_rag}

Question: {prompt}
Please provide a comprehensive answer based on the available information."""
                else:
                    full_prompt = prompt
                    logger.info("Non-RAG mode: No web search context.")


                response = agent.run(full_prompt)
                response_content = response.content
                
                import re
                think_pattern = r'<think>(.*?)</think>'
                think_match = re.search(think_pattern, response_content, re.DOTALL)
                
                if think_match:
                    thinking_process = think_match.group(1).strip()
                    final_response = re.sub(think_pattern, '', response_content, flags=re.DOTALL).strip()
                else:
                    thinking_process = None
                    final_response = response_content
                
                st.session_state.history.append({"role": "assistant", "content": final_response})
                
                with st.chat_message("assistant"):
                    if thinking_process:
                        with st.expander("🤔 See thinking process"):
                            st.markdown(thinking_process)
                    st.markdown(final_response)
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                logger.error("Error generating non-RAG agent response", exc_info=True)

elif not st.session_state.history: # Only show if no prompt and history is empty
    st.warning("You can directly talk to qwen and gemma models locally! Toggle the RAG mode to upload documents!")