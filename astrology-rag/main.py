from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# 1. Load the PDF
print("Loading PDF...")
loader = PyPDFLoader("astrology.pdf")
document = loader.load()

# 2. Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(document)

# 3. Initialize Embedding Model
print("Initializing Hugging Face embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create Vector Store
print("Creating ChromaDB vector store...")
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

# 5. Semantic Search (Retrieval)
query = "What does Surya (Sun) represent? Karakas, soul, and significations."
print(f"\nSearching for chunks matching: '{query}'")
retrieved_docs = vector_store.similarity_search(query, k=5)

# Combine the retrieved chunks into a single text block
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# 6. Generation (Connecting to Ollama)
print("\nConnecting to local Qwen model via Ollama...")
# Note: Ensure the model name exactly matches what you installed in Ollama (e.g., "qwen", "qwen:7b", or "qwen2.5")
llm = OllamaLLM(model="qwen2.5-coder:7b")



# Create the strict instructions for the AI
prompt = PromptTemplate.from_template("""
You are an expert in Vedic astrology. Answer the user's question using ONLY the provided context below.
If the answer is not in the context, just say "I don't know based on the provided text." Do not make up information.

Context:
{context}

Question: {question}

Answer:
""")

# Format the prompt with our specific context and question
formatted_prompt = prompt.format(context=context_text, question=query)

print("Generating answer (this might take a few seconds on your M1 Max)...")
answer = llm.invoke(formatted_prompt)

print("\n=== 🤖 FINAL RAG ANSWER ===")
print(answer)
