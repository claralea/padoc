# modules/cli.py
import sys
import os

# CRITICAL: Apply SQLite fix first before ANY other imports
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass  # Will fail later with ChromaDB if SQLite is too old

# Now safe to import other standard modules
import argparse
import pandas as pd
import json
import time
import glob
import hashlib

# Import ChromaDB AFTER the SQLite fix
try:
    import chromadb
    from chromadb import PersistentClient
    CHROMADB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ChromaDB not available: {e}")
    CHROMADB_AVAILABLE = False
    PersistentClient = None

# Vertex AI imports
try:
    import vertexai
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part, ToolConfig
    from google.api_core.exceptions import InternalServerError, ServiceUnavailable, ResourceExhausted
    VERTEX_AI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vertex AI not available: {e}")
    VERTEX_AI_AVAILABLE = False

# Langchain imports
try:
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Langchain not available: {e}")
    LANGCHAIN_AVAILABLE = False

# Optional imports
try:
    from semantic_splitter import SemanticChunker
except ImportError:
    SemanticChunker = None

try:
    import agent_tools
except ImportError:
    agent_tools = None

# Setup - Handle missing environment variables gracefully
GCP_PROJECT = os.environ.get("GCP_PROJECT", "rag-test-467013")
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-2.5-flash"
INPUT_FOLDER = "input-datasets"
OUTPUT_FOLDER = "outputs"
CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000

# Initialize Vertex AI only if available and GCP_PROJECT is set
embedding_model = None
generative_model = None

if VERTEX_AI_AVAILABLE:
    try:
        vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        print(f"✅ Vertex AI initialized with project {GCP_PROJECT}")
    except Exception as e:
        print(f"Warning: Could not initialize Vertex AI: {e}")

# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.25,
    "top_p": 0.95,
}

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in pharmaceutical manufacturing. You help with interpreting standard operating procedures (SOPs), deviation handling policies, and documentation workflows. You ONLY rely on the provided document chunks.

When answering:
- Be precise and regulation-aware.
- If information is missing, say so clearly.
- Avoid hallucinating or making assumptions.

Example queries:
- "How should I document a temperature deviation?"
- "What are the cleaning requirements after a batch spill?"

Stay factual and compliant.
"""

if VERTEX_AI_AVAILABLE and embedding_model is not None:
    try:
        generative_model = GenerativeModel(
            GENERATIVE_MODEL,
            system_instruction=[SYSTEM_INSTRUCTION]
        )
        print(f"✅ Generative model {GENERATIVE_MODEL} initialized")
    except Exception as e:
        print(f"Warning: Could not initialize GenerativeModel: {e}")

book_mappings = {
    "21_CFR_Part_211-cleaned": {"author":"eCFR", "year": 2025},
    "Q7-Good-Manufacturing-Practice-Guidance-for-Active-Pharmaceutical-Ingredients-Guidance-for-Industry-cleaned":{"author": "FDA", "year": 2016}
}


def generate_query_embedding(query):
    if not embedding_model:
        raise RuntimeError("Embedding model not initialized")
    query_embedding_inputs = [TextEmbeddingInput(task_type='RETRIEVAL_DOCUMENT', text=query)]
    kwargs = dict(output_dimensionality=EMBEDDING_DIMENSION) if EMBEDDING_DIMENSION else {}
    embeddings = embedding_model.get_embeddings(query_embedding_inputs, **kwargs)
    return embeddings[0].values


def generate_text_embeddings(chunks, dimensionality: int = 256, batch_size=250, max_retries=5, retry_delay=5):
    if not embedding_model:
        raise RuntimeError("Embedding model not initialized")
    # Max batch size is 250 for Vertex AI
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in batch]
        kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}

        # Retry logic with exponential backoff
        retry_count = 0
        while retry_count <= max_retries:
            try:
                embeddings = embedding_model.get_embeddings(inputs, **kwargs)
                all_embeddings.extend([embedding.values for embedding in embeddings])
                break
            except (InternalServerError, ServiceUnavailable, ResourceExhausted) as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"Failed to generate embeddings after {max_retries} attempts. Last error: {str(e)}")
                    raise

                # Calculate delay
                wait_time = retry_delay * (2 ** (retry_count - 1))
                print(f"API error: {str(e)}. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})...")
                time.sleep(wait_time)
        
    return all_embeddings


def load_text_embeddings(df, collection, batch_size=500):
    # Generate ids
    df["id"] = df.index.astype(str)
    hashed_books = df["book"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
    df["id"] = hashed_books + "-" + df["id"]

    metadata = {
        "book": df["book"].tolist()[0]
    }
    if metadata["book"] in book_mappings:
        book_mapping = book_mappings[metadata["book"]]
        metadata["author"] = book_mapping["author"]
        metadata["year"] = book_mapping["year"]
   
    # Process data in batches
    total_inserted = 0
    for i in range(0, df.shape[0], batch_size):
        # Create a copy of the batch and reset the index
        batch = df.iloc[i:i+batch_size].copy().reset_index(drop=True)

        ids = batch["id"].tolist()
        documents = batch["chunk"].tolist() 
        metadatas = []
        for book in batch["book"]:
            entry = {"book": book}
            if book in book_mappings:
                entry["author"] = book_mappings[book]["author"]
                entry["year"] = book_mappings[book]["year"]
            metadatas.append(entry)
        embeddings = batch["embedding"].tolist()

        try:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        except Exception as e:
            print("❌ Error during collection.add:", str(e))
            print("Batch size:", len(ids))
            print("First doc:", documents[0][:100])
            print("First embedding length:", len(embeddings[0]))
            raise
        total_inserted += len(batch)
        print(f"Inserted {total_inserted} items...")

    print(f"Finished inserting {total_inserted} items into collection '{collection.name}'")


def chunk(method="char-split"):
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError("Langchain is required for chunking")
        
    print("chunk()")

    # Make dataset folders
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get the list of text file
    text_files = glob.glob(os.path.join(INPUT_FOLDER, "books", "*.txt"))
    print("Number of files to process:", len(text_files))

    # Process
    for text_file in text_files:
        print("Processing file:", text_file)
        filename = os.path.basename(text_file)
        book_name = filename.split(".")[0]

        with open(text_file) as f:
            input_text = f.read()
        
        text_chunks = None
        if method == "char-split":
            chunk_size = 350
            chunk_overlap = 20
            text_splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap, separator='', strip_whitespace=False)
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        elif method == "recursive-split":
            def smart_chunk(input_text, min_chunk_length=600):
                if len(input_text) <= min_chunk_length:
                    print("Short document, no chunking applied.")
                    return [input_text]
                else:
                    print("Long document, applying chunking.")
                    chunk_size = 350
                    chunk_overlap = 20
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", ".", " ", ""]
                    )
                    chunks = text_splitter.create_documents([input_text])
                    return [doc.page_content for doc in chunks]

            text_chunks = smart_chunk(input_text)
            print("Number of chunks:", len(text_chunks))
        
        elif method == "semantic-split":
            if SemanticChunker is None:
                raise RuntimeError("SemanticChunker not available")
            text_splitter = SemanticChunker(embedding_function=generate_text_embeddings)
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        if text_chunks is not None:
            data_df = pd.DataFrame(text_chunks,columns=["chunk"])
            data_df["book"] = book_name
            print("Shape:", data_df.shape)
            print(data_df.head())

            jsonl_filename = os.path.join(OUTPUT_FOLDER, f"chunks-{method}-{book_name}.jsonl")
            with open(jsonl_filename, "w") as json_file:
                json_file.write(data_df.to_json(orient='records', lines=True))


def embed(method="char-split"):
    print("embed()")

    # Get the list of chunk files
    jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"chunks-{method}-*.jsonl"))
    print("Number of files to process:", len(jsonl_files))

    # Process
    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)

        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        chunks = data_df["chunk"].values
        if method == "semantic-split":
            embeddings = generate_text_embeddings(chunks,EMBEDDING_DIMENSION, batch_size=15)
        else:
            embeddings = generate_text_embeddings(chunks,EMBEDDING_DIMENSION, batch_size=100)
        data_df["embedding"] = embeddings

        time.sleep(5)

        # Save 
        print("Shape:", data_df.shape)
        print(data_df.head())

        jsonl_filename = jsonl_file.replace("chunks-","embeddings-")
        with open(jsonl_filename, "w") as json_file:
            json_file.write(data_df.to_json(orient='records', lines=True))


def load(method="char-split"):
    if not CHROMADB_AVAILABLE:
        raise RuntimeError("ChromaDB is required for loading")
        
    print("load()")
    
    # Clear Cache
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Connect to chroma DB
    client = PersistentClient(path="./chroma") 

    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"
    print("Creating collection:", collection_name)

    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except Exception:
        print(f"Collection '{collection_name}' did not exist. Creating new.")

    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    print(f"Created new empty collection '{collection_name}'")
    print("Collection:", collection)

    # Get the list of embedding files
    jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"embeddings-{method}-*.jsonl"))
    print("Number of files to process:", len(jsonl_files))

    # Process
    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)

        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        # Load data
        load_text_embeddings(data_df, collection)

    print("Collections after load():", client.list_collections())
    print("Docs in collection:", collection.count())


def query(method="char-split"):
    if not CHROMADB_AVAILABLE:
        raise RuntimeError("ChromaDB is required for querying")
        
    print("query()")

    # Connect to Chroma DB
    client = PersistentClient(path="./chroma") 

    # Collection name
    collection_name = f"{method}-collection"

    # Load collection
    collection = client.get_collection(name=collection_name)

    # Your custom query
    user_query = "Immediate actions after Lasair particle monitoring equipment stopped recording results"

    # Embed with Vertex AI
    query_embedding_inputs = [TextEmbeddingInput(task_type='RETRIEVAL_DOCUMENT', text=user_query)]
    query_embedding = embedding_model.get_embeddings(query_embedding_inputs, output_dimensionality=EMBEDDING_DIMENSION)[0].values

    # Run query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # Show results
    print("Query:", user_query)
    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult #{i+1}")
        print("Doc:", doc)
        print("Metadata:", results["metadatas"][0][i])
        print("Distance:", results["distances"][0][i])


def chat(method="char-split"):
    if not CHROMADB_AVAILABLE or not generative_model:
        raise RuntimeError("ChromaDB and Vertex AI are required for chat")
        
    print("chat()")

    # Connect to chroma DB
    client = PersistentClient(path="./chroma") 
    collection_name = f"{method}-collection"

    query = "What happened with Lot 10000295 "

    query_embedding = generate_query_embedding(query)
    print("Query:", query)
    print("Embedding values:", query_embedding)
    
    collection = client.get_collection(name=collection_name)

    # Query based on embedding value 
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    print("\n\nResults:", results)

    print(len(results["documents"][0]))

    # FIX: Join documents outside f-string
    documents_text = "\n".join(results["documents"][0])
    INPUT_PROMPT = f"""
    {query}
    {documents_text}
    """
    print("INPUT_PROMPT: ",INPUT_PROMPT)
    response = generative_model.generate_content(
        [INPUT_PROMPT],
        generation_config=generation_config,
        stream=False,
    )
    generated_text = response.text
    print("LLM Response:", generated_text)


def get(method="char-split"):
    if not CHROMADB_AVAILABLE:
        raise RuntimeError("ChromaDB is required")
        
    print("get()")

    client = PersistentClient(path="./chroma")
    collection_name = f"{method}-collection"

    collection = client.get_collection(name=collection_name)

    # Get documents with filters
    results = collection.get(
        where={"book":"The Complete Book of Cheese"},
        limit=10
    )
    print("\n\nResults:", results)


def agent(method="char-split"):
    if not CHROMADB_AVAILABLE or not generative_model or agent_tools is None:
        raise RuntimeError("ChromaDB, Vertex AI, and agent_tools are required")
        
    print("agent()")

    client = PersistentClient(path="./chroma") 
    collection_name = f"{method}-collection"
    collection = client.get_collection(name=collection_name)

    # User prompt
    user_prompt_content = Content(
        role="user",
        parts=[
            Part.from_text("Describe what to include in a deviation report in pharmaceutical manufacturing."),
        ],
    )
    
    print("user_prompt_content: ",user_prompt_content)
    response = generative_model.generate_content(
        user_prompt_content,
        generation_config=GenerationConfig(temperature=0),
        tools=[agent_tools.cheese_expert_tool],
        tool_config=ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        ))
    )
    print("LLM Response:", response)

    # Step 2: Execute the function and send chunks back to LLM
    function_calls = response.candidates[0].function_calls
    print("Function calls:")
    function_responses = agent_tools.execute_function_calls(function_calls,collection,embed_func=generate_query_embedding)
    if len(function_responses) == 0:
        print("Function calls did not result in any responses...")
    else:
        # Call LLM with retrieved responses
        response = generative_model.generate_content(
            [
                user_prompt_content,
                response.candidates[0].content,
                Content(
                    parts=function_responses
                ),
            ],
            tools=[agent_tools.cheese_expert_tool],
        )
        print("LLM Response:", response)


def main(args=None):
    print("CLI Arguments:", args)

    if args.chunk:
        chunk(method=args.chunk_type)

    if args.embed:
        embed(method=args.chunk_type)

    if args.load:
        load(method=args.chunk_type)

    if args.query:
        query(method=args.chunk_type)
    
    if args.chat:
        chat(method=args.chunk_type)
    
    if args.get:
        get(method=args.chunk_type)
    
    if args.agent:
        agent(method=args.chunk_type)


# Export important variables and functions for external imports
__all__ = [
    'GCP_PROJECT',
    'GCP_LOCATION', 
    'embedding_model',
    'generative_model',
    'generate_query_embedding',
    'PersistentClient',
    'CHROMADB_AVAILABLE',
    'VERTEX_AI_AVAILABLE'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI")

    parser.add_argument("--chunk", action="store_true", help="Chunk text")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings")
    parser.add_argument("--load", action="store_true", help="Load embeddings to vector db")
    parser.add_argument("--query", action="store_true", help="Query vector db")
    parser.add_argument("--chat", action="store_true", help="Chat with LLM")
    parser.add_argument("--get", action="store_true", help="Get documents from vector db")
    parser.add_argument("--agent", action="store_true", help="Chat with LLM Agent")
    parser.add_argument("--chunk_type", default="char-split", help="char-split | recursive-split | semantic-split")

    args = parser.parse_args()
    main(args)