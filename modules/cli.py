# Fix SQLite issue for Streamlit Cloud
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import os
import argparse
import pandas as pd
import json
import time
import glob
import hashlib
import chromadb

# Vertex AI
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part, ToolConfig
from google.api_core.exceptions import InternalServerError, ServiceUnavailable, ResourceExhausted

from chromadb import PersistentClient 
# Langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_experimental.text_splitter import SemanticChunker
from semantic_splitter import SemanticChunker
import agent_tools

# Setup
#GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_PROJECT = "${GCP_PROJECT:-rag-test-467013}"
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-2.5-flash"
INPUT_FOLDER = "input-datasets"
OUTPUT_FOLDER = "outputs"
CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#python
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 8192,  # Maximum number of tokens for output
    "temperature": 0.25,  # Control randomness in output
    "top_p": 0.95,  # Use nucleus sampling
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

generative_model = GenerativeModel(
	GENERATIVE_MODEL,
	system_instruction=[SYSTEM_INSTRUCTION]
)

book_mappings = {
	"21_CFR_Part_211-cleaned": {"author":"eCFR", "year": 2025},
	"Q7-Good-Manufacturing-Practice-Guidance-for-Active-Pharmaceutical-Ingredients-Guidance-for-Industry-cleaned":{"author": "FDA", "year": 2016}
}


def generate_query_embedding(query):
	query_embedding_inputs = [TextEmbeddingInput(task_type='RETRIEVAL_DOCUMENT', text=query)]
	kwargs = dict(output_dimensionality=EMBEDDING_DIMENSION) if EMBEDDING_DIMENSION else {}
	embeddings = embedding_model.get_embeddings(query_embedding_inputs, **kwargs)
	return embeddings[0].values


def generate_text_embeddings(chunks, dimensionality: int = 256, batch_size=250, max_retries=5, retry_delay=5):
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
			chunk_size = 350 # you can increase to 1500–2000 for longer sections
			chunk_overlap = 20
			# Init the splitter
			text_splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap, separator='', strip_whitespace=False)
	
			# Perform the splitting
			text_chunks = text_splitter.create_documents([input_text])
			text_chunks = [doc.page_content for doc in text_chunks]
			print("Number of chunks:", len(text_chunks))

		elif method == "recursive-split":
			# chunk_size = 350
			# chunk_overlap = 20
			# # Init the splitter
			# text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ".", " ", ""])

			# # Perform the splitting
			# text_chunks = text_splitter.create_documents([input_text])
			# text_chunks = [doc.page_content for doc in text_chunks]
			# print("Number of chunks:", len(text_chunks))

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

			# Example usage
			text_chunks = smart_chunk(input_text)
			print("Number of chunks:", len(text_chunks))
		
		elif method == "semantic-split":
			# Init the splitter
			text_splitter = SemanticChunker(embedding_function=generate_text_embeddings)
			# Perform the splitting
			text_chunks = text_splitter.create_documents([input_text])
			text_chunks = [doc.page_content for doc in text_chunks]
			print("Number of chunks:", len(text_chunks))

		if text_chunks is not None:
			# Save the chunks
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
	print("load()")
	

	# Clear Cache
	chromadb.api.client.SharedSystemClient.clear_system_cache()

	# Connect to chroma DB
	# client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
	client = PersistentClient(path="./chroma") 

	# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
	collection_name = f"{method}-collection"
	print("Creating collection:", collection_name)

	try:
		# Clear out any existing items in the collection
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
	#client.persist()
	#print("✅ Collection persisted to disk.")


# def query(method="char-split"):
# 	print("load()")

# 	# Connect to chroma DB
# 	# client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
# 	client = chromadb.Client()

# 	# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
# 	collection_name = f"{method}-collection"

# 	query = "Tolminc cheese"
# 	query_embedding = generate_query_embedding(query)
# 	print("Embedding values:", query_embedding)

# 	# Get the collection
# 	collection = client.get_or_create_collection(name=collection_name)

# 	print("Checking sample data in collection...")
# 	all_data = collection.get()
# 	print("Sample IDs:", all_data["ids"][:3])
# 	print("Sample metadata:", all_data["metadatas"][:3])
# 	print("Sample docs:", all_data["documents"][:1])

# 	# 1: Query based on embedding value 
# 	results = collection.query(
# 		query_embeddings=[query_embedding],
# 		n_results=10
# 	)
# 	print("Query:", query)
# 	print("\n\nResults:", results)

# 	# 2: Query based on embedding value + metadata filter
# 	results = collection.query(
# 		query_embeddings=[query_embedding],
# 		n_results=10,
# 		where={"book":"The Complete Book of Cheese"}
# 	)
# 	print("Query:", query)
# 	print("\n\nResults:", results)

# 	# 3: Query based on embedding value + lexical search filter
# 	search_string = "Italian"
# 	results = collection.query(
# 		query_embeddings=[query_embedding],
# 		n_results=10,
# 		where_document={"$contains": search_string}
# 	)
# 	print("Query:", query)
# 	print("\n\nResults:", results)

def query(method="char-split"):
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
		n_results=5  # or however many
	)

	# Show results
	print("Query:", user_query)
	for i, doc in enumerate(results["documents"][0]):
		print(f"\nResult #{i+1}")
		print("Doc:", doc)
		print("Metadata:", results["metadatas"][0][i])
		print("Distance:", results["distances"][0][i])



def chat(method="char-split"):
	print("chat()")

	# Connect to chroma DB
	# client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
	client = PersistentClient(path="./chroma") 
	# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
	collection_name = f"{method}-collection"

	#query = "What is the best practice for batch production records?"
	query = "What happened with Lot 10000295 "
	#query = "Was any equipment operated without dual signoff?"
	#query = "What is gouda cheese?"

	query_embedding = generate_query_embedding(query)
	print("Query:", query)
	print("Embedding values:", query_embedding)
	# Get the collection
	collection = client.get_collection(name=collection_name)

	# Query based on embedding value 
	results = collection.query(
		query_embeddings=[query_embedding],
		n_results=10
	)
	print("\n\nResults:", results)

	print(len(results["documents"][0]))

	INPUT_PROMPT = f"""
	{query}
	{"\n".join(results["documents"][0])}
	"""

	print("INPUT_PROMPT: ",INPUT_PROMPT)
	response = generative_model.generate_content(
		[INPUT_PROMPT],  # Input prompt
		generation_config=generation_config,  # Configuration settings
		stream=False,  # Enable streaming for responses
	)
	generated_text = response.text
	print("LLM Response:", generated_text)


def get(method="char-split"):
	print("get()")

	# Connect to chroma DB
	# client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
	client = PersistentClient(path="./chroma")
	# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
	collection_name = f"{method}-collection"

	# Get the collection
	collection = client.get_collection(name=collection_name)

	# Get documents with filters
	results = collection.get(
		where={"book":"The Complete Book of Cheese"},
		limit=10
	)
	print("\n\nResults:", results)


def agent(method="char-split"):
	print("agent()")

	# Connect to chroma DB
	# client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
	client = PersistentClient(path="./chroma") 
	# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
	collection_name = f"{method}-collection"
	# Get the collection
	collection = client.get_collection(name=collection_name)

	# User prompt
	user_prompt_content = Content(
    	role="user",
		parts=[
			Part.from_text("Describe what to include in a deviation report in pharmaceutical manufacturing."),
		],
	)
	
	# Step 1: Prompt LLM to find the tool(s) to execute to find the relevant chunks in vector db
	print("user_prompt_content: ",user_prompt_content)
	response = generative_model.generate_content(
		user_prompt_content,
		generation_config=GenerationConfig(temperature=0),  # Configuration settings
		tools=[agent_tools.cheese_expert_tool],  # Tools available to the model
		tool_config=ToolConfig(
			function_calling_config=ToolConfig.FunctionCallingConfig(
				# ANY mode forces the model to predict only function calls
				mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
		))
	)
	print("LLM Response:", response)

	# Step 2: Execute the function and send chunks back to LLM to answer get the final response
	function_calls = response.candidates[0].function_calls
	print("Function calls:")
	function_responses = agent_tools.execute_function_calls(function_calls,collection,embed_func=generate_query_embedding)
	if len(function_responses) == 0:
		print("Function calls did not result in any responses...")
	else:
		# Call LLM with retrieved responses
		response = generative_model.generate_content(
			[
				user_prompt_content,  # User prompt
				response.candidates[0].content,  # Function call response
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


if __name__ == "__main__":
	# Generate the inputs arguments parser
	# if you type into the terminal '--help', it will provide the description
	parser = argparse.ArgumentParser(description="CLI")

	parser.add_argument(
		"--chunk",
		action="store_true",
		help="Chunk text",
	)
	parser.add_argument(
		"--embed",
		action="store_true",
		help="Generate embeddings",
	)
	parser.add_argument(
		"--load",
		action="store_true",
		help="Load embeddings to vector db",
	)
	parser.add_argument(
		"--query",
		action="store_true",
		help="Query vector db",
	)
	parser.add_argument(
		"--chat",
		action="store_true",
		help="Chat with LLM",
	)
	parser.add_argument(
		"--get",
		action="store_true",
		help="Get documents from vector db",
	)
	parser.add_argument(
		"--agent",
		action="store_true",
		help="Chat with LLM Agent",
	)
	parser.add_argument("--chunk_type", default="char-split", help="char-split | recursive-split | semantic-split")

	args = parser.parse_args()

	main(args)