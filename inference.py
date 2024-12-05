import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import faiss
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_index(index_file, csv_file):
	"""
	Load the FAISS index and the associated dataset.

	Args:
		index_file (str): Path to the FAISS index file.
		csv_file (str): Path to the CSV file containing the dataset.

	Returns:
		faiss.IndexFlatL2: The loaded FAISS index.
		pd.DataFrame: The dataset containing queries, contexts, and answers.
	"""
	df = pd.read_csv(csv_file)	# load the dataset
	index = faiss.read_index(index_file)  # load the FAISS index
	return index, df


def retrieve_context(query, index, df, embedder, top_k=1):
	"""
	Retrieve the most relevant context(s) from the FAISS index based on the query.

	Args:
		query (str): The user's input question.
		index (faiss.IndexFlatL2): The FAISS index for retrieval.
		df (pd.DataFrame): The dataset to retrieve contexts from.
		embedder (SentenceTransformer): The embedding model to encode the query.
		top_k (int): Number of top contexts to retrieve.

	Returns:
		list: A list of retrieved contexts.
	"""
	query_vector = embedder.encode([query]).astype("float32")  # embed the query
	distances, indices = index.search(query_vector, top_k)	# search the FAISS index
	return [df.iloc[i]["context"] for i in indices[0]]	# retrieve contexts by index


def main():
	"""
	Main function to handle inference in a RAG system.

	Uses a FAISS index for context retrieval and a fine-tuned T5 model for answer generation.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--query", type=str, required=True, help="Input query")
	parser.add_argument("--model_dir", type=str, default="rag_model", help="Fine-tuned model directory")
	parser.add_argument("--csv_file", type=str, default="medquad.csv", help="CSV file for context")
	parser.add_argument("--index_file", type=str, default="context.index", help="FAISS index file")
	parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda, mps, or cpu)")
	parser.add_argument("--top_k", type=int, default=1, help="Number of contexts to retrieve")
	args = parser.parse_args()

	# enhanced device selection logic
	if args.device == "mps" and torch.backends.mps.is_available():
		device = torch.device("mps")
	elif args.device == "cuda" and torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")
	print(f"Using device: {device}")

	# load the fine-tuned model and tokenizer
	tokenizer = T5Tokenizer.from_pretrained(args.model_dir, legacy=False)
	model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)

	# load the FAISS index and dataset
	embedder = SentenceTransformer("all-MiniLM-L6-v2")	# use a sentence embedding model
	index, df = load_index(args.index_file, args.csv_file)

	# retrieve the most relevant context for the input query
	contexts = retrieve_context(args.query, index, df, embedder, args.top_k)
	input_text = f"query: {args.query} context: {' '.join(contexts)}"

	# generate the answer using the fine-tuned model
	input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
	input_length = len(input_ids[0])  # length of the input query + context
	outputs = model.generate(
		input_ids,
		max_length=input_length + 150,	# allow the model to generate a longer response
		num_beams=5,
		no_repeat_ngram_size=2
	)
	answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
	if "." in answer:
		answer = answer[:answer.rfind(".") + 1]  # trim to the last full sentence


	# display the results
	print(f"Query: {args.query}")
	print(f"Retrieved Context: {' '.join(contexts)}")
	print(f"Generated Answer: {answer}")


if __name__ == "__main__":
	main()
