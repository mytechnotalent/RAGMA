import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


def main():
    # load the dataset
    csv_file = "medquad.csv"  # path to your dataset
    df = pd.read_csv(csv_file)

    # add a 'context' column if it doesn't exist
    if 'context' not in df.columns:
        print("No 'context' column found. Creating it from the 'answer' column.")
        if 'answer' not in df.columns:
            raise ValueError("The dataset must have an 'answer' column to create the 'context'.")
        df['context'] = df['answer']  # use 'answer' as the context

    # save the updated dataset with the 'context' column
    updated_csv_file = "medquad_with_context.csv"
    df.to_csv(updated_csv_file, index=False)
    print(f"Updated dataset with 'context' column saved to {updated_csv_file}")

    # use SentenceTransformer to generate embeddings for the context column
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    contexts = df["context"].tolist()
    context_embeddings = embedder.encode(contexts, convert_to_tensor=False).astype("float32")

    # create a FAISS index for the embeddings
    dimension = context_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance
    index.add(context_embeddings)

    # save the FAISS index
    index_file = "context.index"
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}")


if __name__ == "__main__":
    main()
