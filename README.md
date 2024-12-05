# Retrieval-Augmented Generation Medical Assistant (RAGMA)

### [dataset](https://www.kaggle.com/datasets/jpmiller/layoutlm)

Author: [Kevin Thomas](mailto:ket189@pitt.edu)

License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0)

## RAG Model Training, Index Creation, and Inference: Explained

This notebook will help you understand three Python scripts designed to work with a Retrieval-Augmented Generation (RAG) model. These scripts are `train.py`, `create_faiss_index.py`, and `inference.py`. Each serves a distinct purpose in the RAG workflow: training the model, creating a searchable index, and running inference.

---

### 1. **What is RAG?**
Retrieval-Augmented Generation (RAG) is a method in Natural Language Processing (NLP) where:
- **Retrieval:** You fetch relevant information (documents, passages) based on a query.
- **Generation:** You generate meaningful responses by combining the retrieved information with a generative model like T5.

---

### 2. **`train.py`**: Fine-Tuning the T5 Model

This script fine-tunes a pretrained T5 model on a dataset for retrieval-augmented tasks. Here’s what it does:

#### a. **Dataset**
- The dataset must include three columns: `query`, `context`, and `answer`.
  - `query`: The question or input from the user.
  - `context`: Supporting information for the query (in this case, the same as `answer` initially).
  - `answer`: The correct response to the query.

#### b. **Data Preprocessing**
The `preprocess_data` function:
- Reads the dataset from a CSV file.
- Renames columns (`question` to `query` and `answer` remains the same).
- Creates a `context` column from the `answer`.

#### c. **Model Fine-Tuning**
- The T5 model (`T5ForConditionalGeneration`) is fine-tuned using the `RAGDataset` class.
- The dataset is tokenized and padded/truncated to specified lengths.
- Training occurs in multiple epochs, and the loss is minimized using the AdamW optimizer.

#### d. **Output**
- The fine-tuned model and tokenizer are saved to a directory (`rag_model` by default).

#### Key Takeaways:
- `train.py` ensures the T5 model learns to map queries to answers effectively by utilizing the provided context.
- This is the **training phase** of the RAG pipeline.

---

### 3. **`create_faiss_index.py`**: Building the FAISS Index

This script focuses on creating an index for **retrieving** relevant contexts.

#### a. **Dataset**
- Similar to `train.py`, the script processes a dataset with `query`, `context`, and `answer` columns.
- If the `context` column does not exist, it creates one from the `answer` column.

#### b. **Embedding Generation**
- Uses `SentenceTransformer` (model: `all-MiniLM-L6-v2`) to generate dense vector embeddings for the `context` column.
- Each context is transformed into a numerical representation.

#### c. **FAISS Index**
- FAISS (Facebook AI Similarity Search) is a library for efficient similarity search.
- The script:
  - Creates an index using L2 (Euclidean) distance.
  - Adds the embeddings of all contexts to the FAISS index.

#### d. **Output**
- The FAISS index is saved to a file (`context.index`).

#### Key Takeaways:
- `create_faiss_index.py` prepares the **retrieval mechanism** by indexing context embeddings.
- This is the **retrieval phase** of the RAG pipeline.

---

### 4. **`inference.py`**: Running the RAG Model

This script ties everything together for inference, where we can query the model and receive a generated response.

#### a. **Dataset**
- Loads the dataset (`medquad.csv`).
- Ensures the `context` column exists and uses the `answer` column if not.

#### b. **Retrieval**
- The FAISS index (`context.index`) is loaded to fetch the most relevant context for a given query.

#### c. **Inference**
- A query is passed through the retrieval step to find the closest context.
- The fine-tuned T5 model uses the retrieved context to generate an answer.

#### Key Takeaways:
- `inference.py` demonstrates the **full RAG pipeline**, combining retrieval and generation.
- This is the **inference phase** of the RAG workflow.

---

### 5. **Summary**

#### Workflow:
1. **Train (`train.py`)**:
   - Fine-tune the T5 model on a dataset of queries, contexts, and answers.
2. **Create Index (`create_faiss_index.py`)**:
   - Generate embeddings for the contexts and build a FAISS index for retrieval.
3. **Inference (`inference.py`)**:
   - Retrieve the most relevant context using FAISS.
   - Use the fine-tuned T5 model to generate answers based on the query and context.

---

### 6. **Analogy: A Smart Librarian**
Imagine you’re asking a librarian for help:
1. **Train**: The librarian learns how to answer questions (fine-tuning T5).
2. **Index**: The librarian organizes all the books efficiently (FAISS indexing).
3. **Inference**: You ask the librarian a question, they find the relevant book (retrieval), and then summarize the answer for you (generation).

This combination of retrieval and generation makes RAG a powerful tool for tasks like Q&A systems or chatbots.

## Install Libraries


```python
!pip install pandas torch transformers scikit-learn tqdm faiss-cpu sentence-transformers
```

    Requirement already satisfied: pandas in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (2.2.2)
    Requirement already satisfied: torch in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (2.5.1)
    Requirement already satisfied: transformers in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (4.46.2)
    Requirement already satisfied: scikit-learn in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (1.5.1)
    Requirement already satisfied: tqdm in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (4.66.5)
    Requirement already satisfied: faiss-cpu in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (1.9.0)
    Requirement already satisfied: sentence-transformers in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (3.3.0)
    Requirement already satisfied: numpy>=1.26.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from pandas) (1.26.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from pandas) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from pandas) (2023.3)
    Requirement already satisfied: filelock in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (3.13.1)
    Requirement already satisfied: typing-extensions>=4.8.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (4.11.0)
    Requirement already satisfied: networkx in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (3.3)
    Requirement already satisfied: jinja2 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (3.1.4)
    Requirement already satisfied: fsspec in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (2024.6.1)
    Requirement already satisfied: setuptools in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (75.1.0)
    Requirement already satisfied: sympy==1.13.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from sympy==1.13.1->torch) (1.3.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from transformers) (0.26.2)
    Requirement already satisfied: packaging>=20.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from transformers) (24.1)
    Requirement already satisfied: pyyaml>=5.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from transformers) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from transformers) (2024.9.11)
    Requirement already satisfied: requests in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from transformers) (2.32.3)
    Requirement already satisfied: safetensors>=0.4.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from transformers) (0.4.5)
    Requirement already satisfied: tokenizers<0.21,>=0.20 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from transformers) (0.20.3)
    Requirement already satisfied: scipy>=1.6.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from scikit-learn) (1.13.1)
    Requirement already satisfied: joblib>=1.2.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from scikit-learn) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from scikit-learn) (3.5.0)
    Requirement already satisfied: Pillow in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from sentence-transformers) (10.4.0)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from jinja2->torch) (2.1.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from requests->transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from requests->transformers) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from requests->transformers) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from requests->transformers) (2024.8.30)


## Train


```python
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm.auto import tqdm


class RAGDataset(Dataset):
    """
    Custom Dataset class for loading query-context-answer pairs for training the RAG model.
    This class handles tokenizing the data and preparing it for PyTorch's DataLoader.
    """
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        """
        Initialize the dataset.
        
        Args:
            dataframe (pd.DataFrame): The dataset containing query, context, and answer columns.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding text.
            source_len (int): Maximum length for the input sequence.
            target_len (int): Maximum length for the target sequence.
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len
        self.query = self.data['query']
        self.context = self.data['context']
        self.answer = self.data['answer']

    def __len__(self):
        """
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single data point from the dataset.
        
        Args:
            idx (int): Index of the data point.
        
        Returns:
            dict: Dictionary containing tokenized input and target sequences.
        """
        query = str(self.query[idx])
        context = str(self.context[idx])
        answer = str(self.answer[idx])

        # combine query and context into a single input string
        source_text = f"query: {query} context: {context}"
        
        # tokenize the input string
        source = self.tokenizer.encode_plus(
            source_text, max_length=self.source_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        # tokenize the answer string
        target = self.tokenizer.encode_plus(
            answer, max_length=self.target_len, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": target["input_ids"].squeeze(),
        }


def preprocess_data(file_path):
    """
    Preprocess the dataset to include required columns and handle missing values.
    
    Args:
        file_path (str): Path to the dataset CSV file.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe with 'query', 'context', and 'answer' columns.
    """
    # load the CSV file
    df = pd.read_csv(file_path)
    
    # retain only the 'question' and 'answer' columns
    df = df[['question', 'answer']]
    
    # drop rows with missing values
    df = df.dropna(subset=['question', 'answer'])
    
    # rename columns for consistency
    df = df.rename(columns={'question': 'query', 'answer': 'answer'})
    
    # add a 'context' column (using the answer as context for now)
    df['context'] = df['answer']
    return df


def train_epoch(model, loader, optimizer, device, epoch, logging_steps):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model being trained.
        loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to run the model on (CPU, GPU, etc.).
        epoch (int): Current epoch number.
        logging_steps (int): Frequency of logging progress during training.
    
    Returns:
        float: The average training loss for the epoch.
    """
    model.train()  # set the model to training mode
    total_loss = 0  # initialize total loss
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}", disable=False)  # progress bar for tracking

    for step, batch in enumerate(progress_bar):
        # move inputs and labels to the specified device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # forward pass through the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the loss every `logging_steps`
        if (step + 1) % logging_steps == 0:
            progress_bar.set_postfix({"loss": loss.item()})

    # return the average loss for the epoch
    return total_loss / len(loader)


def main():
    """
    Main function to fine-tune the T5 model for Retrieval-Augmented Generation (RAG).
    This version handles Jupyter Notebook's extra arguments gracefully.
    """
    # simulating command-line arguments for Jupyter Notebook
    class Args:
        model_name = "t5-base"
        train_file = "medquad.csv"
        output_dir = "rag_model"
        batch_size = 8
        epochs = 3
        lr = 5e-5
        max_input_length = 512
        max_output_length = 150
        device = "mps"
        logging_steps = 10

    args = Args()  # use the custom Args class to store arguments

    # enhanced device selection logic
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # preprocess the data
    df = preprocess_data(args.train_file)

    # split data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    # create DataLoaders for training and validation datasets
    train_dataset = RAGDataset(train_df, tokenizer, args.max_input_length, args.max_output_length)
    val_dataset = RAGDataset(val_df, tokenizer, args.max_input_length, args.max_output_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args.logging_steps)
        print(f"Epoch {epoch} Training Loss: {train_loss:.4f}")

    # save the model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
```


```python
main()
```

    Using device: mps



    Epoch 1:   0%|          | 0/1846 [00:00<?, ?it/s]


    Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.


    Epoch 1 Training Loss: 0.0865



    Epoch 2:   0%|          | 0/1846 [00:00<?, ?it/s]


    Epoch 2 Training Loss: 0.0135



    Epoch 3:   0%|          | 0/1846 [00:00<?, ?it/s]


    Epoch 3 Training Loss: 0.0099
    Model saved to rag_model


## Create Faiss Index


```python
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# define parameters
csv_file = "medquad.csv"  # Path to your dataset
updated_csv_file = "medquad_with_context.csv"  # Output dataset path
index_file = "context.index"  # Path to save the FAISS index

# load the dataset
df = pd.read_csv(csv_file)

# add a 'context' column if it doesn't exist
if 'context' not in df.columns:
    print("No 'context' column found. Creating it from the 'answer' column.")
    if 'answer' not in df.columns:
        raise ValueError("The dataset must have an 'answer' column to create the 'context'.")
    df['context'] = df['answer']  # Use 'answer' as the context

# save the updated dataset with the 'context' column
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
faiss.write_index(index, index_file)
print(f"FAISS index saved to {index_file}")
```

    No 'context' column found. Creating it from the 'answer' column.
    Updated dataset with 'context' column saved to medquad_with_context.csv
    FAISS index saved to context.index


## Inference


```python
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import faiss
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
    df = pd.read_csv(csv_file)  # Load the dataset
    index = faiss.read_index(index_file)  # Load the FAISS index
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
    query_vector = embedder.encode([query]).astype("float32")  # Embed the query
    distances, indices = index.search(query_vector, top_k)  # Search the FAISS index
    return [df.iloc[i]["context"] for i in indices[0]]  # Retrieve contexts by index


# simulated arguments for the Jupyter Notebook
args = {
    "query": "What are the symptoms of diabetes?",
    "model_dir": "rag_model",  # fine-tuned model directory
    "csv_file": "medquad_with_context.csv",  # CSV file containing the dataset
    "index_file": "context.index",  # FAISS index file
    "device": "mps",  # device to run the inference on
    "top_k": 1  # number of top contexts to retrieve
}

# enhanced device selection logic
if args["device"] == "mps" and torch.backends.mps.is_available():
    device = torch.device("mps")
elif args["device"] == "cuda" and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# load the fine-tuned model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(args["model_dir"], legacy=False)
model = T5ForConditionalGeneration.from_pretrained(args["model_dir"]).to(device)

# load the FAISS index and dataset
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # use a sentence embedding model
index, df = load_index(args["index_file"], args["csv_file"])

# retrieve the most relevant context for the input query
contexts = retrieve_context(args["query"], index, df, embedder, args["top_k"])
input_text = f"query: {args['query']} context: {' '.join(contexts)}"

# generate the answer using the fine-tuned model
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
input_length = len(input_ids[0])  # length of the input query + context
outputs = model.generate(
    input_ids,
    max_length=input_length + 150,  # allow the model to generate a longer response
    num_beams=5,
    no_repeat_ngram_size=2
)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
if "." in answer:
    answer = answer[:answer.rfind(".") + 1]  # trim to the last full sentence

# display the results
print(f"Query: {args['query']}")
print(f"Retrieved Context: {' '.join(contexts)}")
print(f"Generated Answer: {answer}")
```

    Using device: mps
    Query: What are the symptoms of diabetes?
    Retrieved Context: The signs and symptoms of diabetes are
                    
    - being very thirsty  - urinating often  - feeling very hungry  - feeling very tired  - losing weight without trying  - sores that heal slowly  - dry, itchy skin  - feelings of pins and needles in your feet  - losing feeling in your feet  - blurry eyesight
                    
    Some people with diabetes dont have any of these signs or symptoms. The only way to know if you have diabetes is to have your doctor do a blood test.
    Generated Answer: The signs and symptoms of diabetes are - being very thirsty , urinating often – feeling very hungry ­ feeling extremely tired — losing weight without trying . sores that heal slowly : dry, itchy skin  feelings of pins and needles in your feet _ losing feeling in you feet- blurry eyesight Some people with diabetes dont have any of these signs or symptoms. The only way to know if you have diabetes is to have your doctor do a blood test.

