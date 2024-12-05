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
    """
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len
        self.query = self.data['query']
        self.context = self.data['context']
        self.answer = self.data['answer']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = str(self.query[idx])
        context = str(self.context[idx])
        answer = str(self.answer[idx])

        source_text = f"query: {query} context: {context}"
        source = self.tokenizer.encode_plus(
            source_text, max_length=self.source_len, padding="max_length", truncation=True, return_tensors="pt"
        )
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
    """
    df = pd.read_csv(file_path)
    df = df[['question', 'answer']]
    df = df.dropna(subset=['question', 'answer'])
    df = df.rename(columns={'question': 'query', 'answer': 'answer'})
    df['context'] = df['answer']
    return df


def train_epoch(model, loader, optimizer, device, epoch, logging_steps):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}", disable=False)

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % logging_steps == 0:
            progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(loader)


def main():
    """
    Main function to fine-tune the T5 model for Retrieval-Augmented Generation (RAG).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="t5-base", type=str, help="Pretrained model name")
    parser.add_argument("--train_file", default="medquad.csv", type=str, help="Path to the dataset")
    parser.add_argument("--output_dir", default="rag_model", type=str, help="Path to save the fine-tuned model")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for training")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs")
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("--max_input_length", default=512, type=int, help="Max input length")
    parser.add_argument("--max_output_length", default=150, type=int, help="Max output length")
    parser.add_argument("--device", default="cuda", type=str, help="Device to train on (cuda, mps, or cpu)")
    parser.add_argument("--logging_steps", default=10, type=int, help="Logging steps")
    args = parser.parse_args()

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

    # save the model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
