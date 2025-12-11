import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from document_parser.document import Document
import numpy as np

MAX_LEN = 400 
BATCH_SIZE = 8   # Low batch size because sequence length is huge
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertEmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, model_max_length=MAX_LEN)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()

    def embed(self, document: Document) -> np.ndarray:
        """Generate BERT embedding for the given document."""
        inputs = self.tokenizer(document.text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)

        with torch.no_grad():
            outputs = self.model.bert(**inputs)  # accessing .bert to skip the generic head

            # Get the embedding of the [CLS] token (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding.cpu().numpy().squeeze()
    
        
if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    with open("data/corpus.txt", "w", encoding="utf-8") as f:
        # Creating long sentences to test the 2000 limit
        for i in range(100):
            long_text = "This is a word. " * 300  # ~1500 tokens
            f.write(long_text + "\n")


    print("Training Tokenizer...")
    tokenizer = BertWordPieceTokenizer(clean_text=True, lowercase=True)
    tokenizer.train(
        files=["data/corpus.txt"],
        vocab_size=10_000, # Small vocab for demo
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.save_model("my_tokenizer")

    tokenizer = BertTokenizerFast.from_pretrained("my_tokenizer", model_max_length=MAX_LEN)

    class BERTDataset(Dataset):
        def __init__(self, file_path, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = max_length
            with open(file_path, "r", encoding="utf-8") as f:
                self.lines = [line.strip() for line in f if line.strip()]

        def __len__(self):
            return len(self.lines)

        def __getitem__(self, idx):
            text = self.lines[idx]
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].squeeze() 
            attention_mask = encoding["attention_mask"].squeeze()
            
            labels = input_ids.clone()
            
            rand = torch.rand(input_ids.shape)
            
            mask_arr = (rand < 0.15) * (input_ids != 101) * (input_ids != 102) * (input_ids != 0)
            
            selection = torch.flatten(mask_arr.nonzero()).tolist()
            
            input_ids[selection] = 4 
            
            labels[~mask_arr] = -100 
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

    dataset = BERTDataset("data/corpus.txt", tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    config = BertConfig(
        vocab_size=10_000,
        hidden_size=256,     
        num_hidden_layers=4,   
        num_attention_heads=4, 
        intermediate_size=1024,
        max_position_embeddings=MAX_LEN,
        type_vocab_size=1
    )

    model = BertForMaskedLM(config)
    model.to(DEVICE)
    model.train()

    optim = AdamW(model.parameters(), lr=LR)

    print(f"Starting training on {DEVICE}...")
    print(f"Input shape: {MAX_LEN} tokens")

    for epoch in range(EPOCHS):
        loop = tqdm(loader, leave=True)
        total_loss = 0
        
        for batch in loop:
            # Move batch to GPU
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optim.zero_grad()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            
            loss.backward()
            
            optim.step()
            
            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

    model.save_pretrained("./my_long_bert")
    tokenizer.save_pretrained("./my_long_bert")

    print("Training complete. Testing embedding generation...")

    # Extract embeddings
    model.eval()
    test_text = "This is a test of the long embedding model."
    inputs = tokenizer(test_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.bert(**inputs)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]