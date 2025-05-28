"""
Dataloaders with various datasets to train LLMs
Reference : https://github.com/rasbt/LLMs-from-scratch
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import pandas as pd
import json
from functools import partial

# ----------
# Next token predicton LLM datasets
# ---------

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Vocab size
        self.vocab_size = tokenizer.n_vocab

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_gpt_dataloaders(file_path, train_ratio=0.9, batch_size=8, max_length=4, stride=4, num_workers=0, tokenizer=None):

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    text_data = text_data + " <|endoftext|> "

    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    test_data = text_data[split_idx:]

    # Initialize the tokenizer
    #tokenizer = tiktoken.get_encoding("gpt2")

    # Create datasets
    train_dataset = GPTDataset(train_data, tokenizer, max_length, stride)
    val_dataset = GPTDataset(test_data, tokenizer, max_length, stride)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    # # Check dataloader
    # data_iter = iter(dataloader)
    # inputs, targets = next(data_iter)
    # print(inputs)
    # print(targets)

    # # Same as:
    # dataloader = DataLoader(
    #     dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    # for inputs, targets in dataloader:
    #     print(inputs)
    #     print(targets)
    #     break

    return train_loader, val_loader

def create_gpt_dataloaders_full_dataset(train_ratio=0.9, batch_size=8, max_length=4, stride=4, num_workers=0, tokenizer=None):

    # Check data
    data_dir = '../data/gutenberg_preprocessed'
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)
    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files:", total_files)

    # Iterate over the books in the training corpus
    text_data = ""
    for index, file_path in enumerate(all_files, 1):
        print(f"Parsing file {index} of {total_files}: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            text_data_tmp = file.read()
        text_data = text_data_tmp + " <|endoftext|> "

    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    test_data = text_data[split_idx:]

    # Initialize the tokenizer
    #tokenizer = tiktoken.get_encoding("gpt2")

    # Create datasets
    train_dataset = GPTDataset(train_data, tokenizer, max_length, stride)
    val_dataset = GPTDataset(test_data, tokenizer, max_length, stride)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    # # Check dataloader
    # data_iter = iter(dataloader)
    # inputs, targets = next(data_iter)
    # print(inputs)
    # print(targets)

    # # Same as:
    # dataloader = DataLoader(
    #     dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    # for inputs, targets in dataloader:
    #     print(inputs)
    #     print(targets)
    #     break

    return train_loader, val_loader


# ----------
# Spam classifier LLM datasets
# ---------

class SpamDataset(Dataset):

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
        # Note: A more pythonic version to implement this method
        # is the following, which is also used in the next chapter:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)

def create_spam_cls_dataloaders(train_ratio=0.7, 
                                val_ratio=0.1, 
                                batch_size=8, 
                                num_workers=0, 
                                tokenizer=None):

    def create_balanced_dataset(df):
        # Count the instances of "spam"
        num_spam = df[df["Label"] == "spam"].shape[0]
        # Randomly sample "ham" instances to match the number of "spam" instances
        ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
        # Combine ham "subset" with "spam"
        balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
        return balanced_df

    def random_split(df, train_frac, validation_frac):
        # Shuffle the entire DataFrame
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        # Calculate split indices
        train_end = int(len(df) * train_frac)
        validation_end = train_end + int(len(df) * validation_frac)
        # Split the DataFrame
        train_df = df[:train_end]
        validation_df = df[train_end:validation_end]
        test_df = df[validation_end:]
        return train_df, validation_df, test_df

    # Create train, val, test splits
    file_path = "../data/sms_spam_collection/SMSSpamCollection.tsv"
    df = pd.read_csv(file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, validation_df, test_df = random_split(balanced_df, train_ratio, val_ratio)
    train_csv = "../data/sms_spam_collection/train.csv"
    val_csv = "../data/sms_spam_collection/val.csv"
    test_csv = "../data/sms_spam_collection/test.csv"
    train_df.to_csv(train_csv, index=None)
    validation_df.to_csv(val_csv, index=None)
    test_df.to_csv(test_csv, index=None)

    # Initialize the tokenizer
    #tokenizer = tiktoken.get_encoding("gpt2")

    # Create datasets
    train_dataset = SpamDataset(
        csv_file=train_csv,
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file=val_csv,
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file=test_csv,
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

# ----------
# Instruct LLM datasets
# ---------

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

def create_instruct_llm_dataloaders(device, 
                                train_ratio=0.85, 
                                test_ratio=0.1, 
                                batch_size=8, 
                                num_workers=0, 
                                tokenizer=None):

    file_path = "../data/instruction-data.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    train_portion = int(len(data) * train_ratio)  # 85% for training
    test_portion = int(len(data) * test_ratio)    # 10% for testing

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # Use very small subset for testing purposes
    if 0:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50*"-")

    # Initialize the tokenizer
    #tokenizer = tiktoken.get_encoding("gpt2")

    # Create datasets
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    # Create dataloaders
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return train_loader, val_loader, val_data, test_data
