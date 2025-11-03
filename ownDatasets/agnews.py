from torch.utils.data import Dataset
import torch
import numpy as np

class MyAGNewsDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128, imbalance_factor=1.0):
        """
        Initialize the AG News dataset.

        Args:
        hf_dataset (Dataset): Hugging Face dataset object.
        tokenizer (BertTokenizer): Tokenizer used for text encoding.
        max_length (int): Maximum length of text sequences (default: 128).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = hf_dataset['text']
        self.targets = torch.tensor(hf_dataset['label'], dtype=torch.long)
        self.classes = ['World', 'Sports', 'Business', 'Sci/Tech']

        if imbalance_factor:
            # Extract texts and labels
            texts = hf_dataset['text']
            labels = hf_dataset['label']

            # Create a dictionary to store samples for each class
            class_samples = {cls: [] for cls in range(len(self.classes))}

            # Assign samples to the corresponding class
            for text, label in zip(texts, labels):
                class_samples[label].append(text)

            # Compute the number of samples per class to follow a long-tail distribution
            num_classes = len(self.classes)
            max_samples = max(len(samples) for samples in class_samples.values())
            class_sizes = [int(max_samples * (imbalance_factor ** (i / (num_classes - 1))))
                        for i in range(num_classes)]

            # Build the imbalanced dataset
            self.data = []
            self.targets = []
            for cls in range(num_classes):
                samples = class_samples[cls]
                num_samples = min(len(samples), class_sizes[cls])
                selected_samples = np.random.choice(samples, num_samples, replace=False)
                self.data.extend(selected_samples)
                self.targets.extend([cls] * num_samples)

            self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __getitem__(self, index):
        """
        Get a sample by index.

        Args:
        index (int): Index of the sample.

        Returns:
        dict: A dictionary containing input_ids, attention_mask, labels, and index.
        """
        text = self.data[index]
        label = self.targets[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label,
            'index': index
        }

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
        int: Number of samples.
        """
        return len(self.data)


class AGNewsCausalLMOptionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128, imbalance_factor=1.0):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        texts = hf_dataset['text']
        labels = hf_dataset['label']

        self.option_texts = [
            "A",  # 0: World
            "B",  # 1: Sports
            "C",  # 2: Business
            "D",  # 3: Sci/Tech
        ]
        self.classes = [
            "World",
            "Sports",
            "Business",
            "Sci/Tech",
        ]

        self.data_texts = list(texts)
        self.targets    = list(labels)

        # Apply long-tail imbalance if specified (same trigger behavior as MyAGNewsDataset)
        if imbalance_factor:
            # Group samples by class id
            num_classes = len(self.classes)
            class_samples = {cid: [] for cid in range(num_classes)}
            for t, y in zip(texts, labels):
                class_samples[int(y)].append(t)

            # Long-tail sizes (class 0 最大 → class 3 最小)
            max_samples = max(len(v) for v in class_samples.values())
            class_sizes = [
                int(max_samples * (imbalance_factor ** (i / (num_classes - 1))))
                for i in range(num_classes)
            ]

            # Rebuild imbalanced lists
            new_texts, new_labels = [], []
            for cid in range(num_classes):
                samples = class_samples[cid]
                n_take = min(len(samples), class_sizes[cid])
                if n_take > 0:
                    selected = np.random.choice(samples, n_take, replace=False)
                    new_texts.extend(selected.tolist() if hasattr(selected, "tolist") else list(selected))
                    new_labels.extend([cid] * n_take)

            self.data_texts = new_texts
            self.targets    = new_labels

        # Convert targets to torch tensor for consistency (kept as list in __getitem__ for speed)
        self.targets = list(map(int, self.targets))

    def __len__(self):
        return len(self.data_texts)

    def __getitem__(self, idx):
        text = self.data_texts[idx]
        label_id = self.targets[idx]  # 0~3

        prompt = (
            "Classify the following news into one of the options, please answeer with a single charactor 'A', 'B', 'C' or 'D':\n"
            "A. World\n"
            "B. Sports\n"
            "C. Business\n"
            "D. Sci/Tech\n\n"
            f"News: {text}\n"
            "Answer: "
        )

        answer = self.option_texts[label_id]

        full_text = prompt + answer

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        prompt_enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].size(1)

        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "option_id": torch.tensor(label_id, dtype=torch.long),
            "index": idx,
        }
