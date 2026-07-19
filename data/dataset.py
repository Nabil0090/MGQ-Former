import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import torchvision.transforms as T
import kagglehub

# Download dataset dynamically
print("Downloading EarthVQA dataset...")
BASE = kagglehub.dataset_download("alienxc137/earthvqa-semantic-segmentation-visual-question-ans")
print(f"Dataset downloaded to: {BASE}")

QA_DIR         = os.path.join(BASE, "2024EarthVQA", "2024EarthVQA")
TRAIN_IMG_DIR  = os.path.join(BASE, "Train-003", "Train", "images_png")
TRAIN_MASK_DIR = os.path.join(BASE, "Train-003", "Train", "masks_png")
VAL_IMG_DIR    = os.path.join(BASE, "Val-002", "Val", "images_png")
VAL_MASK_DIR   = os.path.join(BASE, "Val-002", "Val", "masks_png")
TEST_IMG_DIR   = os.path.join(BASE, "Test-001", "images_png")
def build_answer_vocab(qa_raw):
    answers = set()
    for qa_pairs in qa_raw.values():
        for pair in qa_pairs:
            answers.add(str(pair['Answer']))
    ans2idx = {a: i for i, a in enumerate(sorted(answers))}
    idx2ans = {i: a for a, i in ans2idx.items()}
    return ans2idx, idx2ans

def format_generative_target(answer, question_type):
    return answer

class EarthVQADataset(Dataset):
    def __init__(self, qa_raw, image_dir, mask_dir=None,
                 ans2idx=None, img_size=224, max_seq_len=128):
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.ans2idx     = ans2idx
        self.max_seq_len = max_seq_len
        self.img_size    = img_size

        self.triplets = []
        for image_id, qa_pairs in qa_raw.items():
            for pair in qa_pairs:
                self.triplets.append({
                    'image_id': image_id,
                    'type':     pair['Type'],
                    'question': pair['Question'],
                    'answer':   str(pair['Answer'])
                })

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711])
        ])

        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
        ])

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item     = self.triplets[idx]
        img_path = os.path.join(self.image_dir, item['image_id'])

        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image)

        q_encoded = self.tokenizer(
            item['question'], padding='max_length',
            truncation=True, max_length=64, return_tensors='pt'
        )

        target_encoded = self.tokenizer(
            item['answer'], padding='max_length',
            truncation=True, max_length=self.max_seq_len, return_tensors='pt'
        )

        mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, item['image_id'])
            if os.path.exists(mask_path):
                m    = Image.open(mask_path)
                mask = torch.from_numpy(np.array(self.mask_transform(m))).long()

        ans_idx = self.ans2idx.get(item['answer'], 0) if self.ans2idx else 0

        return {
            'image_pixel_values': image,
            'question_input_ids': q_encoded['input_ids'].squeeze(0),
            'question_attn_mask': q_encoded['attention_mask'].squeeze(0),
            'decoder_input_ids':  target_encoded['input_ids'].squeeze(0),
            'decoder_labels':     target_encoded['input_ids'].squeeze(0).clone(),
            'relational_mask':    mask,
            'answer_class_idx':   torch.tensor(ans_idx, dtype=torch.long),
            'image_id':           item['image_id'],
            'raw_explanation':    item['answer']
        }

def load_qa_data():
    with open(f"{QA_DIR}/Train_QA.json") as f:
        train_qa_raw = json.load(f)
    with open(f"{QA_DIR}/Val_QA.json") as f:
        val_qa_raw = json.load(f)
    with open(f"{QA_DIR}/Test_QA.json") as f:
        test_qa_raw = json.load(f)
    return train_qa_raw, val_qa_raw, test_qa_raw

def initialize_dataloaders(batch_size=32, num_workers=4, train_fraction=0.9, val_fraction=0.9, test_fraction=0.6):
    train_qa_raw, val_qa_raw, test_qa_raw = load_qa_data()
    ans2idx, idx2ans = build_answer_vocab(train_qa_raw)

    train_dataset = EarthVQADataset(train_qa_raw, TRAIN_IMG_DIR, TRAIN_MASK_DIR, ans2idx, img_size=224)
    val_dataset   = EarthVQADataset(val_qa_raw,   VAL_IMG_DIR,   VAL_MASK_DIR,   ans2idx, img_size=224)
    test_dataset  = EarthVQADataset(test_qa_raw,  TEST_IMG_DIR,  None,           ans2idx, img_size=224)

    # Subset all splits
    def subset(dataset, fraction):
        size = int(len(dataset) * fraction)
        indices = torch.randperm(len(dataset))[:size].tolist()
        return torch.utils.data.Subset(dataset, indices)

    train_dataset = subset(train_dataset, train_fraction)
    val_dataset   = subset(val_dataset,   val_fraction)
    test_dataset  = subset(test_dataset,  test_fraction)

    loader_args = {
        "batch_size":         batch_size,
        "num_workers":        num_workers,
        "pin_memory":         True,
        "persistent_workers": True if num_workers > 0 else False,
        "prefetch_factor":    2 if num_workers > 0 else None
    }

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_args)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_args)
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size:   {len(val_loader.dataset)}")
    print(f"Test size:  {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader, ans2idx, idx2ans

if __name__ == "__main__":
    train_loader, val_loader, test_loader, ans2idx, idx2ans = initialize_dataloaders(
    batch_size=32, num_workers=6,
    train_fraction=0.9, val_fraction=0.9, test_fraction=0.9)
    print("Dataloaders initialized successfully!")