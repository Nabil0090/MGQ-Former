import json
import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

BASE         = r"C:\Users\Nabil\.cache\kagglehub\datasets\alienxc137\earthvqa-semantic-segmentation-visual-question-ans\versions\1"
QA_DIR       = f"{BASE}/2024EarthVQA/2024EarthVQA"
TRAIN_MASK_DIR = f"{BASE}/Train-003/Train/masks_png"
VAL_MASK_DIR   = f"{BASE}/Val-002/Val/masks_png"

VIT_EMB_DIR  = Path(r"D:\amir lab\RS-Vamba-main\image_extract\outputs\vit_encoded\embeddings")
TEXT_EMB_DIR = Path(r"D:\amir lab\RS-Vamba-main\text_extract\outputs\text_encoded\embeddings")

TYPE2IDX = {
    'Basic Counting':           0,
    'Basic Judging':            1,
    'Comprehensive Analysis':   2,
    'Object Situation Analysis':3,
    'Reasoning-based Counting': 4,
    'Reasoning-based Judging':  5,
}
NUM_QUESTION_TYPES = len(TYPE2IDX)


class ViTEmbeddingIndex:
    def __init__(self, emb_dir: Path, split: str):
        self.index = {}
        pattern = f"{split}*.pt" if split == "train" else f"{split}_chunk_*.pt"
        chunk_files = sorted(emb_dir.glob(pattern))
        print(f"  [ViT-{split}] Indexing {len(chunk_files)} chunk files...")
        for cf in chunk_files:
            data = torch.load(cf, map_location='cpu', weights_only=False)
            for img_id in data.keys():
                self.index[img_id] = cf
            del data
        print(f"  [ViT-{split}] Indexed {len(self.index)} unique images")
        self._cache_file = None
        self._cache_data = None

    def get(self, img_id: str) -> torch.Tensor:
        cf = self.index.get(img_id)
        if cf is None:
            raise KeyError(f"image_id '{img_id}' not found in ViT index")
        if self._cache_file != cf:
            self._cache_data = torch.load(cf, map_location='cpu', weights_only=False)
            self._cache_file = cf
        emb = self._cache_data[img_id]
        if emb.dim() == 3:
            emb = emb.flatten(1).transpose(0, 1)
        return emb.float()


class TextEmbeddingIndex:
    def __init__(self, emb_dir: Path, split: str):
        self.entries = []
        chunk_files = sorted(emb_dir.glob(f"{split}_chunk_*.pt"))
        print(f"  [Text-{split}] Indexing {len(chunk_files)} chunk files...")
        for cf in chunk_files:
            data = torch.load(cf, map_location='cpu', weights_only=False)
            for key, val in data.items():
                self.entries.append({
                    'key':      key,
                    'image_id': val['image_id'],
                    'answer':   val['answer'],
                    'seq_len':  val.get('seq_len', val['token_embed'].shape[0]),
                    'chunk':    cf,
                })
            del data
        print(f"  [Text-{split}] Indexed {len(self.entries)} QA pairs")
        self._cache_file = None
        self._cache_data = None

    def __len__(self):
        return len(self.entries)

    def get(self, idx: int):
        entry = self.entries[idx]
        cf    = entry['chunk']
        if self._cache_file != cf:
            self._cache_data = torch.load(cf, map_location='cpu', weights_only=False)
            self._cache_file = cf
        item = self._cache_data[entry['key']]
        return {
            'image_id':    entry['image_id'],
            'answer':      entry['answer'],
            'cls_embed':   item['cls_embed'].float(),
            'token_embed': item['token_embed'].float(),
            'seq_len':     entry['seq_len'],
        }


def build_answer_vocab(qa_files: list):
    answers = set()
    for qa_file in qa_files:
        with open(qa_file) as f:
            qa_raw = json.load(f)
        for qa_pairs in qa_raw.values():
            for pair in qa_pairs:
                answers.add(str(pair['Answer']))
    ans2idx = {a: i for i, a in enumerate(sorted(answers))}
    idx2ans = {i: a for a, i in ans2idx.items()}
    return ans2idx, idx2ans


class FusionDataset(Dataset):
    def __init__(
        self,
        qa_json_path: str,
        mask_dir: str,
        vit_index: ViTEmbeddingIndex,
        text_index: TextEmbeddingIndex,
        ans2idx: dict,
        img_size: int = 224,
    ):
        self.mask_dir  = mask_dir
        self.vit_index = vit_index
        self.text_index = text_index
        self.ans2idx   = ans2idx
        self.img_size  = img_size

        with open(qa_json_path) as f:
            qa_raw = json.load(f)

        self.qa_lookup = {}
        for image_id, qa_pairs in qa_raw.items():
            for pair in qa_pairs:
                self.qa_lookup[f"{image_id}_{pair['Question']}"] = pair

        # align text_index entries with qa_raw to get question type
        self.triplets = []
        for i in range(len(text_index)):
            entry    = text_index.entries[i]
            img_id   = entry['image_id']
            answer   = entry['answer']
            qa_pairs = qa_raw.get(img_id, [])
            q_type   = 'Basic Judging'
            for pair in qa_pairs:
                if str(pair['Answer']) == answer:
                    q_type = pair['Type']
                    break
            self.triplets.append({
                'text_idx': i,
                'image_id': img_id,
                'answer':   answer,
                'type':     q_type,
            })

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item     = self.triplets[idx]
        img_id   = item['image_id']
        answer   = item['answer']
        q_type   = item['type']

        img_patches = self.vit_index.get(img_id)

        text_data   = self.text_index.get(item['text_idx'])
        cls_embed   = text_data['cls_embed']
        token_embed = text_data['token_embed']
        seq_len     = text_data['seq_len']

        mask = torch.zeros(self.img_size, self.img_size, dtype=torch.float32)
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, img_id)
            if os.path.exists(mask_path):
                m    = Image.open(mask_path).resize(
                    (self.img_size, self.img_size), Image.NEAREST
                )
                mask = torch.from_numpy(np.array(m)).float()
                mask = mask / (mask.max() + 1e-6)

        ans_idx  = self.ans2idx.get(answer, 0)
        type_idx = TYPE2IDX.get(q_type, 0)

        return {
            'img_patches': img_patches,
            'cls_embed':   cls_embed,
            'token_embed': token_embed,
            'seq_len':     seq_len,
            'mask':        mask,
            'ans_idx':     torch.tensor(ans_idx,  dtype=torch.long),
            'type_idx':    torch.tensor(type_idx, dtype=torch.long),
            'image_id':    img_id,
            'answer':      answer,
            'q_type':      q_type,
        }


def fusion_collate_fn(batch):
    img_patches = torch.stack([b['img_patches'] for b in batch])
    cls_embeds  = torch.stack([b['cls_embed']   for b in batch])
    masks       = torch.stack([b['mask']         for b in batch])
    ans_idxs    = torch.stack([b['ans_idx']      for b in batch])
    type_idxs   = torch.stack([b['type_idx']     for b in batch])

    token_embeds = [b['token_embed'] for b in batch]
    seq_lens     = [b['seq_len']     for b in batch]
    max_len      = max(seq_lens)

    padded_tokens = pad_sequence(token_embeds, batch_first=True, padding_value=0.0)

    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, sl in enumerate(seq_lens):
        attn_mask[i, :sl] = True

    return {
        'img_patches':   img_patches,
        'cls_embed':     cls_embeds,
        'token_embed':   padded_tokens,
        'attn_mask':     attn_mask,
        'mask':          masks,
        'ans_idx':       ans_idxs,
        'type_idx':      type_idxs,
        'image_ids':     [b['image_id'] for b in batch],
        'answers':       [b['answer']   for b in batch],
        'q_types':       [b['q_type']   for b in batch],
    }


def build_fusion_dataloaders(batch_size=16, num_workers=2):
    qa_files = [
        f"{QA_DIR}/Train_QA.json",
        f"{QA_DIR}/Val_QA.json",
        f"{QA_DIR}/Test_QA.json",
    ]
    ans2idx, idx2ans = build_answer_vocab(qa_files)
    print(f"Answer vocab size: {len(ans2idx)}")
    print(f"Question types:    {NUM_QUESTION_TYPES}")

    print("\nBuilding ViT indices...")
    vit_train = ViTEmbeddingIndex(VIT_EMB_DIR, "train")
    vit_val   = ViTEmbeddingIndex(VIT_EMB_DIR, "val")
    vit_test  = ViTEmbeddingIndex(VIT_EMB_DIR, "test")

    print("\nBuilding Text indices...")
    text_train = TextEmbeddingIndex(TEXT_EMB_DIR, "train")
    text_val   = TextEmbeddingIndex(TEXT_EMB_DIR, "val")
    text_test  = TextEmbeddingIndex(TEXT_EMB_DIR, "test")

    print("\nBuilding datasets...")
    train_ds = FusionDataset(
        f"{QA_DIR}/Train_QA.json", TRAIN_MASK_DIR,
        vit_train, text_train, ans2idx
    )
    val_ds = FusionDataset(
        f"{QA_DIR}/Val_QA.json", VAL_MASK_DIR,
        vit_val, text_val, ans2idx
    )
    test_ds = FusionDataset(
        f"{QA_DIR}/Test_QA.json", None,
        vit_test, text_test, ans2idx
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val:   {len(val_ds)}")
    print(f"  Test:  {len(test_ds)}")

    loader_args = dict(
        batch_size      = batch_size,
        num_workers     = num_workers,
        pin_memory      = True,
        collate_fn      = fusion_collate_fn,
        persistent_workers = num_workers > 0,
        prefetch_factor = 2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_args)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader, ans2idx, idx2ans


if __name__ == "__main__":
    train_loader, val_loader, test_loader, ans2idx, idx2ans = build_fusion_dataloaders(
        batch_size=4, num_workers=0
    )

    print("\nSanity check — first train batch:")
    batch = next(iter(train_loader))
    print(f"  img_patches : {batch['img_patches'].shape}")
    print(f"  cls_embed   : {batch['cls_embed'].shape}")
    print(f"  token_embed : {batch['token_embed'].shape}")
    print(f"  attn_mask   : {batch['attn_mask'].shape}")
    print(f"  mask        : {batch['mask'].shape}")
    print(f"  ans_idx     : {batch['ans_idx'].shape}")
    print(f"  type_idx    : {batch['type_idx'].shape}")
    print(f"  q_types     : {batch['q_types']}")
    print(f"  answers     : {batch['answers']}")
    print("\nPhase 1 complete. Ready for Phase 2 (Q-Former model).")
