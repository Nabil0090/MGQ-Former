
import math
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import gc
import os

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'


@dataclass
class TrainConfig:
    vit_emb_dir     : str   = r"d:\amir lab\MGQ-Former\image_extract\outputs\vit_encoded\embeddings"
    text_emb_dir    : str   = r"d:\amir lab\MGQ-Former\text_extract\outputs\text_encoded\embeddings"
    qa_dir          : str   = r"C:\Users\Nabil\.cache\kagglehub\datasets\alienxc137\earthvqa-semantic-segmentation-visual-question-ans\versions\1\2024EarthVQA\2024EarthVQA"
    train_mask_dir  : str   = r"C:\Users\Nabil\.cache\kagglehub\datasets\alienxc137\earthvqa-semantic-segmentation-visual-question-ans\versions\1\Train-003\Train\masks_png"
    val_mask_dir    : str   = r"C:\Users\Nabil\.cache\kagglehub\datasets\alienxc137\earthvqa-semantic-segmentation-visual-question-ans\versions\1\Val-002\Val\masks_png"
    output_dir      : str   = "./outputs"
    resume          : str   = None
    batch_size      : int   = 16
    num_workers     : int   = 6
    num_queries     : int   = 48
    num_layers      : int   = 2
    num_heads       : int   = 8
    ffn_dim         : int   = 3072
    dropout         : float = 0.2
    num_classes     : int   = 147
    num_types       : int   = 6
    num_epochs      : int   = 10
    warmup_epochs   : int   = 3
    lr_qformer      : float = 1e-4
    lr_head         : float = 5e-4
    lr_mask         : float = 1e-4
    weight_decay    : float = 0.07
    lambda_type     : float = 0.3
    max_grad_norm   : float = 1.0
    label_smoothing : float = 0.1
    use_mask        : bool = True
    pooling         : str = "dual"


TYPE2IDX = {
    'Basic Counting':            0,
    'Basic Judging':             1,
    'Comprehensive Analysis':    2,
    'Object Situation Analysis': 3,
    'Reasoning-based Counting':  4,
    'Reasoning-based Judging':   5,
}
IDX2TYPE = {v: k for k, v in TYPE2IDX.items()}


def build_dataloaders(cfg: TrainConfig):
    from fusion_dataset import (
        ViTEmbeddingIndex, TextEmbeddingIndex,
        FusionDataset, fusion_collate_fn, build_answer_vocab
    )
    from torch.utils.data import DataLoader

    vit_dir  = Path(cfg.vit_emb_dir)
    text_dir = Path(cfg.text_emb_dir)

    qa_files = [
        f"{cfg.qa_dir}/Train_QA.json",
        f"{cfg.qa_dir}/Val_QA.json",
        f"{cfg.qa_dir}/Test_QA.json",
    ]
    ans2idx, idx2ans = build_answer_vocab(qa_files)
    print(f"Answer vocab: {len(ans2idx)} classes")

    print("Building ViT indices...")
    vit_train = ViTEmbeddingIndex(vit_dir, "train")
    vit_val   = ViTEmbeddingIndex(vit_dir, "val")

    print("Building Text indices...")
    text_train = TextEmbeddingIndex(text_dir, "train")
    text_val   = TextEmbeddingIndex(text_dir, "val")

    print("Building datasets...")
    train_ds = FusionDataset(
        f"{cfg.qa_dir}/Train_QA.json",
        cfg.train_mask_dir,
        vit_train, text_train, ans2idx
    )
    val_ds = FusionDataset(
        f"{cfg.qa_dir}/Val_QA.json",
        cfg.val_mask_dir,
        vit_val, text_val, ans2idx
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    loader_args = dict(
        batch_size         = cfg.batch_size,
        num_workers        = cfg.num_workers,
        pin_memory         = True,
        collate_fn         = fusion_collate_fn,
        persistent_workers = cfg.num_workers > 0,
        prefetch_factor    = 2 if cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_args)

    return train_loader, val_loader, ans2idx, idx2ans


def build_model_and_optimizer(cfg: TrainConfig, device: torch.device):
    from fusion_model import MaskedClassifier

    model = MaskedClassifier(
        num_classes = cfg.num_classes,
        num_types   = cfg.num_types,
        num_queries = cfg.num_queries,
        num_layers  = cfg.num_layers,
        num_heads   = cfg.num_heads,
        ffn_dim     = cfg.ffn_dim,
        dropout     = cfg.dropout,
        use_mask    = cfg.use_mask,
        pooling     = cfg.pooling,
    ).to(device)

    optimizer = optim.AdamW([
        {'params': model.mask_encoder.parameters(), 'lr': cfg.lr_mask},
        {'params': model.qformer.parameters(),      'lr': cfg.lr_qformer},
        {'params': model.answer_head.parameters(),  'lr': cfg.lr_head},
        {'params': model.type_head.parameters(),    'lr': cfg.lr_head},
    ], weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params:     {total:.1f}M")
    print(f"Trainable params: {trainable:.1f}M")

    return model, optimizer


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    total_steps  = cfg.num_epochs    * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    criterion_ans, criterion_type,
                    cfg: TrainConfig, device: torch.device, epoch: int):
    model.train()
    total_loss = 0.0
    total_ans  = 0.0
    total_type = 0.0
    correct    = 0
    total_n    = 0

    pbar = tqdm(total=len(loader), desc=f"Train E{epoch}", leave=True)

    for batch in loader:
        img_patches = batch['img_patches'].to(device, non_blocking=True)
        cls_embed   = batch['cls_embed'].to(device,   non_blocking=True)
        token_embed = batch['token_embed'].to(device,  non_blocking=True)
        attn_mask   = batch['attn_mask'].to(device,    non_blocking=True)
        mask        = batch['mask'].to(device,         non_blocking=True)
        ans_idx     = batch['ans_idx'].to(device,      non_blocking=True)
        type_idx    = batch['type_idx'].to(device,     non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            ans_logits, type_logits = model(
                img_patches, cls_embed, token_embed, mask, attn_mask
            )
            l_ans  = criterion_ans(ans_logits,   ans_idx)
            l_type = criterion_type(type_logits, type_idx)
            loss   = l_ans + cfg.lambda_type * l_type

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs          = ans_idx.size(0)
        total_loss += loss.item()   * bs
        total_ans  += l_ans.item()  * bs
        total_type += l_type.item() * bs
        preds       = ans_logits.argmax(dim=-1)
        correct    += (preds == ans_idx).sum().item()
        total_n    += bs

        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct/total_n:.3f}"
        )

    pbar.close()
    return {
        'loss':      total_loss / total_n,
        'loss_ans':  total_ans  / total_n,
        'loss_type': total_type / total_n,
        'acc':       correct    / total_n,
    }


@torch.no_grad()
def validate(model, loader, criterion_ans, criterion_type,
             cfg: TrainConfig, device: torch.device, epoch: int):
    model.eval()
    total_loss   = 0.0
    correct      = 0
    total_n      = 0
    type_correct = {t: 0 for t in TYPE2IDX}
    type_total   = {t: 0 for t in TYPE2IDX}

    pbar = tqdm(total=len(loader), desc=f"Val   E{epoch}", leave=True)

    for batch in loader:
        img_patches = batch['img_patches'].to(device, non_blocking=True)
        cls_embed   = batch['cls_embed'].to(device,   non_blocking=True)
        token_embed = batch['token_embed'].to(device,  non_blocking=True)
        attn_mask   = batch['attn_mask'].to(device,    non_blocking=True)
        mask        = batch['mask'].to(device,         non_blocking=True)
        ans_idx     = batch['ans_idx'].to(device,      non_blocking=True)
        type_idx    = batch['type_idx'].to(device,     non_blocking=True)
        q_types     = batch['q_types']

        with torch.amp.autocast('cuda'):
            ans_logits, type_logits = model(
                img_patches, cls_embed, token_embed, mask, attn_mask
            )
            l_ans  = criterion_ans(ans_logits,   ans_idx)
            l_type = criterion_type(type_logits, type_idx)
            loss   = l_ans + cfg.lambda_type * l_type

        bs          = ans_idx.size(0)
        total_loss += loss.item() * bs
        preds       = ans_logits.argmax(dim=-1)
        correct    += (preds == ans_idx).sum().item()
        total_n    += bs

        for i, qt in enumerate(q_types):
            type_total[qt]   += 1
            type_correct[qt] += int(preds[i].item() == ans_idx[i].item())

        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct/total_n:.3f}"
        )

    pbar.close()

    per_type_acc = {
        t: (type_correct[t] / type_total[t] if type_total[t] > 0 else 0.0)
        for t in TYPE2IDX
    }

    return {
        'loss':         total_loss / total_n,
        'acc':          correct    / total_n,
        'per_type_acc': per_type_acc,
    }


def save_checkpoint(model, optimizer, scheduler, epoch,
                    metrics: dict, cfg: TrainConfig,
                    ckpt_dir: Path, tag: str = None):
    tag       = tag or f"epoch_{epoch:02d}_acc_{metrics['val_acc']:.4f}"
    ckpt_path = ckpt_dir / f"{tag}.pt"
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics':              metrics,
        'config':               asdict(cfg),
    }, ckpt_path)
    return ckpt_path


def keep_top_k_checkpoints(ckpt_dir: Path, k: int = 3):
    all_ckpts = list(ckpt_dir.glob("epoch_*.pt"))
    if not all_ckpts:
        return

    # always protect the checkpoint with the highest epoch number (latest),
    # so a resume point is never deleted even if its val_acc isn't top-k
    def epoch_num(p: Path) -> int:
        return int(p.stem.split("_acc_")[0].split("_")[-1])

    latest_ckpt = max(all_ckpts, key=epoch_num)

    by_acc = sorted(
        all_ckpts,
        key=lambda p: float(p.stem.split("_acc_")[-1]),
        reverse=True
    )
    keep = set(by_acc[:k])
    keep.add(latest_ckpt)

    for ckpt in all_ckpts:
        if ckpt not in keep:
            ckpt.unlink()


def load_checkpoint(model, optimizer, scheduler,
                    ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    print(f"Resumed from epoch {ckpt['epoch']} | "
          f"val_acc: {ckpt['metrics'].get('val_acc', 0):.4f}")
    return ckpt['epoch']


def run_training(cfg: TrainConfig):
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_dir  = Path(cfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    log_dir  = out_dir / "logs"
    for d in [ckpt_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, ans2idx, idx2ans = build_dataloaders(cfg)
    cfg.num_classes = len(ans2idx)

    model, optimizer = build_model_and_optimizer(cfg, device)
    scaler           = torch.amp.GradScaler('cuda')
    scheduler        = build_scheduler(optimizer, cfg, len(train_loader))

    criterion_ans  = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    criterion_type = nn.CrossEntropyLoss()

    start_epoch  = 0
    best_val_acc = 0.0

    history_path = log_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        best_val_acc = max(e['val_acc'] for e in history)
        print(f"Loaded existing history: {len(history)} epochs")
        print(f"Restored best_val_acc: {best_val_acc:.4f}")
    else:
        history = []

    if cfg.resume:
        start_epoch = load_checkpoint(
            model, optimizer, scheduler, cfg.resume, device
        )
        if start_epoch >= cfg.num_epochs:
            raise ValueError(
                f"Resumed from epoch {start_epoch}, but cfg.num_epochs={cfg.num_epochs}. "
                f"This loop would run zero epochs (range({start_epoch+1}, {cfg.num_epochs+1}) is empty). "
                f"Set --num_epochs to a value greater than {start_epoch} "
                f"(e.g. {start_epoch + 10} for 10 more epochs)."
            )

    with open(log_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"\nStarting training: epoch {start_epoch+1} → {cfg.num_epochs}")
    print(f"Steps per epoch: {len(train_loader)}")
    print("-" * 70)

    for epoch in range(start_epoch + 1, cfg.num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, criterion_ans, criterion_type, cfg, device, epoch
        )
        val_metrics = validate(
            model, val_loader, criterion_ans, criterion_type,
            cfg, device, epoch
        )

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        epoch_log = {
            'epoch':           epoch,
            'train_loss':      train_metrics['loss'],
            'train_loss_ans':  train_metrics['loss_ans'],
            'train_loss_type': train_metrics['loss_type'],
            'train_acc':       train_metrics['acc'],
            'val_loss':        val_metrics['loss'],
            'val_acc':         val_metrics['acc'],
            'per_type_acc':    val_metrics['per_type_acc'],
            'lr':              lr_now,
            'elapsed_s':       elapsed,
        }
        history.append(epoch_log)

        with open(log_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(
            f"Epoch {epoch:02d}/{cfg.num_epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} acc: {train_metrics['acc']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} acc: {val_metrics['acc']:.4f} | "
            f"lr: {lr_now:.2e} | {elapsed:.0f}s"
        )
        print("  Per-type val acc:")
        for t, a in val_metrics['per_type_acc'].items():
            print(f"    {t:<30} {a:.4f}")
        print("-" * 70)

        metrics_for_ckpt = {
            'val_acc':   val_metrics['acc'],
            'val_loss':  val_metrics['loss'],
            'train_acc': train_metrics['acc'],
        }

        save_checkpoint(
            model, optimizer, scheduler, epoch,
            metrics_for_ckpt, cfg, ckpt_dir
        )
        keep_top_k_checkpoints(ckpt_dir, k=3)

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                metrics_for_ckpt, cfg, ckpt_dir, tag="best_model"
            )
            print(f"  *** New best model saved! val_acc: {best_val_acc:.4f} ***")

        gc.collect()
        torch.cuda.empty_cache()

    print("\nTraining complete.")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Logs saved to: {log_dir}")
    print(f"Checkpoints:   {ckpt_dir}")
    return history