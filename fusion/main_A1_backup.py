import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataclasses import asdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="MaskQFormer — Mask-Guided Q-Former for RS-VQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Paths ──────────────────────────────────────────────────────────
    parser.add_argument("--vit_emb_dir", type=str,
        default=r"d:\amir lab\MGQ-Former\image_extract\outputs\vit_encoded\embeddings",
        help="Path to ViT embedding chunks directory")
    parser.add_argument("--text_emb_dir", type=str,
        default=r"d:\amir lab\MGQ-Former\text_extract\outputs\text_encoded\embeddings",
        help="Path to RoBERTa embedding chunks directory")
    parser.add_argument("--qa_dir", type=str,
        default=r"C:\Users\Nabil\.cache\kagglehub\datasets\alienxc137\earthvqa-semantic-segmentation-visual-question-ans\versions\1\2024EarthVQA\2024EarthVQA",
        help="Path to EarthVQA QA json directory")
    parser.add_argument("--train_mask_dir", type=str,
        default=r"C:\Users\Nabil\.cache\kagglehub\datasets\alienxc137\earthvqa-semantic-segmentation-visual-question-ans\versions\1\Train-003\Train\masks_png",
        help="Path to training masks directory")
    parser.add_argument("--val_mask_dir", type=str,
        default=r"C:\Users\Nabil\.cache\kagglehub\datasets\alienxc137\earthvqa-semantic-segmentation-visual-question-ans\versions\1\Val-002\Val\masks_png",
        help="Path to validation masks directory")
    parser.add_argument("--output_dir", type=str,
        default="./outputs",
        help="Directory to save checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint to resume from")

    # ── Dataloader ─────────────────────────────────────────────────────
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--num_workers", type=int,   default=4)

    # ── Model ──────────────────────────────────────────────────────────
    parser.add_argument("--num_queries", type=int,   default=48)
    parser.add_argument("--num_layers",  type=int,   default=2)
    parser.add_argument("--num_heads",   type=int,   default=8)
    parser.add_argument("--ffn_dim",     type=int,   default=3072)
    parser.add_argument("--dropout",     type=float, default=0.2)

    # ── Training ───────────────────────────────────────────────────────
    parser.add_argument("--num_epochs",    type=int,   default=10)
    parser.add_argument("--warmup_epochs", type=int,   default=2)
    parser.add_argument("--lr_qformer",    type=float, default=1e-4)
    parser.add_argument("--lr_head",       type=float, default=5e-4)
    parser.add_argument("--lr_mask",       type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=0.07)
    parser.add_argument("--lambda_type",   type=float, default=0.3,
        help="Weight for question-type auxiliary loss")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_mask", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--pooling", type=str, default="dual", choices=["dual", "mean"])

    return parser.parse_args()


def main():
    args = parse_args()

    # import here so path is resolved after sys.path append
    from training import TrainConfig, run_training

    cfg = TrainConfig(
        vit_emb_dir     = args.vit_emb_dir,
        text_emb_dir    = args.text_emb_dir,
        qa_dir          = args.qa_dir,
        train_mask_dir  = args.train_mask_dir,
        val_mask_dir    = args.val_mask_dir,
        output_dir      = args.output_dir,
        resume          = args.resume,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        num_queries     = args.num_queries,
        num_layers      = args.num_layers,
        num_heads       = args.num_heads,
        ffn_dim         = args.ffn_dim,
        dropout         = args.dropout,
        num_epochs      = args.num_epochs,
        warmup_epochs   = args.warmup_epochs,
        lr_qformer      = args.lr_qformer,
        lr_head         = args.lr_head,
        lr_mask         = args.lr_mask,
        weight_decay    = args.weight_decay,
        lambda_type     = args.lambda_type,
        max_grad_norm   = args.max_grad_norm,
        label_smoothing = args.label_smoothing,
        use_mask       = args.use_mask,
        pooling        = args.pooling,
    )

    print("=" * 70)
    print("  MaskQFormer — Mask-Guided Q-Former for RS-VQA")
    print("=" * 70)
    print("Config:")
    for k, v in asdict(cfg).items():
        print(f"  {k:<20} {v}")
    print("=" * 70)

    run_training(cfg)


if __name__ == "__main__":
    main()