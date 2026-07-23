import json
import shutil
from pathlib import Path

OUTPUTS_LOGS = Path(r"D:\amir lab\MGQ-Former\fusion\outputs\logs")
ABLATION_ROOT = Path(r"D:\amir lab\MGQ-Former\fusion\ablation_logs")

A1_DIR = ABLATION_ROOT / "A1_no_mask_mean_pool"
A1_DIR.mkdir(parents=True, exist_ok=True)

history_path = OUTPUTS_LOGS / "history.json"
config_path  = OUTPUTS_LOGS / "config.json"

with open(history_path) as f:
    full_history = json.load(f)

print(f"Total entries in merged history.json: {len(full_history)}")

ORIGINAL_MODEL_EPOCH_COUNT = 12
A1_EPOCH_COUNT = 10

if len(full_history) != ORIGINAL_MODEL_EPOCH_COUNT + A1_EPOCH_COUNT:
    print(f"WARNING: expected {ORIGINAL_MODEL_EPOCH_COUNT + A1_EPOCH_COUNT} entries, "
          f"found {len(full_history)}. Double-check before trusting this split.")

original_history = full_history[:ORIGINAL_MODEL_EPOCH_COUNT]
a1_history        = full_history[ORIGINAL_MODEL_EPOCH_COUNT:]

print(f"Original model entries: {len(original_history)}")
print(f"A1 entries:             {len(a1_history)}")

a1_best = max(a1_history, key=lambda e: e['val_acc'])
print(f"\nA1 best epoch (within its own 10): epoch {a1_best['epoch']}, "
      f"val_acc {a1_best['val_acc']:.4f}")
print("A1 best per-type accuracy:")
for t, a in a1_best['per_type_acc'].items():
    print(f"  {t:<30} {a:.4f}")

with open(A1_DIR / "history.json", "w") as f:
    json.dump(a1_history, f, indent=2)
print(f"\nSaved A1 history -> {A1_DIR / 'history.json'}")

if config_path.exists():
    shutil.copy(config_path, A1_DIR / "config.json")
    print(f"Copied A1 config  -> {A1_DIR / 'config.json'}")

with open(history_path, "w") as f:
    json.dump(original_history, f, indent=2)
print(f"\nRestored original model's history.json to its true {len(original_history)} entries.")
print("outputs/logs/history.json is now safe again for the original model's record.")

original_best = max(original_history, key=lambda e: e['val_acc'])
print(f"\nSanity check — original model best: epoch {original_best['epoch']}, "
      f"val_acc {original_best['val_acc']:.4f}")