"""
Step2_pretrain_graph.py — GraphMACCSEncoder 사전학습 (Stage 1)

Step1_preprocess.py의 Step 1-B에서 저장한 {train,test}_graphs.pt 를 사용.
학습 전략은 기존 Step2_pretrain.py와 동일:
  - val_AUC 기반 early stopping (mode=max, patience=30)
  - ReduceLROnPlateau(mode=max)
  - 차등 학습률: ChemBERTa lr=1e-4, 나머지 lr=3e-4
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch_geometric.data import Batch as PyGBatch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from config import (
    K, D_MODEL, NUM_HEADS, DROPOUT, MODEL_NAME,
    BATCH_SIZE, MAX_LENGTH, SEED, EPOCHS, PATIENCE,
    LR_CHEM, LR_OTHER, WEIGHT_DECAY,
    SCHED_PATIENCE, SCHED_FACTOR, SCHED_MIN_LR,
    DATA_DIR, OUT_DIR,
    MACCS_DIM, MAX_ATOMS, GINE_LAYERS, GINE_HIDDEN, ATOM_FEAT_DIM, BOND_FEAT_DIM,
    N_AUG,
)
from utils import set_seed
from model import GraphMACCSEncoder
from graph_utils import augment_smiles, smiles_to_pyg

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("Step 2 (Graph): GraphMACCSEncoder Pre-training (Stage 1)")
print(f"  ChemBERTa last-layer lr={LR_CHEM}  |  Graph/DiffAttn lr={LR_OTHER}")
print(f"  k={K}, d_model={D_MODEL}, num_heads={NUM_HEADS}, max_atoms={MAX_ATOMS}")
print(f"  epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}, early_stop=val_AUC")
print("=" * 65)

set_seed(SEED)


# ── Dataset ──────────────────────────────────────────────────────────────────

class MolGraphDataset(Dataset):
    """
    Step1-B에서 저장된 {split}_graphs.pt 로드.
    각 항목: {pyg, maccs, label, smiles}

    __getitem__ 반환:
        input_ids    : (MAX_LENGTH,)
        attn_mask    : (MAX_LENGTH,)
        maccs        : (167,) float32
        pyg_data     : torch_geometric.data.Data
        label        : scalar float32
    """

    def __init__(self, data_list: list, tokenizer, max_length: int, cache_path: str | None = None):
        self.data_list = data_list
        self.max_length = max_length

        smiles_list = [d["smiles"] for d in data_list]

        if cache_path and os.path.exists(cache_path):
            print(f"  [Cache] 토큰 로딩: {cache_path}")
            enc = torch.load(cache_path, weights_only=False)
        else:
            print(f"  [Tokenize] {len(smiles_list)}개 토크나이징 중...")
            enc = tokenizer(
                smiles_list,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            if cache_path:
                torch.save({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}, cache_path)
                print(f"  [Cache] 저장: {cache_path}")

        self.input_ids      = enc["input_ids"]       # (N, MAX_LENGTH)
        self.attention_mask = enc["attention_mask"]  # (N, MAX_LENGTH)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d = self.data_list[idx]
        return {
            "input_ids":  self.input_ids[idx],
            "attn_mask":  self.attention_mask[idx],
            "maccs":      d["maccs"],       # (167,)
            "pyg_data":   d["pyg"],
            "label":      d["label"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """PyG Data 리스트를 Batch로 병합."""
    return {
        "input_ids":  torch.stack([b["input_ids"] for b in batch]),   # (B, MAX_LENGTH)
        "attn_mask":  torch.stack([b["attn_mask"] for b in batch]),
        "maccs":      torch.stack([b["maccs"] for b in batch]),        # (B, 167)
        "graph":      PyGBatch.from_data_list([b["pyg_data"] for b in batch]),
        "label":      torch.stack([b["label"] for b in batch]),        # (B,)
    }


def augment_train_subset(data_list: list, n_aug: int, seed: int) -> list:
    """
    Split된 train 서브셋에만 SMILES 증강 적용.
    Val 서브셋은 이 함수를 호출하지 않으므로 Data Leakage 없음.

    Args:
        data_list: Split 이후의 train 서브셋 [{pyg, maccs, label, smiles}, ...]
        n_aug:     증강 개수 (config.N_AUG)
        seed:      재현성용 시드

    Returns:
        원본 + 증강 합산 리스트 (원본 순서 유지, 증강 후 추가)
        MACCS는 분자 동일 → 원본 maccs 재사용 (재계산 생략으로 속도 향상)
    """
    if n_aug == 0:
        return data_list

    augmented = list(data_list)  # 원본 유지
    n_skip = 0
    for d in data_list:
        for rand_smi in augment_smiles(d["smiles"], n_aug=n_aug, seed=seed):
            pyg = smiles_to_pyg(rand_smi)
            if pyg is None:
                n_skip += 1
                continue  # 파싱 실패 스킵 (희귀 케이스)
            augmented.append({
                "pyg":    pyg,
                "maccs":  d["maccs"],   # 동일 분자 → 원본 MACCS 재사용
                "label":  d["label"],
                "smiles": rand_smi,     # 실제 학습 SMILES (랜덤)
            })
    if n_skip:
        print(f"  [Aug] 파싱 실패 스킵: {n_skip}개")
    return augmented


# ── 데이터 로드 ──────────────────────────────────────────────────────────────

train_cache_path = os.path.join(DATA_DIR, "train_graphs.pt")
assert os.path.exists(train_cache_path), (
    f"Missing: {train_cache_path}\nStep 1-B를 먼저 실행하세요: python Step1_preprocess.py"
)

all_data = torch.load(train_cache_path, weights_only=False)
all_labels = np.array([d["label"].item() for d in all_data])
n_pos = int(all_labels.sum()); n_neg = len(all_labels) - n_pos
print(f"Train cache: {len(all_data)}개  |  pos={n_pos}, neg={n_neg}")

# Train / Val 분리
idx = np.arange(len(all_data))
idx_tr, idx_val = train_test_split(
    idx, test_size=0.15, stratify=all_labels.astype(int), random_state=SEED
)
print(f"  train subset: {len(idx_tr)}, val subset: {len(idx_val)}")

# ── SMILES 오프라인 증강 (Train Subset Only — 데이터 누수 방지) ──────────────
# split 먼저 → 증강: Val/Test는 일절 건드리지 않음.
# 캐시 경로에 N_AUG 포함 → 값 변경 시 자동 재생성.
train_subset_orig = [all_data[i] for i in idx_tr]
val_data          = [all_data[i] for i in idx_val]

# 누수 검증 (split 후, 증강 전 원본 SMILES 기준)
_train_orig_smiles = {d["smiles"] for d in train_subset_orig}
_val_smiles        = {d["smiles"] for d in val_data}
_overlap = _train_orig_smiles & _val_smiles
assert len(_overlap) == 0, f"Leakage detected: {len(_overlap)} SMILES overlap"
print(f"[OK] No data leakage (train_orig={len(_train_orig_smiles)}, val={len(_val_smiles)})")

if N_AUG > 0:
    aug_cache_path = os.path.join(DATA_DIR, f"train_graphs_aug_{N_AUG}x_{SEED}.pt")
    if os.path.exists(aug_cache_path):
        print(f"  [Aug Cache] 로딩: {aug_cache_path}")
        train_data = torch.load(aug_cache_path, weights_only=False)
    else:
        print(f"\n[Augment] N_AUG={N_AUG} → train {len(train_subset_orig)}개 × 최대 {N_AUG+1}배...")
        train_data = augment_train_subset(train_subset_orig, n_aug=N_AUG, seed=SEED)
        torch.save(train_data, aug_cache_path)
        print(f"  [Aug Cache] 저장: {aug_cache_path}  ({len(train_data)}개)")
    print(f"  augmented train: {len(train_subset_orig)} → {len(train_data)}개")
    print(f"  val (원본 유지): {len(val_data)}개")
else:
    train_data = train_subset_orig
    print(f"  N_AUG=0: 증강 없음 (train={len(train_data)}, val={len(val_data)})")

# pos_weight를 증강 후 train 기준으로 재계산 (val 포함 전체에서 계산하면 부정확)
_train_labels = np.array([d["label"].item() for d in train_data])
n_pos = int(_train_labels.sum()); n_neg = len(_train_labels) - n_pos
print(f"  최종 train: {len(train_data)}개  |  pos={n_pos}, neg={n_neg}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# N_AUG가 달라지면 토큰 캐시도 자동으로 다른 파일 사용
# 파일명 형식: graph_train_tokens_42.pt (N_AUG=0) / graph_train_tokens_42_aug2.pt (N_AUG=2)
_aug_suffix = f"_aug{N_AUG}" if N_AUG > 0 else ""
_train_token_cache = os.path.join(DATA_DIR, f"graph_train_tokens_{SEED}{_aug_suffix}.pt")

train_ds = MolGraphDataset(
    train_data, tokenizer, MAX_LENGTH,
    cache_path=_train_token_cache,
)
val_ds = MolGraphDataset(
    val_data, tokenizer, MAX_LENGTH,
    cache_path=os.path.join(DATA_DIR, f"graph_val_tokens_{SEED}.pt"),
)

train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, collate_fn=collate_fn,
)
val_dl = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, collate_fn=collate_fn,
)

# ── 모델 ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = GraphMACCSEncoder(
    atom_feat_dim=ATOM_FEAT_DIM,
    maccs_dim=MACCS_DIM,
    bond_feat_dim=BOND_FEAT_DIM,
    gine_hidden=GINE_HIDDEN,
    gine_layers=GINE_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    k=K,
    max_atoms=MAX_ATOMS,
    dropout=DROPOUT,
    model_name=MODEL_NAME,
).to(device)

trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total     = sum(p.numel() for p in encoder.parameters())
print(f"Parameters: trainable={trainable:,} / total={total:,}  |  Device: {device}")

# ── 차등 학습률 ───────────────────────────────────────────────────────────────

n_layers = len(encoder.chemberta.encoder.layer)
chem_last_ids = {id(p) for p in encoder.chemberta.encoder.layer[n_layers - 1].parameters()}
chem_all_ids  = {id(p) for p in encoder.chemberta.parameters()}

optimizer = torch.optim.AdamW(
    [
        {"params": [p for p in encoder.parameters() if id(p) in chem_last_ids],   "lr": LR_CHEM},
        {"params": [p for p in encoder.parameters() if id(p) not in chem_all_ids and id(p) not in chem_last_ids], "lr": LR_OTHER},
    ],
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=SCHED_FACTOR,
    patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR,
)
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ── 추론 헬퍼 ─────────────────────────────────────────────────────────────────

def run_inference(model, dl):
    model.eval()
    all_logits, all_proba = [], []
    with torch.no_grad():
        for batch in dl:
            ids  = batch["input_ids"].to(device)
            mask = batch["attn_mask"].to(device)
            mac  = batch["maccs"].to(device)
            grph = batch["graph"].to(device)
            logits = model(ids, mask, mac, grph).squeeze(1)
            all_logits.append(logits.cpu())
            all_proba.append(torch.sigmoid(logits).cpu().numpy())
    return torch.cat(all_logits), np.concatenate(all_proba)


# ── 학습 루프 ─────────────────────────────────────────────────────────────────

best_val_auc, best_state, patience_cnt = 0.0, None, 0
y_val_np = np.array([d["label"].item() for d in val_data])
y_val_t  = torch.tensor(y_val_np, dtype=torch.float32)

for epoch in range(1, EPOCHS + 1):
    encoder.train()
    epoch_loss = 0.0

    for batch in train_dl:
        ids  = batch["input_ids"].to(device)
        mask = batch["attn_mask"].to(device)
        mac  = batch["maccs"].to(device)
        grph = batch["graph"].to(device)
        y_b  = batch["label"].to(device)

        optimizer.zero_grad()
        loss = criterion(encoder(ids, mask, mac, grph).squeeze(1), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(y_b)

    epoch_loss /= len(train_ds)

    val_logits, val_proba = run_inference(encoder, val_dl)
    val_loss = criterion(val_logits.to(device), y_val_t.to(device)).item()
    val_auc  = roc_auc_score(y_val_np, val_proba)

    scheduler.step(val_auc)

    if epoch % 5 == 0 or epoch == 1:
        cur_lr    = optimizer.param_groups[0]["lr"]
        best_mark = " * best" if val_auc > best_val_auc else ""
        print(
            f"  Epoch {epoch:>4d} | train_loss={epoch_loss:.4f}"
            f" | val_loss={val_loss:.4f} | val_AUC={val_auc:.4f}"
            f" | lr={cur_lr:.2e}{best_mark}"
        )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state   = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} (best val_AUC={best_val_auc:.4f})")
            break

save_path = os.path.join(OUT_DIR, "pretrained_graph_encoder.pt")
torch.save(best_state, save_path)

print(f"\nBest val AUC: {best_val_auc:.4f}")
print(f"Saved: {save_path}")
print("Step 2 (Graph) OK")
