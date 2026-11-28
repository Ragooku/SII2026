"""
k-fold кросс-валидация для датасета Garbage Classification (фолды берутся из splits.csv).

Схема:
  * test остаётся нетронутым и одинаковым для всех фолдов;
  * остальные данные уже поделены split_data.py на фолды (колонка fold в splits.csv);
  * на каждом фолде: обучение на k-1 частях, валидация на оставшейся (ранняя остановка по val F1),
    затем оценка лучшей модели на test;
  * в конце: среднее ± std по фолдам и метрики ансамбля (усреднение вероятностей фолдов) на test.

ЛР2 (аугментации): метод выбирается константой EXPERIMENT (см. словарь EXPERIMENTS ниже).
Для каждого эксперимента запускайте скрипт отдельно, результаты сравнивает compare_experiments.py.

Запуск без аргументов:  python train_garbage_cv.py
Настройки — в блоке ниже. Файл самодостаточный.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ---------- настройки ----------
SPLITS_CSV = "splits.csv"               # создаётся split_data.py
MODEL = "resnet"                        # "simple" или "resnet"
EXPERIMENT = "flip"                 # см. EXPERIMENTS: baseline, flip, geom, geom_color, randaug, mixup, cutmix, full
RUN = f"{MODEL}_{EXPERIMENT}"           # префикс имён файлов этого запуска
IMG_SIZE = 128                          # изображения разного размера -> приводим к квадрату
EPOCHS, BATCH, LR, WD, WORKERS = 60, 64, 1e-3, 1e-2, 4   # EPOCHS - максимум
PATIENCE = 8                            # ранняя остановка: эпох без роста val F1
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- аугментации (ЛР2) ----------
# Преобразования над PIL-изображением (применяются только к train). Все они сохраняют класс объекта.
_resize = [T.Resize((IMG_SIZE, IMG_SIZE))]
_crop = [T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0))]      # случайный масштаб и положение объекта
_flip = [T.RandomHorizontalFlip()]                              # зеркальное отражение
_rot = [T.RandomRotation(15)]                                   # небольшой поворот
_color = [T.ColorJitter(0.3, 0.3, 0.3, 0.05)]                   # яркость, контраст, насыщенность, оттенок
_randaug = [T.RandAugment(num_ops=2, magnitude=9)]              # случайные операции из набора

# aug  - список PIL-преобразований; mix - "mixup"/"cutmix" на уровне батча; smooth - label smoothing
EXPERIMENTS = {
    "baseline":   dict(aug=_resize,                               mix=None,     smooth=0.0),   # ЛР1
    "flip":       dict(aug=_resize + _flip,                       mix=None,     smooth=0.0),
    "geom":       dict(aug=_crop + _flip + _rot,                  mix=None,     smooth=0.0),
    "geom_color": dict(aug=_crop + _flip + _rot + _color,         mix=None,     smooth=0.0),
    "randaug":    dict(aug=_crop + _flip + _randaug,              mix=None,     smooth=0.0),
    "mixup":      dict(aug=_crop + _flip + _rot + _color,         mix="mixup",  smooth=0.0),
    "cutmix":     dict(aug=_crop + _flip + _rot + _color,         mix="cutmix", smooth=0.0),
    "full":       dict(aug=_crop + _flip + _rot + _color,         mix="cutmix", smooth=0.1),
}
if EXPERIMENT not in EXPERIMENTS:
    raise SystemExit(f"Неизвестный EXPERIMENT '{EXPERIMENT}'. Доступно: {list(EXPERIMENTS)}")
CFG = EXPERIMENTS[EXPERIMENT]
MIX_ALPHA = 1.0            # параметр Beta-распределения для mixup/cutmix

_norm = [T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)]
train_tf = T.Compose(CFG["aug"] + _norm)
eval_tf = T.Compose(_resize + _norm)                            # val/test всегда без аугментаций


# ---------- данные ----------
class SamplesDataset(Dataset):
    """Датасет по списку (путь, индекс класса)."""

    def __init__(self, samples, tf):
        self.samples, self.tf = samples, tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.tf(Image.open(path).convert("RGB")), int(label)


def load_samples():
    """Читает splits.csv: пул (train+val) с номерами фолдов и общий test."""
    df = pd.read_csv(SPLITS_CSV)
    classes = sorted(df["label"].unique())
    idx = {k: i for i, k in enumerate(classes)}
    pool_df, test_df = df[df["fold"] >= 0], df[df["fold"] < 0]
    pool = list(zip(pool_df["path"], pool_df["label"].map(idx)))
    test = list(zip(test_df["path"], test_df["label"].map(idx)))
    return pool, pool_df["fold"].to_numpy(), test, classes


def make_loader(ds, shuffle=False):
    return DataLoader(ds, BATCH, shuffle=shuffle, drop_last=shuffle,
                      num_workers=WORKERS, pin_memory=True, persistent_workers=WORKERS > 0)


# ---------- модели (с нуля) ----------
def conv_bn(c_in, c_out, stride=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        w, layers, c = (32, 64, 128, 256), [], 3
        for o in w:
            layers += [conv_bn(c, o), conv_bn(o, o), nn.MaxPool2d(2)]
            c = o
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Dropout(0.3), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1, self.conv2 = conv_bn(c_in, c_out, stride), nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out))
        self.skip = nn.Identity() if stride == 1 and c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return torch.relu(self.conv2(self.conv1(x)) + self.skip(x))


class ResNetSmall(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        layers, c = [conv_bn(3, 32, 2)], 32
        for o, s in ((32, 1), (64, 2), (128, 2), (256, 2)):
            layers += [Block(c, o, s), Block(o, o, 1)]
            c = o
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Dropout(0.3), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.net(x)


def build_model(name, n_classes):
    return (ResNetSmall if name == "resnet" else SimpleCNN)(n_classes).to(DEVICE)


# ---------- mixup / cutmix (на уровне батча) ----------
def mix_batch(x, y, mode):
    """Возвращает смешанный батч, две метки (исходная и от пары) и долю lam исходного изображения."""
    lam = float(np.random.beta(MIX_ALPHA, MIX_ALPHA))
    idx = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam
    h, w = x.shape[2:]                                           # cutmix: вырезаем прямоугольник из другой картинки
    r = np.sqrt(1 - lam)
    ch, cw = int(h * r), int(w * r)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[idx][:, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)                    # точная доля после обрезки по границам
    return x, y, y[idx], lam


# ---------- оценка ----------
@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(DEVICE)).float(), 1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def metrics(y, probs):
    p = probs.argmax(1)
    return {"accuracy": accuracy_score(y, p),
            "balanced_acc": balanced_accuracy_score(y, p),
            "precision": precision_score(y, p, average="macro", zero_division=0),
            "recall": recall_score(y, p, average="macro", zero_division=0),
            "f1": f1_score(y, p, average="macro"),
            "roc_auc": roc_auc_score(y, probs, multi_class="ovr", average="macro",
                                     labels=np.arange(probs.shape[1]))}


# ---------- обучение одного фолда ----------
def run_fold(fold, pool, tr_idx, va_idx, test_dl, classes):
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    tr = [pool[i] for i in tr_idx]
    va = [pool[i] for i in va_idx]
    train_dl = make_loader(SamplesDataset(tr, train_tf), shuffle=True)
    val_dl = make_loader(SamplesDataset(va, eval_tf))

    model = build_model(MODEL, len(classes))
    counts = np.bincount([y for _, y in tr], minlength=len(classes))
    w = 1 / np.sqrt(counts)                                   # мягкая компенсация дисбаланса
    cw = torch.tensor(w / w.mean(), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=CFG["smooth"])   # для обучения
    val_criterion = nn.CrossEntropyLoss(weight=cw)               # для val loss: без сглаживания, сопоставим между экспериментами
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    ckpt = f"{RUN}_fold{fold}_best.pt"
    history, best_f1, best_val, bad = [], -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if CFG["mix"]:
                x, ya, yb, lam = mix_batch(x, y, CFG["mix"])
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                out = model(x)
                loss = (lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)) if CFG["mix"] \
                    else criterion(out, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_sum, n = loss_sum + loss.item() * len(y), n + len(y)

        probs, y_val = predict(model, val_dl)
        val_loss = val_criterion(torch.tensor(np.log(probs + 1e-9)).to(DEVICE),
                             torch.tensor(y_val).to(DEVICE)).item()
        m = metrics(y_val, probs)
        history.append({"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val_loss,
                        "val_acc": m["accuracy"], "val_f1": m["f1"]})
        print(f"[fold {fold}] {epoch:02d}/{EPOCHS} train_loss {loss_sum / n:.3f} "
              f"val_loss {val_loss:.3f} val_acc {m['accuracy']:.3f} val_f1 {m['f1']:.3f}")
        sched.step(m["f1"])
        if m["f1"] > best_f1:
            best_f1, best_val, bad = m["f1"], m, 0
            torch.save({"model": MODEL, "state_dict": model.state_dict(), "classes": classes}, ckpt)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[fold {fold}] ранняя остановка на эпохе {epoch}")
                break
    json.dump(history, open(f"{RUN}_fold{fold}_history.json", "w"))

    model.load_state_dict(torch.load(ckpt)["state_dict"])
    test_probs, y_test = predict(model, test_dl)
    del train_dl, val_dl
    return best_val, test_probs, y_test, len(history)


def main():
    pool, folds, test_samples, classes = load_samples()
    fold_ids = sorted(set(folds))
    print(f"Эксперимент: {EXPERIMENT} | аугментации: {[type(t).__name__ for t in CFG['aug']]} | "
          f"mix: {CFG['mix']} | label smoothing: {CFG['smooth']}")
    print(f"Классов: {len(classes)} {classes}")
    print(f"В пуле CV: {len(pool)} | фолдов: {len(fold_ids)} | test: {len(test_samples)} | устройство: {DEVICE}")

    test_dl = make_loader(SamplesDataset(test_samples, eval_tf))

    rows, all_test_probs, y_test = [], [], None
    for fold in fold_ids:
        tr_idx, va_idx = np.where(folds != fold)[0], np.where(folds == fold)[0]
        print(f"\n===== Фолд {fold + 1}/{len(fold_ids)}: train={len(tr_idx)}, val={len(va_idx)} =====")
        val_m, test_probs, y_test, n_ep = run_fold(fold, pool, tr_idx, va_idx, test_dl, classes)
        test_m = metrics(y_test, test_probs)
        all_test_probs.append(test_probs)
        print(f"[fold {fold}] VAL f1 {val_m['f1']:.4f} | TEST: "
              f"{ {k: round(float(v), 4) for k, v in test_m.items()} }")
        rows.append({"experiment": EXPERIMENT, "model": MODEL, "fold": fold, "epochs": n_ep,
                     **{f"val_{k}": float(v) for k, v in val_m.items()},
                     **{f"test_{k}": float(v) for k, v in test_m.items()}})

    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c.startswith(("val_", "test_"))]
    summary = {"experiment": EXPERIMENT, "model": MODEL, "fold": "mean", "epochs": df["epochs"].mean(),
               **df[num_cols].mean().to_dict()}
    summary_std = {"experiment": EXPERIMENT, "model": MODEL, "fold": "std", "epochs": df["epochs"].std(),
                   **df[num_cols].std(ddof=1).to_dict()}

    ens = metrics(y_test, np.mean(all_test_probs, axis=0))   # ансамбль фолдов на test
    ens_row = {"experiment": EXPERIMENT, "model": MODEL, "fold": "ensemble", "epochs": np.nan,
               **{f"test_{k}": float(v) for k, v in ens.items()}}

    out = pd.concat([df, pd.DataFrame([summary, summary_std, ens_row])], ignore_index=True)
    out.round(4).to_csv(f"{RUN}_cv_metrics.csv", index=False)

    print("\n===== ИТОГ (test, mean ± std по фолдам) =====")
    for k in ("accuracy", "balanced_acc", "precision", "recall", "f1", "roc_auc"):
        print(f"{k:13s} {summary[f'test_{k}']:.4f} ± {summary_std[f'test_{k}']:.4f}"
              f"   | ансамбль: {ens[k]:.4f}")
    print(f"\nСохранено: {RUN}_cv_metrics.csv")


if __name__ == "__main__":
    main()