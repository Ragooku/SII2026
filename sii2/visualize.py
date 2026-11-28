import glob
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from train_garbage_cv import (RUN, SamplesDataset, build_model, eval_tf, load_samples,
                              make_loader, metrics, predict)


def show(ds, idxs, probs, y, classes, title, fname):
    rows = int(np.ceil(len(idxs) / 6))
    fig, axes = plt.subplots(rows, 6, figsize=(13, 2.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idxs):
        pred = probs[i].argmax()
        img = np.clip(ds[i][0].numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)
        ax.imshow(img)
        ax.set_title(f"T: {classes[y[i]][:16]}\nP: {classes[pred][:16]} ({probs[i][pred]:.2f})",
                     fontsize=8, color="green" if pred == y[i] else "red")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_curves(histories, fname):
    """histories: {номер_фолда: список записей из history.json}."""
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for n, (k, h) in enumerate(sorted(histories.items())):
        c, lab = colors[n % 10], f"fold {k + 1}"
        ax[0].plot([r["train_loss"] for r in h], "--", color=c, alpha=.6)
        ax[0].plot([r["val_loss"] for r in h], color=c, label=lab)
        ax[1].plot([r["val_acc"] for r in h], color=c, label=f"{lab} acc")
        ax[1].plot([r["val_f1"] for r in h], ":", color=c, label=f"{lab} F1")
    ax[0].set_title("Loss (пунктир - train, сплошная - val)")
    ax[1].set_title("Val metrics (сплошная - accuracy, точки - macro-F1)")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_confusion(y, pred, classes, fname):
    cm = confusion_matrix(y, pred, normalize="true", labels=np.arange(len(classes)))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > .5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def main():
    ckpts = sorted(glob.glob(f"{RUN}_fold*_best.pt"))
    if not ckpts:
        raise SystemExit(f"Не найдены чекпоинты {RUN}_fold*_best.pt. Сначала запустите train_garbage_cv.py")

    _, _, test_samples, _ = load_samples()
    ds = SamplesDataset(test_samples, eval_tf)
    test_dl = make_loader(ds)

    all_probs, classes, y = [], None, None
    for path in ckpts:
        ckpt = torch.load(path)
        classes = ckpt["classes"]
        model = build_model(ckpt["model"], len(classes))
        model.load_state_dict(ckpt["state_dict"])
        probs, y = predict(model, test_dl)
        all_probs.append(probs)
        print(f"{path}: test acc {(probs.argmax(1) == y).mean():.4f}")

    probs = np.mean(all_probs, axis=0)                    # ансамбль фолдов
    pred, conf = probs.argmax(1), probs.max(1)
    nll = -np.log(probs[np.arange(len(y)), y] + 1e-9)
    err = pred != y
    print(f"\nАнсамбль из {len(ckpts)} моделей: "
          f"{ {k: round(float(v), 4) for k, v in metrics(y, probs).items()} }")
    if err.any():
        print(f"loss {nll.mean():.3f}, доля loss от ошибок {nll[err].sum() / nll.sum():.0%}, "
              f"средняя уверенность на ошибках {conf[err].mean():.2f}")

    # кривые обучения всех фолдов
    histories = {int(re.search(r"fold(\d+)_history", p).group(1)): json.load(open(p))
                 for p in glob.glob(f"{RUN}_fold*_history.json")}
    if histories:
        plot_curves(histories, f"{RUN}_curves.png")

    # матрица ошибок и отчёт по классам
    plot_confusion(y, pred, classes, f"{RUN}_confusion.png")
    rep = pd.DataFrame(classification_report(y, pred, labels=np.arange(len(classes)),
                                             target_names=classes, output_dict=True,
                                             zero_division=0)).T
    rep.round(4).to_csv(f"{RUN}_class_report.csv", encoding="utf-8")
    print("\nОтчёт по классам (test, ансамбль):")
    print(rep.round(3).to_string())

    # примеры классификации и самые уверенные ошибки
    rng = np.random.default_rng(0)
    show(ds, rng.choice(len(ds), min(24, len(ds)), replace=False), probs, y, classes,
         "Test predictions (green - correct, red - wrong)", f"{RUN}_examples.png")
    wrong = np.where(err)[0]
    if len(wrong):
        show(ds, wrong[np.argsort(-conf[wrong])][:18], probs, y, classes,
             "Most confident errors", f"{RUN}_errors.png")
    print("\nГрафики сохранены в текущей папке")


if __name__ == "__main__":
    main()