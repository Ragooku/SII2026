"""
Сравнение экспериментов ЛР2 (аугментации) с базовым запуском ЛР1.

Читает все файлы *_cv_metrics.csv в текущей папке (их создаёт train_garbage_cv.py)
и строит сводную таблицу и график:
  * experiments_summary.csv  - таблица по всем экспериментам
  * experiments_compare.png  - F1 и accuracy: среднее по фолдам и ансамбль

Запуск без аргументов:  python compare_experiments.py
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["accuracy", "balanced_acc", "f1", "roc_auc"]


def main():
    files = sorted(glob.glob("*_cv_metrics.csv"))
    if not files:
        raise SystemExit("Не найдены файлы *_cv_metrics.csv. Сначала запустите train_garbage_cv.py")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        df["fold"] = df["fold"].astype(str)
        mean, std, ens = (df[df["fold"] == k].iloc[0] for k in ("mean", "std", "ensemble"))
        row = {"experiment": df["experiment"].iloc[0], "model": df["model"].iloc[0],
               "epochs_mean": round(float(mean["epochs"]), 1)}
        for m in METRICS:
            row[f"{m}_mean"] = float(mean[f"test_{m}"])
            row[f"{m}_std"] = float(std[f"test_{m}"])
            row[f"{m}_ens"] = float(ens[f"test_{m}"])
        rows.append(row)

    res = pd.DataFrame(rows)
    if "baseline" in set(res["experiment"]):                      # baseline первым, остальные по F1 ансамбля
        base = res[res["experiment"] == "baseline"]
        res = pd.concat([base, res[res["experiment"] != "baseline"].sort_values("f1_ens", ascending=False)])
        res["dF1_vs_baseline"] = res["f1_mean"] - float(base["f1_mean"].iloc[0])
        res["dF1_ens_vs_baseline"] = res["f1_ens"] - float(base["f1_ens"].iloc[0])
    res = res.reset_index(drop=True)
    res.round(4).to_csv("experiments_summary.csv", index=False, encoding="utf-8")

    # красивая таблица в консоль
    show = pd.DataFrame({
        "experiment": res["experiment"],
        "acc (mean)": res["accuracy_mean"].round(4),
        "F1 (mean ± std)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(res["f1_mean"], res["f1_std"])],
        "acc (ens)": res["accuracy_ens"].round(4),
        "F1 (ens)": res["f1_ens"].round(4)})
    if "dF1_vs_baseline" in res:
        show["dF1 (mean)"] = res["dF1_vs_baseline"].round(4)
        show["dF1 (ens)"] = res["dF1_ens_vs_baseline"].round(4)
    print(show.to_string(index=False))

    # график
    x, wd = np.arange(len(res)), 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for a, m, name in ((ax[0], "f1", "macro-F1"), (ax[1], "accuracy", "accuracy")):
        a.bar(x - wd / 2, res[f"{m}_mean"], wd, yerr=res[f"{m}_std"], capsize=3, label="среднее по фолдам")
        a.bar(x + wd / 2, res[f"{m}_ens"], wd, label="ансамбль")
        a.set_xticks(x)
        a.set_xticklabels(res["experiment"], rotation=30, ha="right")
        lo = min(res[f"{m}_mean"].min(), res[f"{m}_ens"].min())
        a.set_ylim(max(0, lo - 0.05), 1)
        a.set_title(f"Test {name}")
        a.grid(axis="y", alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("experiments_compare.png", dpi=120)
    plt.close(fig)
    print("\nСохранено: experiments_summary.csv, experiments_compare.png")


if __name__ == "__main__":
    main()