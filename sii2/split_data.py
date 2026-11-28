import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("dataset.csv"), help="CSV от prepare_data.py")
    p.add_argument("--out", type=Path, default=Path("splits.csv"))
    p.add_argument("--folds", type=int, default=5, help="число фолдов K (>= 2)")
    p.add_argument("--test-size", type=float, default=0.15, help="доля test от всего датасета")
    p.add_argument("--val-fold", type=int, default=0, help="какой фолд считать val в одиночном обучении")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.folds < 2:
        raise SystemExit("--folds должно быть >= 2")
    if not 0 <= args.val_fold < args.folds:
        raise SystemExit(f"--val-fold должен быть от 0 до {args.folds - 1}")
    if not args.data.exists():
        raise SystemExit(f"Файл {args.data} не найден. Сначала запустите prepare_data.py")

    df = pd.read_csv(args.data)
    min_cls = df["label"].value_counts().min()
    if min_cls < args.folds + 2:
        raise SystemExit(f"В самом малом классе всего {min_cls} изображений — мало для {args.folds} фолдов. "
                         "Увеличьте --min-fraction в prepare_data.py или уменьшите --folds.")

    # 1) test
    pool, test = train_test_split(df, test_size=args.test_size, stratify=df["label"],
                                  random_state=args.seed)
    pool = pool.reset_index(drop=True)

    # 2) фолды
    pool["fold"] = -1
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (_, va_idx) in enumerate(skf.split(np.zeros(len(pool)), pool["label"])):
        pool.loc[va_idx, "fold"] = k
    pool["split"] = np.where(pool["fold"] == args.val_fold, "val", "train")

    test = test.assign(fold=-1, split="test")
    out = pd.concat([pool, test], ignore_index=True)[["path", "label", "split", "fold"]]

    # проверка на утечки: ни один файл не должен попасть в две части
    assert out["path"].is_unique, "Обнаружены дубликаты путей"
    assert not set(pool["path"]) & set(test["path"]), "Пересечение train/val и test"

    out.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Всего: {len(out)} | классов: {out['label'].nunique()}")
    print("\nРазбиение для одиночного обучения (val = фолд {}):".format(args.val_fold))
    print(out.groupby(["label", "split"]).size().unstack().fillna(0).astype(int)
          [["train", "val", "test"]].assign(total=lambda d: d.sum(axis=1)))
    print("\nРазмеры фолдов (train+val пул):")
    print(pool.groupby("fold").size().to_string())
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()