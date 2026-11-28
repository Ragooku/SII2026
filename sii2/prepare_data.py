from pathlib import Path

import pandas as pd


# ---------- настройки ----------
DATA_DIR = Path(r"C:\Users\Влад\OneDrive\Desktop\dataset\original")
MIN_FRACTION = 0.03        # класс исключается, если его доля от всего датасета меньше (0.03 = 3%)
MIN_COUNT = 0              # ... или если в нём меньше стольких изображений
VERIFY_IMAGES = True       # проверять изображения на целостность (медленнее, нужен Pillow)
OUT = Path("dataset.csv")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_dataset_root(data_dir: Path) -> Path:
    cur = data_dir
    for _ in range(4):
        subdirs = [d for d in cur.iterdir() if d.is_dir() and not d.name.startswith(".")]
        has_images = any(
            f.suffix.lower() in IMG_EXTS for d in subdirs for f in d.iterdir() if f.is_file()
        )
        if has_images:
            return cur
        if len(subdirs) == 1:  # обёртка вида root/garbage-dataset/...
            cur = subdirs[0]
        else:
            break
    return cur


def is_valid_image(path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return True  # без Pillow проверку пропускаем
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_images(root: Path, check: bool):
    classes, broken = {}, []
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")):
        files = []
        for f in cls_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                if check and not is_valid_image(f):
                    broken.append(f)
                else:
                    files.append(f)
        if files:
            classes[cls_dir.name] = sorted(files)
    return classes, broken


def print_table(counts: dict, total: int, kept: set):
    w = max(len(c) for c in counts) + 2
    print(f"\n{'Класс':<{w}}{'Кол-во':>8}{'Доля':>9}  Статус")
    print("-" * (w + 28))
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        status = "оставлен" if cls in kept else "ИСКЛЮЧЁН"
        print(f"{cls:<{w}}{n:>8}{n / total:>8.1%}  {status}")
    print("-" * (w + 28))
    print(f"{'ВСЕГО':<{w}}{total:>8}")


def main():
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Папка не найдена: {DATA_DIR}")

    root = find_dataset_root(DATA_DIR)
    print(f"Корень датасета: {root}")
    classes, broken = collect_images(root, check=VERIFY_IMAGES)
    if not classes:
        raise SystemExit("Изображения не найдены. Проверьте DATA_DIR.")

    counts = {c: len(f) for c, f in classes.items()}
    total = sum(counts.values())
    kept = {c for c, n in counts.items() if n / total >= MIN_FRACTION and n >= MIN_COUNT}

    print_table(counts, total, kept)
    print(f"\nКлассов всего: {len(counts)} | оставлено: {len(kept)} | "
          f"исключено: {[c for c in counts if c not in kept]}")
    if broken:
        print(f"Битых файлов пропущено: {len(broken)}")

    df = pd.DataFrame([(str(f), c) for c in sorted(kept) for f in classes[c]], columns=["path", "label"])
    df.to_csv(OUT, index=False, encoding="utf-8")

    report = pd.DataFrame({"class": list(counts), "count": list(counts.values())})
    report["fraction"] = (report["count"] / total).round(4)
    report["kept"] = report["class"].isin(kept)
    report.to_csv("class_report.csv", index=False, encoding="utf-8")

    print(f"\nИзображений после фильтрации: {len(df)} из {total}")
    print(f"Сохранено: {OUT}, class_report.csv")


if __name__ == "__main__":
    main()