"""Сегментация области показаний водосчётчика (Water Meters Dataset, Kaggle).

Адаптация исходной лабораторной: U-Net / ResNet34, CrossEntropy + Dice.
Классы: 0 — фон, 1 — область показаний. Маски — отдельные JPEG/PNG-файлы.
Источник: https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset
Полный архив Kaggle около 1,1 ГБ, включая иллюстративные коллажи.

Установка зависимостей в активном .venv (Python 3.10+, PyTorch 2.3+):
    python -m pip install torch torchvision segmentation-models-pytorch numpy scikit-learn pillow matplotlib tqdm kagglehub

Автоматическое скачивание публичного датасета и обучение:
    python severstal.py --download

Или скачайте ZIP со страницы Kaggle и распакуйте в ./water_meters:
    python severstal.py --data-dir ./water_meters
Скрипт сам находит вложенную папку WaterMeters с images/ и masks/.
CSV и collage для сегментации не используются.

Быстрая проверка загрузчика без обучения и скачивания весов:
    python severstal.py --download --check-data
Пробное обучение на одной эпохе:
    python severstal.py --data-dir ./water_meters --epochs 1
Более подробная сегментация с большим потреблением памяти:
    python severstal.py --data-dir ./water_meters --height 512 --width 512 --batch-size 2
Предсказание после обучения:
    python severstal.py --predict ./photo.jpg --checkpoint ./water_meter_results/model.pth

По умолчанию снимки масштабируются до 256x256, batch-size=4, epochs=30.
Выборки 70/15/15 разделяются до аугментаций; списки сохраняются в split.json.
ImageNet-веса скачиваются при первом обучении; --encoder-weights none отключает
их загрузку. Устройство выбирается автоматически: CUDA, затем Apple MPS, CPU.
Метрики (IoU и Dice) относятся к области показаний и считаются по всей выборке.
Это сегментация области, а не распознавание цифр или значений счётчика.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ColorJitter, functional as TF
from tqdm import tqdm
import segmentation_models_pytorch as smp


NUM_CLASSES = 2
DATASET_HANDLE = "tapakah68/yandextoloka-water-meters-dataset"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def find_data_root(data_dir):
    """Принимает корень архива, WaterMeters или более глубоко вложенную папку."""
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Нет папки {root.resolve()}. Скачайте и распакуйте датасет "
            "или запустите python severstal.py --download."
        )
    direct = [root, root / "WaterMeters"]
    for candidate in direct:
        if (candidate / "images").is_dir() and (candidate / "masks").is_dir():
            return candidate
    matches = sorted({p.parent for p in root.rglob("images")
                      if p.is_dir() and (p.parent / "masks").is_dir()})
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Найдено несколько пар images/masks. Укажите нужную папку в --data-dir.")
    raise FileNotFoundError(
        f"В {root.resolve()} не найдены соседние папки images и masks. "
        "Распакуйте ZIP полностью или используйте --download."
    )


def load_samples(data_dir):
    """Сопоставляет каждое исходное изображение с маской по имени без расширения."""
    root = find_data_root(data_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in (root / "images").iterdir()
                   if p.is_file() and p.suffix.lower() in extensions and not p.name.startswith("."))
    masks = {}
    for path in sorted((root / "masks").iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions or path.name.startswith("."):
            continue
        if path.stem in masks:
            raise ValueError(f"Несколько масок с одним именем: {path.stem}")
        masks[path.stem] = path
    if not paths:
        raise ValueError(f"В {root / 'images'} нет изображений.")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Повторяющиеся имена изображений с разными расширениями.")
    missing = [p.name for p in paths if p.stem not in masks]
    if missing:
        raise FileNotFoundError(f"Нет масок для {len(missing)} изображений. Примеры: {missing[:5]}")
    return [{"image_id": p.name, "path": str(p), "mask_path": str(masks[p.stem])} for p in paths]


def read_image_mask(sample):
    """Учитывает EXIF-ориентацию и убирает артефакты сжатия JPEG-маски."""
    with Image.open(sample["path"]) as source:
        # Маски датасета размечены в отображаемой ориентации фотографии.
        # Например, EXIF=6 поворачивает исходные 1333x1000 в 1000x1333.
        image = ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(sample["mask_path"]) as source:
        mask = ImageOps.exif_transpose(source).convert("L")
    if image.size != mask.size:
        raise ValueError(f"{sample['image_id']}: размеры фото {image.size} и маски {mask.size} различаются.")
    mask = Image.fromarray((np.asarray(mask) > 127).astype(np.uint8))
    return image, mask


def split_samples(samples, seed):
    if len(samples) < 20:
        raise ValueError("Для разбиения 70/15/15 нужно хотя бы 20 изображений.")
    test_count = max(1, round(len(samples) * 0.15))
    val_count = max(1, round(len(samples) * 0.15))
    train_val, test = train_test_split(samples, test_size=test_count, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_count, random_state=seed)
    return train, val, test


class WaterMeterSegmentationDataset(Dataset):
    def __init__(self, samples, image_size, augment=False):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, mask = read_image_mask(self.samples[index])
        if self.augment:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, expand=True)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, expand=True)
            image = self.color_jitter(image)
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        image = TF.normalize(TF.to_tensor(image), mean=MEAN, std=STD)
        return image, torch.from_numpy(np.array(mask, dtype=np.int64))


def create_model(weights="imagenet"):
    return smp.Unet(
        encoder_name="resnet34", encoder_weights=weights,
        in_channels=3, classes=NUM_CLASSES, activation=None
    )


class DiceLoss(nn.Module):
    def forward(self, predictions, targets):
        # Считаем Dice только для области показаний; float32 стабилизирует расчёт при AMP.
        probabilities = torch.softmax(predictions.float(), dim=1)[:, 1]
        target = (targets == 1).float()
        intersection = (probabilities * target).sum()
        denominator = probabilities.sum() + target.sum()
        return 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)


def calculate_loss(predictions, targets):
    return F.cross_entropy(predictions.float(), targets) + DiceLoss()(predictions, targets)


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Обучение", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            predictions = model(images)
            loss = calculate_loss(predictions, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = total_pixels = intersection = pred_area = target_area = 0
    image_dice_sum = positive_dice_sum = 0.0
    positive_count = 0
    for images, masks in tqdm(loader, desc="Оценка", leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = calculate_loss(logits, masks)
        predictions = logits.argmax(dim=1)
        pred, target = predictions == 1, masks == 1
        inter = (pred & target).sum(dim=(1, 2)).float()
        pred_sum = pred.sum(dim=(1, 2)).float()
        true_sum = target.sum(dim=(1, 2)).float()
        denom = pred_sum + true_sum
        # Пара из двух пустых масок получает Dice=1.
        image_dice = torch.where(denom > 0, 2 * inter / denom.clamp_min(1), torch.ones_like(denom))
        positives = true_sum > 0
        image_dice_sum += image_dice.sum().item()
        positive_dice_sum += image_dice[positives].sum().item()
        positive_count += positives.sum().item()
        intersection += inter.sum().item()
        pred_area += pred_sum.sum().item()
        target_area += true_sum.sum().item()
        correct += (predictions == masks).sum().item()
        total_pixels += masks.numel()
        total_loss += loss.item() * images.size(0)
    union = pred_area + target_area - intersection
    denom = pred_area + target_area
    # Метрики суммируются по всей выборке, а не усредняются по батчам.
    return {
        "loss": total_loss / len(loader.dataset),
        "pixel_accuracy": correct / total_pixels,
        "foreground_iou": intersection / union if union else 1.0,
        "foreground_dice": 2 * intersection / denom if denom else 1.0,
        "mean_image_dice": image_dice_sum / len(loader.dataset),
        "positive_image_dice": positive_dice_sum / positive_count if positive_count else None,
    }


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_history(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0].set_title("CrossEntropy + Dice")
    axes[1].plot(epochs, history["val_iou"], label="Foreground IoU")
    axes[1].plot(epochs, history["val_dice"], label="Foreground Dice")
    axes[1].set_title("Validation metrics")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def visualize_predictions(model, dataset, device, output_dir, count=5):
    model.eval()
    chosen = random.Random(42).sample(range(len(dataset)), min(count, len(dataset)))
    for number, index in enumerate(chosen[:count], 1):
        image, mask = dataset[index]
        prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
        image_np = np.clip(image.numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN), 0, 1)
        fig, axes = plt.subplots(3, 1, figsize=(14, 7))
        axes[0].imshow(image_np)
        axes[0].set_title(f"Image: {dataset.samples[index]['image_id']}")
        axes[1].imshow(mask.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground truth: meter reading area")
        axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Prediction: meter reading area")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"prediction_{number}.png", dpi=150)
        plt.close(fig)


@torch.no_grad()
def predict_image(model, image_path, image_size, device, output_dir):
    model.eval()
    with Image.open(image_path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    image = TF.resize(original, image_size, interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(image), mean=MEAN, std=STD).unsqueeze(0).to(device)
    logits = model(tensor)
    logits = F.interpolate(logits.float(), size=(original.height, original.width), mode="bilinear", align_corners=False)
    mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    Image.fromarray(mask * 255).save(output_dir / "mask.png")
    fig, axes = plt.subplots(2, 1, figsize=(14, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original image")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Meter reading area mask")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "segmentation.png", dpi=150)
    plt.close(fig)


def train_model(args, device, output_dir):
    samples = load_samples(args.data_dir)
    train, val, test = split_samples(samples, args.seed)
    splits = {"train": train, "validation": val, "test": test}
    save_json(output_dir / "split.json", {key: [s["image_id"] for s in value] for key, value in splits.items()})
    for name, part in splits.items():
        print(f"{name}: {len(part)} изображений")
    image_size = [args.height, args.width]
    datasets = {
        name: WaterMeterSegmentationDataset(part, image_size, augment=name == "train")
        for name, part in splits.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=name == "train",
            num_workers=0, pin_memory=device.type == "cuda", generator=generator
        )
        for name, dataset in datasets.items()
    }
    model = create_model(None if args.encoder_weights == "none" else "imagenet").to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = {key: [] for key in ["train_loss", "val_loss", "val_iou", "val_dice"]}
    best_val_loss = float("inf")
    model_path = output_dir / "model.pth"
    config = vars(args).copy()
    config.update({"num_classes": NUM_CLASSES, "image_size": image_size, "task": "binary_water_meter_reading_area_segmentation"})
    save_json(output_dir / "config.json", config)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, scaler, device)
        metrics = evaluate(model, loaders["validation"], device)
        scheduler.step(metrics["loss"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["val_iou"].append(metrics["foreground_iou"])
        history["val_dice"].append(metrics["foreground_dice"])
        print(
            f"Эпоха {epoch}/{args.epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={metrics['loss']:.4f}, IoU={metrics['foreground_iou']:.4f}, "
            f"Dice={metrics['foreground_dice']:.4f}"
        )
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, model_path)
        save_json(output_dir / "history.json", history)
    plot_history(history, output_dir)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    results = evaluate(model, loaders["test"], device)
    results["best_epoch"] = checkpoint["epoch"]
    save_json(output_dir / "test_results.json", results)
    print("\nРезультаты на отложенной тестовой выборке:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    visualize_predictions(model, datasets["test"], device, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="./water_meters")
    parser.add_argument("--output-dir", default="./water_meter_results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--download", action="store_true", help="Скачать Water Meters Dataset через kagglehub")
    parser.add_argument("--check-data", action="store_true", help="Проверить пары фото/масок без обучения")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--predict", help="Путь к изображению для предсказания вместо обучения")
    parser.add_argument("--checkpoint", default="./water_meter_results/model.pth")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size и learning-rate должны быть положительными.")
    if args.height < 64 or args.width < 64 or args.height % 32 or args.width % 32:
        parser.error("height и width должны быть кратны 32 и не меньше 64.")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device_name = args.device
    device = torch.device(device_name)
    print(f"Устройство: {device}")
    if args.predict:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model = create_model(weights=None).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        predict_image(model, args.predict, checkpoint["config"]["image_size"], device, output_dir)
    else:
        if args.download:
            try:
                import kagglehub
            except ImportError:
                parser.error("Установите загрузчик: python -m pip install kagglehub")
            args.data_dir = kagglehub.dataset_download(DATASET_HANDLE)
            print(f"Датасет: {args.data_dir}")
        if args.check_data:
            samples = load_samples(args.data_dir)
            nonempty = 0
            for sample in tqdm(samples, desc="Проверка фото и масок"):
                _, mask = read_image_mask(sample)
                nonempty += int(np.asarray(mask).any())
            train, val, test = split_samples(samples, args.seed)
            print(f"Проверено {len(samples)} пар. Непустых масок: {nonempty}.")
            print(f"Разбиение: train={len(train)}, val={len(val)}, test={len(test)}.")
            return
        train_model(args, device, output_dir)
    print(f"Файлы сохранены в {output_dir.resolve()}")


if __name__ == "__main__":
    main()
