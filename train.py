from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)


# =========================
# CONFIGURACIÓN GENERAL
# =========================

SEED = 42

PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "DataSet"
IMAGE_DIR = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_sargazo_efficientnet_b0.pt"

LABELS = ["nada", "bajo", "moderado", "abundante", "excesivo"]
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}

NUM_CLASSES = len(LABELS)
BATCH_SIZE = 32
NUM_WORKERS = 0  # En Windows/local empieza con 0. Si todo funciona, prueba 2 o 4.
EPOCHS_STAGE_1 = 10
EPOCHS_STAGE_2 = 15

LR_STAGE_1 = 1e-3
LR_STAGE_2 = 3e-5

WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("Dispositivo usado:", DEVICE)

def load_metadata():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print("\nColumnas de train.csv:")
    print(train_df.columns.tolist())

    required = {"label"}

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"Faltan columnas en {name}.csv: {missing}")

        if "label_id" not in df.columns:
            df["label"] = df["label"].astype(str).str.strip().str.lower()
            df["label_id"] = df["label"].map(label2id)

        if df["label_id"].isna().any():
            bad_labels = df[df["label_id"].isna()]["label"].unique()
            raise ValueError(f"Hay etiquetas no reconocidas en {name}.csv: {bad_labels}")

        df["label_id"] = df["label_id"].astype(int)

    return train_df, val_df, test_df


def compute_class_weights(train_df):
    counts = train_df["label"].value_counts().reindex(LABELS, fill_value=0)

    if (counts == 0).any():
        raise ValueError(f"Hay clases sin ejemplos en train.csv:\n{counts}")

    total = counts.sum()
    class_weights = total / (NUM_CLASSES * counts)

    class_weights_tensor = torch.tensor(
        class_weights.values,
        dtype=torch.float32
    )

    print("\nDistribución de clases en train:")
    print(counts)

    print("\nPesos por clase:")
    for label, weight in zip(LABELS, class_weights_tensor):
        print(f"{label:10s}: {weight.item():.4f}")

    return class_weights_tensor


class SargassumDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _get_image_path(self, row):
        if "image_path" in row and pd.notna(row["image_path"]):
            path = Path(row["image_path"])

            if path.exists():
                return path

            candidate = self.image_dir / path.name

            if candidate.exists():
                return candidate

        if "image_name" in row and pd.notna(row["image_name"]):
            candidate = self.image_dir / str(row["image_name"])

            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"No encontré la imagen para la fila:\n{row}"
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self._get_image_path(row)
        label = int(row["label_id"])

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.03
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform

def create_dataloaders(train_df, val_df, test_df):
    train_transform, eval_transform = get_transforms()

    train_dataset = SargassumDataset(
        dataframe=train_df,
        image_dir=IMAGE_DIR,
        transform=train_transform
    )

    val_dataset = SargassumDataset(
        dataframe=val_df,
        image_dir=IMAGE_DIR,
        transform=eval_transform
    )

    test_dataset = SargassumDataset(
        dataframe=test_df,
        image_dir=IMAGE_DIR,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("\nTamaños de datasets:")
    print("Train:", len(train_dataset))
    print("Val:  ", len(val_dataset))
    print("Test: ", len(test_dataset))

    return train_loader, val_loader, test_loader

def create_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )

    model = model.to(DEVICE)

    print("\nModelo creado: EfficientNet-B0")
    print("Clases de salida:", NUM_CLASSES)
    print("Base congelada: sí")

    return model

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, balanced_acc, macro_f1, all_labels, all_preds


def train_stage_1(model, train_loader, val_loader, class_weights):
    print("\n==============================")
    print("ETAPA 1: entrenar clasificador")
    print("==============================")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE_1,
        weight_decay=WEIGHT_DECAY
    )

    best_val_f1 = -1.0

    for epoch in range(1, EPOCHS_STAGE_1 + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, val_bal_acc, val_macro_f1, y_true, y_pred = evaluate(
            model,
            val_loader,
            criterion
        )

        print(f"\nEpoch {epoch}/{EPOCHS_STAGE_1}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")
        print(f"Val balanced acc: {val_bal_acc:.4f}")
        print(f"Val macro-F1:     {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Nuevo mejor modelo guardado.")

    print("\nMejor macro-F1 en etapa 1:", round(best_val_f1, 4))


def train_stage_2(model, train_loader, val_loader, class_weights):
    print("\n==============================")
    print("ETAPA 2: fine-tuning suave")
    print("==============================")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    for param in model.features[-2:].parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE_2,
        weight_decay=WEIGHT_DECAY
    )

    best_val_f1 = -1.0

    for epoch in range(1, EPOCHS_STAGE_2 + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, val_bal_acc, val_macro_f1, y_true, y_pred = evaluate(
            model,
            val_loader,
            criterion
        )

        print(f"\nFine-tuning epoch {epoch}/{EPOCHS_STAGE_2}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")
        print(f"Val balanced acc: {val_bal_acc:.4f}")
        print(f"Val macro-F1:     {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Nuevo mejor modelo fine-tuned guardado.")

    print("\nMejor macro-F1 en etapa 2:", round(best_val_f1, 4))


def evaluate_on_test(model, test_loader, class_weights):
        print("\n==============================")
        print("EVALUACIÓN FINAL EN TEST")
        print("==============================")

        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(DEVICE)
        )

        test_loss, test_acc, test_bal_acc, test_macro_f1, y_true, y_pred = evaluate(
            model,
            test_loader,
            criterion
        )

        print(f"\nTest loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test balanced accuracy: {test_bal_acc:.4f}")
        print(f"Test macro-F1: {test_macro_f1:.4f}")

        print("\nReporte por clase:")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=LABELS,
                digits=4
            )
        )

        print("\nMatriz de confusión:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)


def main():
    print("Inicio del entrenamiento")
    print("PyTorch CUDA disponible:", torch.cuda.is_available())

    train_df, val_df, test_df = load_metadata()

    class_weights = compute_class_weights(train_df)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df,
        val_df,
        test_df
    )

    model = create_model()

    train_stage_1(
        model,
        train_loader,
        val_loader,
        class_weights
    )

    train_stage_2(
        model,
        train_loader,
        val_loader,
        class_weights
    )

    evaluate_on_test(
        model,
        test_loader,
        class_weights
    )

    print("\nEntrenamiento terminado.")
    print("Mejor modelo guardado en:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()