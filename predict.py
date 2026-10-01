from pathlib import Path
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0


# =========================
# CONFIGURACIÓN
# =========================

PROJECT_DIR = Path(".")

TEST_CSV = PROJECT_DIR / "DataSet" / "labels" / "test.csv"
IMAGE_DIR = PROJECT_DIR / "DataSet" / "images"

MODEL_PATH = PROJECT_DIR / "models" / "best_sargazo_efficientnet_b0.pt"

OUTPUT_CSV = PROJECT_DIR / "predicciones_test.csv"

LABELS = ["nada", "bajo", "moderado", "abundante", "excesivo"]
NUM_CLASSES = len(LABELS)

BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATASET PARA PREDICCIÓN
# =========================

class SargassumTestDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        if "image_name" not in self.df.columns:
            raise ValueError("El archivo test.csv debe tener una columna llamada image_name.")

    def __len__(self):
        return len(self.df)

    def _get_image_path(self, row):
        # Caso 1: si tu CSV ya tiene image_path
        if "image_path" in row and pd.notna(row["image_path"]):
            path = Path(row["image_path"])

            if path.exists():
                return path

            candidate = self.image_dir / path.name
            if candidate.exists():
                return candidate

        # Caso 2: si tu CSV solo tiene image_name
        candidate = self.image_dir / str(row["image_name"])

        if candidate.exists():
            return candidate

        raise FileNotFoundError(f"No se encontró la imagen: {row['image_name']}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = row["image_name"]
        image_path = self._get_image_path(row)

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, image_name


# =========================
# TRANSFORMACIONES
# =========================

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =========================
# CARGAR MODELO
# =========================

def create_model():
    model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )

    return model


def load_trained_model():
    model = create_model()

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # Por si el modelo fue guardado usando DataParallel
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key.replace("module.", ""): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model


# =========================
# GENERAR PREDICCIONES
# =========================

def predict_test():
    print("Dispositivo usado:", DEVICE)
    print("Cargando test.csv desde:", TEST_CSV)
    print("Cargando modelo desde:", MODEL_PATH)

    transform = get_eval_transform()

    test_dataset = SargassumTestDataset(
        csv_path=TEST_CSV,
        image_dir=IMAGE_DIR,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = load_trained_model()

    predictions = []

    with torch.no_grad():
        for images, image_names in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            pred_ids = outputs.argmax(dim=1).cpu().numpy()

            for image_name, pred_id in zip(image_names, pred_ids):
                predictions.append({
                    "image_name": image_name,
                    "label": LABELS[int(pred_id)]
                })

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\nPredicciones generadas:", len(pred_df))
    print("CSV guardado en:", OUTPUT_CSV)
    print("\nPrimeras predicciones:")
    print(pred_df.head())


if __name__ == "__main__":
    predict_test()