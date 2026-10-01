from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import hashlib

# Rutas
BASE_DIR = Path("DataSet")
IMG_DIR = BASE_DIR / "images"
CSV_PATH = BASE_DIR / "labels/labels.csv"
OUT_DIR = BASE_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Etiquetas en orden
LABELS = ["nada", "bajo", "moderado", "abundante", "excesivo"]
label2id = {label: i for i, label in enumerate(LABELS)}

# Leer CSV
df = pd.read_csv(CSV_PATH)

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower()

# Revisar columnas necesarias
required_cols = {"image_name", "place", "date", "label", "scene"}
missing_cols = required_cols - set(df.columns)

if missing_cols:
    raise ValueError(f"Faltan columnas en el CSV: {missing_cols}")

# Limpiar texto
df["image_name"] = df["image_name"].astype(str).str.strip()
df["place"] = df["place"].astype(str).str.strip()
df["date"] = df["date"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["scene"] = df["scene"].astype(str).str.strip().str.lower()

# Validar etiquetas
invalid_labels = sorted(set(df["label"]) - set(LABELS))

if invalid_labels:
    raise ValueError(f"Hay etiquetas no válidas: {invalid_labels}")

# Convertir etiqueta a número
df["label_id"] = df["label"].map(label2id)

# Procesar fechas
df["date_clean"] = df["date"].replace({
    "desconocido": np.nan,
    "": np.nan,
    "nan": np.nan
})

df["date_parsed"] = pd.to_datetime(
    df["date_clean"],
    dayfirst=True,
    errors="coerce"
)

# Crear ruta completa de cada imagen
df["image_path"] = df["image_name"].apply(lambda x: str(IMG_DIR / x))

# Verificar si existe cada imagen
df["exists"] = df["image_path"].apply(lambda x: Path(x).exists())

print("Imágenes totales en CSV:", len(df))
print("Imágenes encontradas:", df["exists"].sum())
print("Imágenes faltantes:", (~df["exists"]).sum())

# Guardar imágenes faltantes para revisarlas
df_missing = df[~df["exists"]]
df_missing.to_csv(OUT_DIR / "imagenes_faltantes.csv", index=False)

# Quedarse solo con imágenes existentes
df = df[df["exists"]].copy()


def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
        return True, width, height, mode
    except Exception:
        return False, None, None, None


results = df["image_path"].apply(check_image)

df["valid_image"] = results.apply(lambda x: x[0])
df["width"] = results.apply(lambda x: x[1])
df["height"] = results.apply(lambda x: x[2])
df["mode"] = results.apply(lambda x: x[3])

print("Imágenes válidas:", df["valid_image"].sum())
print("Imágenes corruptas:", (~df["valid_image"]).sum())

# Guardar corruptas
df_corrupt = df[~df["valid_image"]]
df_corrupt.to_csv(OUT_DIR / "imagenes_corruptas.csv", index=False)

# Quedarse solo con imágenes válidas
df = df[df["valid_image"]].copy()



class_counts = df["label"].value_counts().reindex(LABELS)

print(class_counts)

class_percentages = df["label"].value_counts(normalize=True).reindex(LABELS) * 100

summary = pd.DataFrame({
    "conteo": class_counts,
    "porcentaje": class_percentages.round(2)
})

print(summary)

summary.to_csv(OUT_DIR / "distribucion_clases.csv")

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


df["sha256"] = df["image_path"].apply(file_hash)

duplicated = df.duplicated(subset="sha256", keep=False)

print("Imágenes duplicadas exactas:", duplicated.sum())

df_duplicates = df[duplicated].sort_values("sha256")
df_duplicates.to_csv(OUT_DIR / "duplicados_exactos.csv", index=False)

# Eliminar duplicados exactos, dejando solo el primero
df = df.drop_duplicates(subset="sha256", keep="first").copy()

print("Dataset después de eliminar duplicados:", len(df))

from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nDistribución train:")
print(train_df["label"].value_counts(normalize=True).reindex(LABELS))

print("\nDistribución validation:")
print(val_df["label"].value_counts(normalize=True).reindex(LABELS))

print("\nDistribución test:")
print(test_df["label"].value_counts(normalize=True).reindex(LABELS))

# Guardar archivos
train_df.to_csv(OUT_DIR / "train.csv", index=False)
val_df.to_csv(OUT_DIR / "val.csv", index=False)
test_df.to_csv(OUT_DIR / "test.csv", index=False)

full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
full_df.to_csv(OUT_DIR / "metadata_clean_splits.csv", index=False)