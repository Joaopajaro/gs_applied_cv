import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

app = FastAPI(title="CIFAR-10 CNN Classifier", version="1.0")

# Carrega modelo e labels ao iniciar
_loaded = tf.saved_model.load("saved_model")
_infer = _loaded.signatures["serving_default"]
labels = np.load("labels.npy", allow_pickle=True).tolist()

# descobre o nome da chave de saida
_output_key = list(_infer.structured_outputs.keys())[0]


def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((32, 32))
    arr = np.array(img, dtype="float32") / 255.0
    return tf.constant(np.expand_dims(arr, axis=0))


@app.get("/")
def root():
    return {"message": "CIFAR-10 Classifier — use POST /predict com uma imagem"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem")

    image_bytes = await file.read()
    img = preprocess_image(image_bytes)
    preds = _infer(img)[_output_key].numpy()[0]

    top_idx = int(np.argmax(preds))
    return {
        "classe": labels[top_idx],
        "confianca": float(round(preds[top_idx] * 100, 2)),
        "todas_probabilidades": {labels[i]: float(round(preds[i] * 100, 2)) for i in range(10)},
    }
