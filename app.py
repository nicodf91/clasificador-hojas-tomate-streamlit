from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow import keras


MODEL_PATH = Path(__file__).parent / "models" / "modelo_hojas_tomate_mobilenetv2.keras"
IMAGE_SIZE = (224, 224)

CLASSES = [
    ("Tomato___healthy", "Hoja sana"),
    ("Tomato___Early_blight", "Early Blight"),
    ("Tomato___Late_blight", "Late Blight"),
    ("Tomato___Leaf_Mold", "Leaf Mold"),
]


@st.cache_resource
def load_tomato_model():
    """Carga el modelo una sola vez durante la sesion de Streamlit."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontro el modelo en: {MODEL_PATH}")

    return keras.models.load_model(MODEL_PATH)


def read_image(image_file):
    """Lee la imagen subida o tomada con camara y la convierte a RGB."""
    return Image.open(image_file).convert("RGB")


def prepare_image(image):
    """Redimensiona la imagen sin normalizar, porque el modelo ya lo incluye."""
    resized_image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def prediction_to_probabilities(prediction):
    """Devuelve probabilidades aunque el modelo entregue logits."""
    values = np.asarray(prediction, dtype=np.float32).reshape(-1)

    if values.shape[0] != len(CLASSES):
        raise ValueError(
            f"El modelo devolvio {values.shape[0]} salidas, pero se esperaban {len(CLASSES)}."
        )

    already_probabilities = (
        np.all(values >= 0)
        and np.all(values <= 1)
        and np.isclose(np.sum(values), 1.0, atol=1e-3)
    )

    if already_probabilities:
        return values

    return tf.nn.softmax(values).numpy()


def classify_image(model, image):
    batch = prepare_image(image)
    raw_prediction = model.predict(batch, verbose=0)[0]
    probabilities = prediction_to_probabilities(raw_prediction)
    best_index = int(np.argmax(probabilities))
    return CLASSES[best_index], float(probabilities[best_index]), probabilities


st.set_page_config(
    page_title="Clasificador de hojas de tomate",
    layout="centered",
)

st.title("Clasificador de hojas de tomate")
st.write(
    "Subi una imagen o toma una foto de una hoja de tomate para clasificarla con "
    "un modelo TensorFlow/Keras entrenado previamente."
)

st.info(
    "Esta app usa un modelo educativo. No reemplaza el diagnostico de un "
    "profesional agronomo o especialista en sanidad vegetal."
)

source = st.radio(
    "Selecciona el origen de la imagen",
    ["Subir imagen", "Tomar foto"],
    horizontal=True,
)

image_file = None
if source == "Subir imagen":
    image_file = st.file_uploader(
        "Subi una imagen desde tu dispositivo",
        type=["jpg", "jpeg", "png"],
    )
else:
    image_file = st.camera_input("Toma una foto con la camara del dispositivo")

image = None
if image_file is not None:
    try:
        image = read_image(image_file)
        st.image(image, caption="Vista previa", use_container_width=True)
    except UnidentifiedImageError:
        st.error("No se pudo leer la imagen. Proba con un archivo JPG o PNG valido.")

if st.button("Clasificar imagen"):
    if image is None:
        st.warning("Primero carga una imagen o toma una foto.")
        st.stop()

    try:
        model = load_tomato_model()
        (technical_name, friendly_name), confidence, probabilities = classify_image(
            model, image
        )
    except Exception as error:
        st.error("No se pudo clasificar la imagen.")
        st.caption(str(error))
        st.stop()

    st.subheader("Resultado principal")
    st.success(f"{friendly_name}")
    st.metric("Confianza", f"{confidence * 100:.2f}%")
    st.caption(f"Clase tecnica: {technical_name}")

    st.subheader("Probabilidades por clase")
    for (_, friendly_name), probability in zip(CLASSES, probabilities):
        st.write(f"**{friendly_name}**: {probability * 100:.2f}%")
        st.progress(float(probability))
