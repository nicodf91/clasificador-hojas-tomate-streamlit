# Clasificador de hojas de tomate

App web educativa hecha con Streamlit para clasificar imagenes de hojas de tomate usando un modelo TensorFlow/Keras entrenado previamente.

## Estructura

```text
app.py
requirements.txt
README.md
models/
  modelo_hojas_tomate_mobilenetv2.keras
```

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Ejecutar la app

```powershell
python -m streamlit run app.py
```

Luego abri la URL local que muestra Streamlit en la terminal.

## Clases del modelo

El modelo usa este orden exacto de clases:

1. `Tomato___healthy` - Hoja sana
2. `Tomato___Early_blight` - Early Blight
3. `Tomato___Late_blight` - Late Blight
4. `Tomato___Leaf_Mold` - Leaf Mold

## Notas

- La imagen se convierte a RGB y se redimensiona a `224x224`.
- No se divide manualmente la imagen por `255`, porque el modelo incluye preprocesamiento interno.
- No se usa `kaggle.json`.
- No se incluyen claves ni credenciales.
- No se modifica ni reentrena el modelo.
- El resultado es educativo y no reemplaza un diagnostico profesional.
