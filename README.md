# Intérprete de MLP y entrenamiento en MNIST

Este repositorio traslada el trabajo realizado en Google Colab a un proyecto estructurado en Git. Incluye:

- Una implementación *forward* de un perceptrón multicapa usando solo NumPy.
- Un mini-lenguaje textual y su intérprete para generar modelos de `tf.keras.Sequential`.
- Utilidades para entrenar y evaluar el modelo sobre MNIST desde la línea de comandos.
- Una interfaz web sencilla (Flask) para experimentar con arquitecturas y lanzar entrenamientos sin salir del navegador.

## Requisitos

```bash
python >= 3.9
pip install -r requirements.txt
```

> **Nota**: TensorFlow puede tardar en instalarse dependiendo de la plataforma. Si cuentas con GPU puedes instalar `tensorflow-gpu`.

## Estructura

```
.
├── docs/analysis.md          # Respuestas a las preguntas de reflexión
├── scripts/train_mnist.py    # Script CLI para entrenar modelos
├── src/mlp_compiler/         # Paquete con el MLP NumPy y el compilador a Keras
└── web/                      # Aplicación Flask para gestionar el ejercicio desde una web
```

## Uso desde la línea de comandos

Entrena el modelo por defecto (usa 5 épocas, batch de 128, valida con el 10 %) y muestra métricas en consola:

```bash
python scripts/train_mnist.py
```

Para experimentar con otras arquitecturas basta con pasar la cadena al parámetro `--architecture`:

```bash
python scripts/train_mnist.py \
  --architecture "Dense(256, relu) -> Dropout(0.3) -> Dense(128, relu) -> Dense(10, softmax)" \
  --epochs 3 --train-size 10000
```

El argumento `--train-size` permite limitar la cantidad de ejemplos usados durante el entrenamiento, útil para pruebas rápidas.

Si añades `--plot-path outputs/history.png` el script guardará las curvas de accuracy y pérdida.

## Interfaz web

1. Arranca el servidor Flask:

   ```bash
   ./scripts/run_web.sh
   ```

   El script se encarga de exportar las variables de entorno necesarias y, en caso de
   no encontrar el comando `flask`, instalará automáticamente las dependencias
   listadas en `requirements.txt`. Si prefieres hacerlo manualmente, puedes ejecutar
   `export FLASK_APP=web.app` seguido de `flask run`, o lanzar `python web/app.py` para
   el modo *debug*.

2. Visita `http://127.0.0.1:5000` y completa el formulario. El servidor entrenará el modelo usando 5 000 ejemplos por defecto para ofrecer una respuesta rápida y mostrará las métricas y gráficas generadas.

## Mini-lenguaje soportado

- `Dense(units, activation)`
- `Dropout(rate)`
- `Input(dim)` como nodo opcional al inicio (alternativa a pasar `input_dim` en la API).

Las activaciones disponibles son `relu`, `sigmoid`, `tanh`, `softmax` y `linear`.

## Referencias

- [Implementación de MLP en NumPy](src/mlp_compiler/numpy_mlp.py)
- [Compilador de arquitecturas a Keras](src/mlp_compiler/compiler.py)
- [Carga y entrenamiento en MNIST](src/mlp_compiler/training.py)
- [Preguntas de análisis](docs/analysis.md)
