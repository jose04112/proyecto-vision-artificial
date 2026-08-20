# proyecto-vision-artificial
# Generación de conjuntos de entrenamiento para detección y reconocimiento de objetos de ensamble mediante un sistema de visión 

Este repositorio contiene la documentación, scripts y cuadernos de entrenamiento para el sistema de detección y reconocimiento de objetos de ensamble.

##  Cuadernos de Entrenamiento (Google Colab)

Puedes revisar y ejecutar los entornos de entrenamiento de los modelos directamente en google colab

| Modelo | Enlace de Acceso Directo |
| :--- | :--- |
| **Entrenamiento CLIP** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jose04112/proyecto-vision/blob/main/vision/entrenamientos/CLIP.ipynb) |
| **Entrenamiento YOLO** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jose04112/proyecto-vision/blob/main/vision/entrenamientos/YOLO.ipynb) |

---

##  Modelos Entrenados (Pesos .pt)

Los archivos con los pesos óptimos obtenidos tras los entrenamientos se encuentran alojados en la siguiente carpeta compartida de Google Drive. 

*  [Descargar Pesos de los Modelos (YOLO y CLIP)](https://drive.google.com/drive/folders/1v8PbTGbu0utwCL89MzKtbMJwgR01BEps?usp=sharing)

> **Nota:** Es necesario colocar los archivos `.pt` descargados dentro del directorio correspondiente del sistema para asegurar que el script principal los localice correctamente al iniciar el programa.



## Arquitectura y Flujo del Sistema Principal


1. **Captura y Alineación (Intel RealSense):** El sistema inicializa la cámara para capturar los streams de video en tiempo real, alineando los mapas de profundidad con la imagen a color.
2. **Detección y Segmentación (YOLOv11):** El modelo YOLO procesa los cuadros para detectar y localizar espacialmente los componentes de ensamble.
3. **Clasificación Contextual (CLIP):** Las regiones de interés detectadas son evaluadas por el modelo CLIP para clasificar el estado actual del proceso (por ejemplo, determinar si se encuentra en estado de "ensamble" o "desensamble").

### Componentes Clave

* **`requirements.txt`**: Archivo de dependencias que instala el entorno necesario 
* **`vision/`**: Contiene la lógica del sistema y los scripts de ejecución principal.
* **`vision/entrenamientos/`**: Carpeta destinada a los cuadernos `.ipynb` de preparación de datos y entrenamiento de CLIP (empleando un conjunto de 700 imágenes) y YOLO (configurado con 1,500 objetos segmentados por clase).
