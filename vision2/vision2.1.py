import time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLOE
import open_clip
from PIL import Image
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32, String
from vision_msgs.msg import ObjectDetection

# CONFIGURACIÓN - MODELOS YOLOE

CUSTOM_MODEL_PATH = "best YOLOE.pt"
BASE_MODEL_PATH = "yoloe-26s-seg.pt"

CUSTOM_CONF = 0.25
BASE_CONF = 0.3

BASE_PROMPTS = [
  "white block on a gray workbench ",
  "metal T-shaped pin on a workbench",
  "silvery circular piece with four crosswise holes and a circular bar on a work table",
  "light blue box",
  "black rectangular block on a workbench"
]

# Umbral de IoU para comparar detecciones del BASE contra el CUSTOM (evita que el base dibuje un "obstaculo" encima de una pieza ya detectada por el custom).
IOU_OVERLAP_THRESHOLD = 0.3

# Umbral de IoU para el NMS class-agnostic DENTRO del propio modelo base
# (evita que dos prompts distintos generen dos cajas casi idénticas sobre el
# mismo objeto). 
BASE_NMS_IOU = 0.5

# Si una caja del BASE cubre más de este % del área total del frame, se
# descarta directamente: un obstáculo real es un objeto localizado, no algo
# que ocupe casi toda la imagen. Esto filtra falsos positivos de prompts

MAX_BASE_BOX_AREA_RATIO = 0.35

COLOR_CUSTOM = (0, 255, 0)   # verde
COLOR_BASE = (255, 0, 0)     # azul

# Radio (en píxeles) alrededor del centroide de cada caja para medir distancia
RADIO_CENTROIDE = 4

# OPTIMIZACIÓN DE VELOCIDAD (CPU)
DEVICE = "cpu"
IMG_SIZE = 320
BASE_MODEL_EVERY_N_FRAMES = 3
TORCH_THREADS = 3

# La distancia a un obstáculo se recalcula cada
# DEPTH_EVERY_N_FRAMES frames y se reutiliza la última medición mientras tanto.
DEPTH_EVERY_N_FRAMES = 3

# CONFIGURACIÓN - CLIP

CLIP_MODEL_PATH = "best_clip_V7.pt"   # modelo entrenado
CLIP_DEVICE = "cpu"  

# correr CLIP cada 9 frames ≈ cada 0.8-0.9s.
FRECUENCIA_CLIP = 9

clases_clip = [
    "part_circular_female_white", "part_circular_female_yellow", "part_circular_male_white", "part_circular_male_yellow",
    "part_circular_white_assembled", "part_circular_white_dissasembled", "part_circular_yellow_assembled", "part_circular_yellow_dissasembled2",
    "part_circularsq_female_white", "part_circularsq_female_yellow", "part_circularsq_male_white", "part_circularsq_white_assembled",
    "part_circularsq_white_dissasembled", "part_circularsq_yellow_assembled", "part_circularsq_yellow_dissasembled", "part_cricularsq_male_yellow",
    "part_square_female_white", "part_square_female_yellow", "part_square_male_white", "part_square_male_yellow",
    "part_square_white_assembled", "part_square_white_dissasembled", "part_square_yellow_assembled", "part_square_yellow_dissasembled",
    "scene_without_pcs"
]
clases_prompts = [f"a photo of a {name.replace('_', ' ').lower()}" for name in clases_clip]

# NODO ROS2 - SOLO PUBLISHERS 

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision')
        self.centroid_pub = self.create_publisher(Float64MultiArray, '/object_centroid', 10)
        self.count_pub = self.create_publisher(Int32, '/object_count', 10)
        self.label_pub = self.create_publisher(String, '/object_labels', 10)
        self.det_pub = self.create_publisher(ObjectDetection, '/object_detections', 10)
        self.obstacle_pub = self.create_publisher(ObjectDetection, '/obstacle_detections', 10)
        self.scene_pub = self.create_publisher(String, '/scene_description', 10)
        self.get_logger().info('Iniciando Vision Node (YOLOE + CLIP)...')

    def publish_custom_detections(self, custom_boxes, depth_matrix, depth_scale, frame_w, frame_h):
        """Publica conteo, labels y detecciones (centroid + ObjectDetection) de piezas reales."""
        object_count = len(custom_boxes)
        labels = [b[5] for b in custom_boxes]

        self.count_pub.publish(Int32(data=object_count))
        self.label_pub.publish(String(data=",".join(labels)))

        for (x1, y1, x2, y2, conf, label) in custom_boxes:
            cX, cY = (x1 + x2) // 2, (y1 + y2) // 2
            distancia_m = get_distancia_centroide(
                x1, y1, x2, y2, depth_matrix, depth_scale, frame_w, frame_h
            ) if depth_matrix is not None else 0.0

            # Topic original (compatibilidad)
            msg = Float64MultiArray()
            msg.data = [float(cX), float(cY), float(distancia_m)]
            self.centroid_pub.publish(msg)

            # Topic de detección estructurada
            det = ObjectDetection()
            det.label = label
            det.cx = float(cX)
            det.cy = float(cY)
            det.depth = float(distancia_m)
            self.det_pub.publish(det)

    def publish_obstacle_detections(self, base_boxes, depth_matrix, depth_scale, frame_w, frame_h):
        """Publica las detecciones filtradas del modelo base (obstáculos) en su propio topic."""
        for (x1, y1, x2, y2, conf, label) in base_boxes:
            cX, cY = (x1 + x2) // 2, (y1 + y2) // 2
            distancia_m = get_distancia_centroide(
                x1, y1, x2, y2, depth_matrix, depth_scale, frame_w, frame_h
            ) if depth_matrix is not None else 0.0

            det = ObjectDetection()
            det.label = "obstaculo"
            det.cx = float(cX)
            det.cy = float(cY)
            det.depth = float(distancia_m)
            self.obstacle_pub.publish(det)

    def publish_scene(self, descripcion):
        self.scene_pub.publish(String(data=descripcion))

# UTILIDADES

def get_boxes_info(results):
    """Extrae (x1, y1, x2, y2, conf, label) de los resultados de un modelo."""
    boxes_info = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            boxes_info.append((x1, y1, x2, y2, conf, label))
    return boxes_info


def compute_iou(boxA, boxB):
    """Calcula IoU entre dos cajas (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if areaA + areaB - inter_area == 0:
        return 0.0
    return inter_area / (areaA + areaB - inter_area)


def compute_containment(boxA, boxB):
    """
    Qué fracción del área de la caja MÁS CHICA (entre A y B) está cubierta
    por la intersección. A diferencia del IoU, esto sí detecta el caso de
    una caja grande que engloba a una chica, aunque el IoU entre ambas sea bajo.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_menor = min(areaA, areaB)

    if area_menor == 0:
        return 0.0
    return inter_area / area_menor


def deduplicate_nested_boxes(boxes_info, containment_threshold=0.75):
    """
    Elimina cajas del BASE que en realidad son la MISMA detección repetida
    con distinto tamaño.Se queda con la caja más chica y ajustada de cada
    grupo, que normalmente es la correcta.
    """
    boxes_ordenadas = sorted(boxes_info, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    conservadas = []
    for box in boxes_ordenadas:
        es_duplicado = False
        for box_conservada in conservadas:
            if compute_containment(box[:4], box_conservada[:4]) > containment_threshold:
                es_duplicado = True
                break
        if not es_duplicado:
            conservadas.append(box)
    return conservadas


def filter_oversized_boxes(boxes_info, max_area_ratio, frame_width, frame_height):
    """
    Descarta cajas cuya área supere max_area_ratio del área total del frame.
    """
    frame_area = frame_width * frame_height
    filtrado = []
    for box in boxes_info:
        x1, y1, x2, y2, conf, label = box
        area = (x2 - x1) * (y2 - y1)
        if area / frame_area <= max_area_ratio:
            filtrado.append(box)
    return filtrado


def filter_base_detections(custom_boxes, base_boxes, iou_threshold):
    """Elimina detecciones del base_model que se solapen demasiado con el custom_model."""
    filtered = []
    for base_box in base_boxes:
        overlap_found = False
        for custom_box in custom_boxes:
            iou = compute_iou(base_box[:4], custom_box[:4])
            if iou > iou_threshold:
                overlap_found = True
                break
        if not overlap_found:
            filtered.append(base_box)
    return filtered


def get_distancia_centroide(x1, y1, x2, y2, depth_matrix, depth_scale, frame_width, frame_height, radio=RADIO_CENTROIDE):
    """
    Mide la distancia (en metros) hacia el centroide de una caja, usando la
    mediana de una pequeña ventana de píxeles alrededor del centro.
    """
    cX, cY = (x1 + x2) // 2, (y1 + y2) // 2
    ymin, ymax = max(0, cY - radio), min(frame_height, cY + radio + 1)
    xmin, xmax = max(0, cX - radio), min(frame_width, cX + radio + 1)

    zona_centroide = depth_matrix[ymin:ymax, xmin:xmax]
    valores_validos = zona_centroide[zona_centroide > 0]

    if len(valores_validos) > 0:
        return np.median(valores_validos) * depth_scale
    return 0.0


def draw_detections(frame, boxes_info, color, depth_matrix=None, depth_scale=None,
                     prefix="", override_label=None):
    """
    Dibuja las detecciones sobre el frame. Si se pasan depth_matrix y
    depth_scale.
    """
    frame_height, frame_width = frame.shape[:2]
    for (x1, y1, x2, y2, conf, label) in boxes_info:
        display_label = override_label if override_label is not None else label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if depth_matrix is not None and depth_scale is not None:
            distancia_m = get_distancia_centroide(
                x1, y1, x2, y2, depth_matrix, depth_scale, frame_width, frame_height
            )
            text = f"[{prefix}] {display_label} {conf:.2f} | {distancia_m:.2f}m"
        else:
            text = f"[{prefix}] {display_label} {conf:.2f}"

        cv2.putText(frame, text, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# CLIP EN HILO APARTE (ahora recibe el nodo para publicar directo al terminar)

ultima_descripcion_clip = "INICIALIZANDO"
clip_bloqueado = False
lock = threading.Lock()


def hilo_inferencia_clip(model_clip, preprocess, text_tokens, img_para_clip, vision_node):
    
    global ultima_descripcion_clip, clip_bloqueado
    try:
        pil_img = Image.fromarray(img_para_clip)
        img_input = preprocess(pil_img).unsqueeze(0).to(CLIP_DEVICE)

        with torch.no_grad():
            image_features = model_clip.encode_image(img_input)
            text_features = model_clip.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            idx_max = similarity.argmax().item()

            temp_desc = clases_clip[idx_max].replace("_", " ").upper()

            with lock:
                ultima_descripcion_clip = temp_desc

            vision_node.publish_scene(temp_desc)
    finally:
        clip_bloqueado = False

# MAIN

def main(args=None):
    global clip_bloqueado

    torch.set_num_threads(TORCH_THREADS)

    rclpy.init(args=args)
    vision_node = VisionNode()

    print("Cargando modelo custom")
    custom_model = YOLOE(CUSTOM_MODEL_PATH)
    custom_model.to(DEVICE)
    print(f"Modelo custom listo con {len(custom_model.names)} clases fijas.")

    print("Cargando modelo base")
    base_model = YOLOE(BASE_MODEL_PATH)
    base_model.to(DEVICE)
    base_model.set_classes(BASE_PROMPTS)
    print(f"Modelo base listo con {len(BASE_PROMPTS)} prompts: {BASE_PROMPTS}")

    print(f"Cargando CLIP en {CLIP_DEVICE.upper()}...")
    model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model_clip.load_state_dict(torch.load(CLIP_MODEL_PATH, map_location=CLIP_DEVICE, weights_only=True))
    model_clip.to(CLIP_DEVICE)
    model_clip.eval()
    text_tokens = tokenizer(clases_prompts).to(CLIP_DEVICE)
    print("CLIP listo.")

    # Configurar pipeline de Intel RealSense (color + profundidad alineados)
    pipeline = rs.pipeline()
    config = rs.config()
    WIDTH, HEIGHT = 640, 480
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

    print("Iniciando cámara RealSense...")
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    frame_count = 0
    base_boxes_raw = []
    base_boxes_filtered = []
    depth_matrix = None

    prev_time = time.time()
    fps = 0.0

    try:
        while rclpy.ok():
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            frame_h, frame_w = frame.shape[:2]

            # La profundidad se recalcula solo cada DEPTH_EVERY_N_FRAMES frames.
            if frame_count % DEPTH_EVERY_N_FRAMES == 0:
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                if depth_frame:
                    depth_matrix = np.asanyarray(depth_frame.get_data())

            # El modelo custom corre en cada frame
            results_custom = custom_model.predict(
                frame, conf=CUSTOM_CONF, imgsz=IMG_SIZE, device=DEVICE, verbose=False
            )
            custom_boxes = get_boxes_info(results_custom)

            # El modelo base solo corre cada N frames.
            if frame_count % BASE_MODEL_EVERY_N_FRAMES == 0:
                results_base = base_model.predict(
                    frame, conf=BASE_CONF, imgsz=IMG_SIZE, device=DEVICE, verbose=False,
                    agnostic_nms=True, iou=BASE_NMS_IOU
                )
                base_boxes_raw = get_boxes_info(results_base)
                base_boxes_raw = filter_oversized_boxes(
                    base_boxes_raw, MAX_BASE_BOX_AREA_RATIO,
                    frame_width=frame_w, frame_height=frame_h
                )
                base_boxes_raw = deduplicate_nested_boxes(base_boxes_raw, containment_threshold=0.75)

            base_boxes_filtered = filter_base_detections(
                custom_boxes, base_boxes_raw, IOU_OVERLAP_THRESHOLD
            )

            #Publicación ROS2 
            vision_node.publish_custom_detections(
                custom_boxes, depth_matrix, depth_scale, frame_w, frame_h
            )
            vision_node.publish_obstacle_detections(
                base_boxes_filtered, depth_matrix, depth_scale, frame_w, frame_h
            )

            # Dibujar resultado combinado (con distancia al centroide de cada caja)
            annotated_frame = frame.copy()
            annotated_frame = draw_detections(
                annotated_frame, custom_boxes, COLOR_CUSTOM,
                depth_matrix=depth_matrix, depth_scale=depth_scale
            )
            annotated_frame = draw_detections(
                annotated_frame, base_boxes_filtered, COLOR_BASE,
                depth_matrix=depth_matrix, depth_scale=depth_scale,
                override_label="obstaculo"
            )

            #cada FRECUENCIA_CLIP frames, en hilo aparte 
            if frame_count % FRECUENCIA_CLIP == 0 and not clip_bloqueado:
                clip_bloqueado = True
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hilo = threading.Thread(
                    target=hilo_inferencia_clip,
                    args=(model_clip, preprocess, text_tokens, img_rgb.copy(), vision_node),
                    daemon=True
                )
                hilo.start()

            frame_count += 1

            # Dibujar el último resultado conocido de CLIP 
            with lock:
                escena_actual = ultima_descripcion_clip

            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (30, 30, 30), -1)
            cv2.putText(annotated_frame, f"ESCENA GLOBAL: {escena_actual}", (10, 25),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)

            # Contador de FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
            prev_time = curr_time
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (annotated_frame.shape[1] - 100, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("YOLOE + CLIP ", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Cerrando cámara")
        pipeline.stop()
        cv2.destroyAllWindows()
        vision_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()