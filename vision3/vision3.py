"""
Detección de piezas (YOLOE custom + CLIP) y obstáculos (YOLOE open-vocabulary)
con profundidad (RealSense) para ambos, y segmentación visual de obstáculos.
"""

import time
import threading
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import open_clip
from PIL import Image
from ultralytics import YOLOE

# CONFIGURACIÓN

# Modelos YOLOE 
CUSTOM_MODEL_PATH = "best yoloe jueves).pt"
BASE_MODEL_PATH = "yoloe-26s-seg.pt"
CUSTOM_CONF = 0.25
BASE_CONF = 0.3

#Prompts utilizados en yoloe-26s-seg.pt
BASE_PROMPTS = [
    "white block on a gray workbench",
    "metal T-shaped pin on a workbench",
    "silvery circular piece with four crosswise holes and a circular bar on a work table",
    "light blue box",
    "black rectangular block on a workbench",
]

#CLIP
CLIP_MODEL_PATH = "best_clip_V7.pt"
CLIP_DEVICE = "cpu"
FRECUENCIA_CLIP = 5
CLIP_CROP_PADDING = 10
CLIP_CROP_MIN_SIZE = 20

#Prompts utilizados en CLIP
clases_clip = [
    "part_squarerounded_yellow_dissasembled", "part_squarerounded_yellow_assembled",
    "part_squarerounded_white_dissasembled", "part_squarerounded_white_assembled",
    "part_square_yellow_dissasembled", "part_square_yellow_assembled",
    "part_square_white_dissasembled", "part_square_white_assembled",
    "part_circular_yellow_dissasembled", "part_circular_yellow_assembled",
    "part_circular_white_dissasembled", "part_circular_white_assembled",
]
clases_prompts = [f"a photo of a {name.replace('_', ' ').lower()}" for name in clases_clip]

#Filtros 
IOU_OVERLAP_THRESHOLD = 0.3         
CLIP_MATCH_IOU_THRESHOLD = 0.15      
BASE_NMS_IOU = 0.5
MAX_BASE_BOX_AREA_RATIO = 0.35
NESTED_CONTAINMENT_THRESHOLD = 0.75

#Colores / dibujo
COLOR_BASE = (0, 0, 255)
COLOR_CLIP = (0, 200, 255)
COLOR_YOLO = (255, 0, 0)
COLOR_PIEZA_MASK = (255, 0, 0)
MASK_ALPHA = 0.35
RADIO_CENTROIDE = 4

# --- Rendimiento ---
DEVICE = "cpu"
IMG_SIZE = 320
TORCH_THREADS = 3
BASE_MODEL_EVERY_N_FRAMES = 3
DEPTH_EVERY_N_FRAMES = 3

#Segmentación 
DRAW_OBSTACLE_MASKS = True
DRAW_PIEZA_MASKS = True

# UTILIDADES GEOMÉTRICAS
def compute_iou(box_a, box_b):
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union else 0.0

def compute_containment(box_a, box_b):
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    area_menor = min(area_a, area_b)

    return inter / area_menor if area_menor else 0.0


def get_distancia_centroide(bbox, depth_matrix, depth_scale, frame_w, frame_h, radio=RADIO_CENTROIDE):
    """Distancia (m) usando la mediana de una ventana alrededor del centroide de la caja."""
    if depth_matrix is None or depth_scale is None:
        return 0.0

    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    ymin, ymax = max(0, cy - radio), min(frame_h, cy + radio + 1)
    xmin, xmax = max(0, cx - radio), min(frame_w, cx + radio + 1)

    zona = depth_matrix[ymin:ymax, xmin:xmax]
    validos = zona[zona > 0]

    return float(np.median(validos) * depth_scale) if len(validos) else 0.0

# EXTRACCIÓN DE DETECCIONES (YOLO)

def get_detections(results, with_masks=False):

    detections = []

    for r in results:
        if r.boxes is None:
            continue

        masks_xy = r.masks.xy if (with_masks and r.masks is not None) else None

        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = r.names[int(box.cls[0])]

            mask = masks_xy[i].astype(np.int32) if masks_xy is not None else None

            detections.append({"bbox": (x1, y1, x2, y2), "conf": conf, "label": label, "mask": mask})

    return detections


def filter_oversized_boxes(detections, max_area_ratio, frame_w, frame_h):
    frame_area = frame_w * frame_h
    return [
        d for d in detections
        if ((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])) / frame_area <= max_area_ratio
    ]


def deduplicate_nested_boxes(detections, containment_threshold=NESTED_CONTAINMENT_THRESHOLD):
    ordenadas = sorted(detections, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))

    conservadas = []
    for d in ordenadas:
        if not any(compute_containment(d["bbox"], c["bbox"]) > containment_threshold for c in conservadas):
            conservadas.append(d)
    return conservadas


def filter_base_detections(custom_dets, base_dets, iou_threshold):
    """Descarta obstáculos (base) que en realidad ya son piezas detectadas (custom)."""
    return [
        b for b in base_dets
        if not any(compute_iou(b["bbox"], c["bbox"]) > iou_threshold for c in custom_dets)
    ]


def encontrar_deteccion_actual(bbox_original, custom_dets, iou_threshold=CLIP_MATCH_IOU_THRESHOLD):
    """vincula una caja CLIP (de un frame anterior) con la caja YOLO del frame actual."""
    mejor_det, mejor_iou = None, 0.0

    for d in custom_dets:
        iou = compute_iou(bbox_original, d["bbox"])
        if iou > mejor_iou:
            mejor_iou, mejor_det = iou, d

    return mejor_det if mejor_iou >= iou_threshold else None

# DIBUJO
def draw_masks(frame, detections, color, alpha=MASK_ALPHA):
    overlay = frame.copy()
    dibujado = False

    for d in detections:
        mask = d.get("mask")
        if mask is not None and len(mask) > 0:
            cv2.fillPoly(overlay, [mask], color)
            dibujado = True

    if dibujado:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame


def draw_detections(frame, detections, color, depth_matrix=None, depth_scale=None, prefix="", override_label=None):
    frame_h, frame_w = frame.shape[:2]

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = override_label if override_label is not None else d["label"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        texto = f"[{prefix}] {label} {d['conf']:.2f}"

        if depth_matrix is not None:
            dist = get_distancia_centroide(d["bbox"], depth_matrix, depth_scale, frame_w, frame_h)
            texto += f" | {dist:.2f}m"

        cv2.putText(frame, texto, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


# RECORTES PARA CLIP
def preparar_crops_para_clip(frame_rgb, custom_dets, padding=CLIP_CROP_PADDING, min_size=CLIP_CROP_MIN_SIZE):
    frame_h, frame_w = frame_rgb.shape[:2]
    crops, metas = [], []

    for d in custom_dets:
        x1, y1, x2, y2 = d["bbox"]
        xa, ya = max(x1 - padding, 0), max(y1 - padding, 0)
        xb, yb = min(x2 + padding, frame_w), min(y2 + padding, frame_h)

        if xb - xa < min_size or yb - ya < min_size:
            continue

        crops.append(frame_rgb[ya:yb, xa:xb])
        metas.append({"bbox": (xa, ya, xb, yb), "label": d["label"], "conf": d["conf"]})

    return crops, metas

# HILO DE INFERENCIA CLIP

ultimos_resultados_clip = []
clip_bloqueado = False
lock = threading.Lock()


def hilo_inferencia_clip(model_clip, preprocess, text_tokens, crops, metas):
    global ultimos_resultados_clip, clip_bloqueado

    try:
        if not crops:
            with lock:
                ultimos_resultados_clip = []
            return

        batch = torch.stack([preprocess(Image.fromarray(c)) for c in crops]).to(CLIP_DEVICE)

        with torch.no_grad():
            image_features = model_clip.encode_image(batch)
            text_features = model_clip.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            confs, idxs = similarity.max(dim=-1)

        resultados = [
            {
                "bbox_original": metas[i]["bbox"],
                "label_clip": clases_clip[idx],
                "clip_conf": conf,
                "label_yolo": metas[i]["label"],
                "yolo_conf": metas[i]["conf"],
            }
            for i, (idx, conf) in enumerate(zip(idxs.tolist(), confs.tolist()))
        ]

        with lock:
            ultimos_resultados_clip = resultados

    except Exception as e:
        print(f"Error en hilo CLIP: {e}")

    finally:
        clip_bloqueado = False

# CARGA DE MODELOS
def cargar_modelo_custom():
    print("Cargando modelo custom...")
    modelo = YOLOE(CUSTOM_MODEL_PATH)
    modelo.to(DEVICE)
    print(f"Modelo custom listo con {len(modelo.names)} clases fijas: {modelo.names}")
    return modelo


def cargar_modelo_base():
    print("Cargando modelo base (open-vocabulary)...")
    modelo = YOLOE(BASE_MODEL_PATH)
    modelo.to(DEVICE)
    modelo.set_classes(BASE_PROMPTS)
    print(f"Modelo base listo con {len(BASE_PROMPTS)} prompts: {BASE_PROMPTS}")
    return modelo


def cargar_clip():
    print(f"Cargando CLIP en {CLIP_DEVICE.upper()}...")
    model_clip, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model_clip.load_state_dict(torch.load(CLIP_MODEL_PATH, map_location=CLIP_DEVICE, weights_only=True))
    model_clip.to(CLIP_DEVICE)
    model_clip.eval()

    text_tokens = tokenizer(clases_prompts).to(CLIP_DEVICE)
    print("CLIP listo.")
    return model_clip, preprocess, text_tokens


def iniciar_realsense(width=1280, height=720, fps=15):
    print("Iniciando cámara RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    return pipeline, align, depth_scale

# PROCESAMIENTO POR FRAME
def procesar_obstaculos(base_model, frame, frame_count, base_dets_cache, custom_dets, frame_w, frame_h):
    """Corre el modelo base cada N frames y filtra los obstáculos que ya son piezas."""
    if frame_count % BASE_MODEL_EVERY_N_FRAMES == 0:
        results_base = base_model.predict(
            frame, conf=BASE_CONF, imgsz=IMG_SIZE, device=DEVICE,
            verbose=False, agnostic_nms=True, iou=BASE_NMS_IOU,
        )

        dets = get_detections(results_base, with_masks=DRAW_OBSTACLE_MASKS)
        dets = filter_oversized_boxes(dets, MAX_BASE_BOX_AREA_RATIO, frame_w, frame_h)
        dets = deduplicate_nested_boxes(dets)
        base_dets_cache["raw"] = dets

    return filter_base_detections(custom_dets, base_dets_cache["raw"], IOU_OVERLAP_THRESHOLD)


def lanzar_clip_si_corresponde(frame, frame_count, custom_dets, model_clip, preprocess, text_tokens):
    global clip_bloqueado

    if frame_count % FRECUENCIA_CLIP != 0 or clip_bloqueado:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crops, metas = preparar_crops_para_clip(frame_rgb, custom_dets)

    if not crops:
        with lock:
            ultimos_resultados_clip.clear()
        return

    clip_bloqueado = True
    threading.Thread(
        target=hilo_inferencia_clip,
        args=(model_clip, preprocess, text_tokens, crops, metas),
        daemon=True,
    ).start()


def dibujar_piezas_clip(frame, resultados_clip, custom_dets, depth_matrix, depth_scale):
    """Dibuja la caja actual, etiqueta YOLO+CLIP y profundidad para cada pieza detectada."""
    frame_h, frame_w = frame.shape[:2]

    for r in resultados_clip:
        det_actual = encontrar_deteccion_actual(r["bbox_original"], custom_dets)
        if det_actual is None:
            continue

        x1, y1, x2, y2 = det_actual["bbox"]

        dist = get_distancia_centroide(det_actual["bbox"], depth_matrix, depth_scale, frame_w, frame_h)

        texto_yolo = f"YOLO: {r['label_yolo'].replace('_', ' ').upper()} ({r['yolo_conf'] * 100:.0f}%)"#| {dist:.2f}m
        texto_clip = f"CLIP: {r['label_clip'].replace('_', ' ').upper()} ({r['clip_conf'] * 100:.0f}%) "
        texto_distancia = f"DISTANCIA: {dist:.2f}m"  
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_CLIP, 2)
        cv2.putText(frame, texto_yolo, (x1, max(y1 - 50, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YOLO, 2)
        cv2.putText(frame, texto_clip, (x1, max(y1 - 30, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_CLIP, 2)
        cv2.putText(frame, texto_distancia, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YOLO, 2)
        
    return frame


def dibujar_barra_superior(frame, resultados_clip, fps):
    n_assembled = sum(1 for r in resultados_clip if r["label_clip"].endswith ("_assembled"))
    n_dissasembled = sum(1 for r in resultados_clip if r["label_clip"].endswith ("_dissasembled"))

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (30, 30, 30), -1)
    cv2.putText(
        frame,
        f"assembled parts: {n_assembled} | dissasembled parts: {n_dissasembled}",
        (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 255), 1,
    )
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )
    return frame

# MAIN
def main():
    torch.set_num_threads(TORCH_THREADS)

    custom_model = cargar_modelo_custom()
    base_model = cargar_modelo_base()
    model_clip, preprocess, text_tokens = cargar_clip()
    pipeline, align, depth_scale = iniciar_realsense()

    frame_count = 0
    base_dets_cache = {"raw": []}
    depth_matrix = None
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            frame_h, frame_w = frame.shape[:2]

            #Profundidad (no en todos los frames)
            if frame_count % DEPTH_EVERY_N_FRAMES == 0:
                aligned = align.process(frames)
                depth_frame = aligned.get_depth_frame()
                if depth_frame:
                    depth_matrix = np.asanyarray(depth_frame.get_data())

            #Piezas (custom, con segmentación)
            results_custom = custom_model.predict(
                frame, conf=CUSTOM_CONF, imgsz=IMG_SIZE, device=DEVICE, verbose=False
            )
            custom_dets = get_detections(results_custom, with_masks=DRAW_PIEZA_MASKS)

            #Obstáculos (base, con segmentación)
            base_dets_filtered = procesar_obstaculos(
                base_model, frame, frame_count, base_dets_cache, custom_dets, frame_w, frame_h
            )

            annotated = frame.copy()

            if DRAW_PIEZA_MASKS:
                annotated = draw_masks(annotated, custom_dets, COLOR_PIEZA_MASK)

            if DRAW_OBSTACLE_MASKS:
                annotated = draw_masks(annotated, base_dets_filtered, COLOR_BASE)

            annotated = draw_detections(
                annotated, base_dets_filtered, COLOR_BASE,
                depth_matrix=depth_matrix, depth_scale=depth_scale, override_label="obstaculo",
            )

            #CLIP 
            lanzar_clip_si_corresponde(frame, frame_count, custom_dets, model_clip, preprocess, text_tokens)
            frame_count += 1

            with lock:
                resultados_clip_actuales = list(ultimos_resultados_clip)

            annotated = dibujar_piezas_clip(annotated, resultados_clip_actuales, custom_dets, depth_matrix, depth_scale)
            annotated = dibujar_barra_superior(annotated, resultados_clip_actuales, fps)

            #FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
            prev_time = curr_time

            cv2.imshow("YOLOE + CLIP", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("Cerrando cámara")
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()