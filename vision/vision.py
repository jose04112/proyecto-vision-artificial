import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import open_clip
from ultralytics import YOLO
from PIL import Image
import time
import threading

# 1. CONFIGURACIÓN DE MODELOS 
device = "cuda" if torch.cuda.is_available() else "cpu"

# MODELO DE YOLO: 
model_yolo = YOLO("best_YOLO_V4.pt") 

# CLIP: Clasificación de escena global (OpenCLIP)
model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# modelo CLIP 
model_clip.load_state_dict(torch.load("best_clip_v6.pt", map_location=device, weights_only=True))
model_clip.to(device)
model_clip.eval()


# Clases de CLIP
clases_clip = [
   "part_circular_female_white", "part_circular_female_yellow", "part_circular_male_white", "part_circular_male_yellow",
   "part_circular_white_assembled", "part_circular_white_dissasembled", "part_circular_yellow_assembled", "part_circular_yellow_dissasembled2",
   "part_circularsq_female_white", "part_circularsq_female_yellow", "part_circularsq_male_white", "part_circularsq_white_assembled",
   "part_circularsq_white_dissasembled", "part_circularsq_yellow_assembled", "part_circularsq_yellow_dissasembled", "part_cricularsq_male_yellow",
   "part_square_female_white", "part_square_female_yellow", "part_square_male_white", "part_square_male_yellow",
   "part_square_white_assembled", "part_square_white_dissasembled", "part_square_yellow_assembled", "part_square_yellow_dissasembled",
   "scene_without_pcs"
]

# Reemplaza los guiones bajos por espacios y convierte a minúsculas para estructurar los prompts de CLIP.
clases_prompts = [f"a photo of a {name.replace('_', ' ').lower()}" for name in clases_clip]
text_tokens = tokenizer(clases_prompts).to(device)

# 2. CONFIGURAR CÁMARA REALSENSE
pipeline = rs.pipeline()
config = rs.config()
WIDTH, HEIGHT = 640, 480 #formatos aceptados desde el viewer 
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

profile = pipeline.start(config) #encender la cámara 
align = rs.align(rs.stream.color) #junta la imagen de color y profundidad 

spatial_filter = rs.spatial_filter()   # Quita los píxeles con ruido que parpadean en la pantalla
temporal_filter = rs.temporal_filter() # Compara con los frames anteriores para hacer un promedio en la altura
hole_filling = rs.hole_filling_filter(2) # Rellena los huecos negros donde la cámara no alcanza a medir

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale() # Obtiene la escala de la cámara para convertir los datos a metros r

# 3. CONFIGURACIÓN GEOMÉTRICA MÉTRICA DE OBSTÁCULOS 
UMBRAL_MIN_DISTANCIA_M = 0.08   # A partir de 8 cm alejándose del lente de la cámara empieza a detectar 
TOLERANCIA_RELIEVE_M = 0.02     # Margen mínimo para empezar a detectar volumen (2 cm), para no detectar la mesa como obstáculo 
ALTURA_MAX_ENSAMBLE_M = 0.042   # altura promedio de un ensmable (4.2 cm). Si mide más, es obstáculo.

#si un obstáculo mide menos de 15x15 píxeles, el programa no lo detecta como obstáculo 
MIN_ANCHO_OBSTACULO = 15
MIN_ALTO_OBSTACULO = 15

#4. VARIABLES DE CONTROL Y MANEJO DE THREADS  
contador_frames = 0
FRECUENCIA_CLIP = 12  #CLIP trabaja cada 12 frames 
ultima_descripcion_clip = "INICIALIZANDO..."

frames_vacio_consecutivos = 0 # contador de frames con la mesa sin piezas 
UMBRAL_CONFIRMACION_VACIO = 15 #limite de frames con la mesa vacia para decidir que no hay piezas 

# Estabilizador del plano de la mesa (Filtro de Promedio Móvil)
distancia_suavizada_mesa = 0.67  # Valor inicial de respaldo
ALFA_SUAVIZADO = 0.10            #"filtro" frente a ruido de barrenos

clip_bloqueado = False  
lock = threading.Lock() 


# [HILO SECUNDARIO: clasificación de CLIP]

def hilo_inferencia_clip(img_para_clip, total_vacio_actual, hay_piezas):
    global ultima_descripcion_clip, clip_bloqueado
    try:
        if total_vacio_actual >= UMBRAL_CONFIRMACION_VACIO:
            with lock:
                ultima_descripcion_clip = "SCENE WITHOUT PCS"
            return

        pil_img = Image.fromarray(img_para_clip)
        img_input = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model_clip.encode_image(img_input)
            text_features = model_clip.encode_text(text_tokens)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            idx_max = similarity.argmax().item()
            
            temp_desc = clases_clip[idx_max].replace("_", " ").upper()
            
            with lock:
                if temp_desc == "SCENE WITHOUT PCS" and not hay_piezas:
                    ultima_descripcion_clip = "SCENE WITHOUT PCS"
                elif temp_desc != "SCENE WITHOUT PCS":
                    ultima_descripcion_clip = temp_desc
    finally:
        clip_bloqueado = False

print(f"Sistema iniciado en {device.upper()}.")


# [HILO PRINCIPAL: VIDEO EN TIEMPO REAL, YOLO Y GEOMETRÍA 3D]

try:
    while True:
        start_time = time.time()
        
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()

        frame = np.asanyarray(color_frame.get_data())
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_matrix = np.asanyarray(depth_frame.get_data())

        # Calibración dinámica suavizada de la mesa
        depth_meters = depth_matrix * depth_scale
        X_MIN, X_MAX = 0, int(WIDTH * 0.74)  
        Y_MIN, Y_MAX = 0, int(HEIGHT * 0.90) 
        
        zona_mesa = depth_meters[Y_MIN:Y_MAX, X_MIN:X_MAX]
        valores_validos_mesa = zona_mesa[(zona_mesa > 0.15) & (zona_mesa < 1.5)] 
        
        if len(valores_validos_mesa) > 0:
            distancia_instantanea = np.median(valores_validos_mesa)
            distancia_suavizada_mesa = (ALFA_SUAVIZADO * distancia_instantanea) + ((1 - ALFA_SUAVIZADO) * distancia_suavizada_mesa)
        
        UMBRAL_MAX_DINAMICO = distancia_suavizada_mesa - TOLERANCIA_RELIEVE_M
        
        # Generar máscara base de relieve
        mascara_base = ((depth_meters > UMBRAL_MIN_DISTANCIA_M) & (depth_meters <= UMBRAL_MAX_DINAMICO)).astype(np.uint8) * 255
        
        roi_mask = np.zeros_like(mascara_base)
        roi_mask[Y_MIN:Y_MAX, X_MIN:X_MAX] = 255
        mascara_obstaculos = cv2.bitwise_and(mascara_base, roi_mask)

        # Inferencia de YOLO
        results = model_yolo(frame, conf=0.58, verbose=False)
        hay_piezas_en_este_frame = False
        
        for r in results:
            if r.masks is not None and len(r.masks.xy) > 0:
                hay_piezas_en_este_frame = True

        if hay_piezas_en_este_frame:
            frames_vacio_consecutivos = 0
        else:
            frames_vacio_consecutivos += 1

        # Exclusión geométrica (Borra piezas de YOLO para que no sea detectado como obstáculo)
        for r in results:
            if r.masks is not None:
                for i in range(len(r.masks.xy)):
                    mask_coords = r.masks.xy[i]
                    polygon = np.array(mask_coords, dtype=np.int32)
                    cv2.fillPoly(mascara_obstaculos, [polygon], 0)
                    
                    box = r.boxes[i]
                    conf = float(box.conf[0])
                    label_yolo = model_yolo.names[int(box.cls)]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cX, cY = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    radio = 4
                    ymin, ymax = max(0, cY - radio), min(HEIGHT, cY + radio + 1)
                    xmin, xmax = max(0, cX - radio), min(WIDTH, cX + radio + 1)
                    zona_centroide = depth_matrix[ymin:ymax, xmin:xmax]
                    valores_validos = zona_centroide[zona_centroide > 0]
                    
                    distancia_m = np.median(valores_validos) * depth_scale if len(valores_validos) > 0 else 0.0

                    # Dibujar YOLO
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [polygon], (0, 255, 0)) 
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    cv2.polylines(frame, [polygon], True, (255, 255, 255), 1)

                    cv2.putText(frame, f"{label_yolo.upper()}", (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"{distancia_m:.2f}m | {conf:.1%}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # LIMPIEZA DE IMAGEN: Borra manchas falsas de la cámara 
        kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mascara_obstaculos = cv2.morphologyEx(mascara_obstaculos, cv2.MORPH_OPEN, kernel_3x3)
        
        kernel_9x9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mascara_obstaculos = cv2.morphologyEx(mascara_obstaculos, cv2.MORPH_CLOSE, kernel_9x9)
        mascara_obstaculos = cv2.dilate(mascara_obstaculos, kernel_3x3, iterations=1)
        
        # Encontrar contornos de obstáculos candidatos
        contornos, _ = cv2.findContours(mascara_obstaculos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contornos:
            ox, oy, ow, oh = cv2.boundingRect(c)
            
            # ignora el perimetro de la imagen 5 pixeles 
            if ox <= X_MIN + 5 or oy <= Y_MIN + 5 or (ox + ow) >= X_MAX - 5 or (oy + oh) >= Y_MAX - 5:
                continue
                
            if ow > MIN_ANCHO_OBSTACULO and oh > MIN_ALTO_OBSTACULO:
                zona_obstaculo_m = depth_meters[oy:oy+oh, ox:ox+ow]
                valores_reales = zona_obstaculo_m[zona_obstaculo_m > 0.10]
                
                if len(valores_reales) > 0:
                    # Discriminación por altura (Saber si es el ensamble inicial y no falsos positivos)
                    altura_real_objeto_m = distancia_suavizada_mesa - np.min(valores_reales)
                    
                    if altura_real_objeto_m > ALTURA_MAX_ENSAMBLE_M:
                        # calcular distancia hacia el obstáculo
                        distancia_obstaculo_m = np.min(valores_reales)
                        
                        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 2)
                        cv2.putText(frame, f"OBSTACULO  {distancia_obstaculo_m:.2f}m", 
                                    (ox, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # llamar a CLIP
        contador_frames += 1
        if contador_frames % FRECUENCIA_CLIP == 0:
            if not clip_bloqueado:
                clip_bloqueado = True
                hilo = threading.Thread(
                    target=hilo_inferencia_clip, 
                    args=(img_rgb.copy(), frames_vacio_consecutivos, hay_piezas_en_este_frame),
                    daemon=True
                )
                hilo.start()

        # mostrar texto en la pantalla 
        with lock:
            escena_actual = ultima_descripcion_clip

        cv2.rectangle(frame, (0, 0), (WIDTH, 40), (30, 30, 30), -1)
        cv2.putText(frame, f"ESCENA GLOBAL: {escena_actual}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
        
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (WIDTH - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Manufactura: YOLO (Piezas) + CLIP (Escena)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()