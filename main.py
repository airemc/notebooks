import os
import cv2
import numpy as np
import torch
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Detectron2 Bileşenleri
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Uygulama Başlatma
app = FastAPI(title="HematosAI Blood Cell Detection API (Cloud Production)")

# CORS Ayarları (React Frontend'in sunucuya erişebilmesi için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Güvenlik için canlıda buraya React'ın domainini yazabilirsin
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. MODEL YAPILANDIRMASI (CONFIG)
# ==========================================
cfg = get_cfg()
cfg.merge_from_file("./config.yaml")

# ⚠️ DİKKAT: Cloud sunucuna yüklediğin model ağırlığının tam yolunu buraya yaz
cfg.MODEL.WEIGHTS = "./model_final.pth" 

# 🔥 NATIVE ZOOM AYARI: Hücreleri modelin gözüne sokuyoruz
cfg.INPUT.MIN_SIZE_TEST = 1200 
cfg.INPUT.MAX_SIZE_TEST = 2000

# Eşiği 0.35'e çektik ki gizlenen/utangaç hücreler de ekrana yansısın
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.35  
cfg.MODEL.DEVICE = "cpu"

# Model Cloud'un belleğine yükleniyor
print("🚀 Model sunucuda belleğe yükleniyor, lütfen bekleyin...")
predictor = DefaultPredictor(cfg)

# Metadata Tanımları
class_names = ["BAND CELLS", "BASOPHILS", "BLAST CELLS", "EOSINOPHILS", "IG", "LYMPHOCYTES", "MONOCYTES", "NEUTROPHILS"]
MetadataCatalog.get("blood_cell_cloud").set(thing_classes=class_names)
metadata = MetadataCatalog.get("blood_cell_cloud")

# ==========================================
# 2. POST-PROCESSING (FİLTRELER)
# ==========================================
def apply_post_processing(instances, area_threshold=800, iou_threshold=0.3):
    """
    1. Alan Filtresi: Yanlışlıkla hücre sanılan küçük trombositleri siler.
    2. Class-Agnostic NMS: Bir hücrede hem IG hem Blast varsa düşük puanlıyı ezer.
    """
    scores = instances.scores
    bboxes = instances.pred_boxes.tensor
    areas = instances.pred_boxes.area()

    # Adım 1: Alanı küçük olanları (kir/trombosit) filtrele
    keep_area = areas > area_threshold
    
    # Adım 2: Sınıf Bağımsız Agresif NMS
    keep_nms = torch.ops.torchvision.nms(bboxes, scores, iou_threshold)
    
    # İki filtreyi kesiştir
    final_keep = []
    for idx in keep_nms:
        if keep_area[idx]:
            final_keep.append(idx.item())
            
    return instances[final_keep]

# ==========================================
# 3. API ENDPOINT
# ==========================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Görseli belleğe al
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Geçersiz görsel dosyası.")

        # Modeli çalıştır
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        # Filtreleri uygula (Çöplüğü temizle)
        clean_instances = apply_post_processing(instances)

        # Temizlenmiş maskeleri çiz
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(clean_instances)
        res_img = out.get_image()[:, :, ::-1]
        
        # React arayüzü için Base64 formata çevir
        _, buffer = cv2.imencode('.jpg', res_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Dashboard istatistikleri için hücreleri say
        counts = {}
        for cls_id in clean_instances.pred_classes:
            name = class_names[cls_id]
            counts[name] = counts.get(name, 0) + 1

        return {
            "success": True,
            "image": f"data:image/jpeg;base64,{img_base64}",
            "counts": counts,
            "total_detected": len(clean_instances)
        }
    except Exception as e:
        print(f"Sunucu Hatası: {str(e)}")
        raise HTTPException(status_code=500, detail="Görüntü işlenirken bir hata oluştu.")

# ==========================================
# 4. SUNUCU BAŞLATMA
# ==========================================
if __name__ == "__main__":
    import uvicorn
    # Standart, temiz ve her IP'ye açık Cloud başlatma komutu
    uvicorn.run(app, host="0.0.0.0", port=8000)