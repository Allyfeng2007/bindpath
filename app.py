# app.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64, os, json
from concurrent.futures import ThreadPoolExecutor

# ==== 腾讯云 TTS ====
from tencentcloud.common import credential
from tencentcloud.tts.v20190823 import tts_client, models

app = Flask(__name__)

# ========================
# 配置：两套模型与阈值
# ========================
GENERAL_WEIGHTS = './yolov8n.pt'  # 通用80类
BLIND_WEIGHTS   = './best.pt'  # 你的盲道专用权重

GENERAL_CONF = 0.25   # 通用模型阈值：更稳则调高
BLIND_CONF   = 0.58   # 盲道模型阈值：取F1最佳点（你之前曲线≈0.58）
NMS_IOU      = 0.50

# ========================
# 加载两个 YOLOv8 模型
# ========================
model_general = YOLO(GENERAL_WEIGHTS)
# best.pt 可能尚未训练好或路径不同，这里做容错
try:
    model_blind = YOLO(BLIND_WEIGHTS)
except Exception as e:
    print(f'[WARN] 加载盲道模型失败：{e}\n将仅使用通用模型。')
    model_blind = None

# 线程池（并行两次推理；单GPU下主要重叠CPU预处理，GPU会串行，仍然可用）
executor = ThreadPoolExecutor(max_workers=2)


def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    xx1 = max(0, min(w - 1, x1))
    yy1 = max(0, min(h - 1, y1))
    xx2 = max(0, min(w - 1, x2))
    yy2 = max(0, min(h - 1, y2))
    if xx2 <= xx1 or yy2 <= yy1:
        return None
    return img[yy1:yy2, xx1:xx2]


def generate_alert_text(dets, img_w):
    alert = []
    for det in dets:
        cls_name = det["class"]
        x1, y1, x2, y2 = det["box"]
        if cls_name == "traffic light":
            color = det.get("color")
            if color in ("红灯", "绿灯", "黄灯"):
                alert.append(f"前方{color}")
            else:
                alert.append("前方交通灯")
        elif cls_name == "blind_path":
            alert.append("前方盲道")
        else:
            cx = (x1 + x2) / 2
            pos = "左侧" if cx < img_w * 0.3 else ("右侧" if cx > img_w * 0.7 else "前方")
            cname_zh = {
                'person': '行人', 'bicycle': '自行车', 'car': '汽车',
                'motorbike': '摩托车', 'bus': '公交车', 'truck': '卡车'
            }.get(cls_name, "障碍物")
            alert.append(f"{pos}有{cname_zh}")
    alert = list(dict.fromkeys(alert))
    return "，".join(alert) + "。" if alert else "未检测到目标"


def run_one_model(model, img, conf, iou):
    """对单个模型推理，并转成 dets 结构"""
    r = model.predict(source=img, conf=conf, iou=iou, verbose=False)[0]
    out = []
    if r.boxes is not None and len(r.boxes) > 0:
        names = r.names
        for b in r.boxes:
            cls_id = int(b.cls[0])
            cls_name = names[cls_id]
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            det = {"class": cls_name, "box": [x1, y1, x2, y2]}
            out.append(det)
    return out, r


def enrich_traffic_light_color(img, dets):
    """给 traffic light 加颜色标签"""
    for det in dets:
        if det["class"] == "traffic light":
            x1, y1, x2, y2 = det["box"]
            roi = safe_crop(img, x1, y1, x2, y2)
            if roi is not None and roi.size > 0:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | \
                           cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
                mask_green = cv2.inRange(hsv, (50, 100, 100), (90, 255, 255))
                red_count = int(cv2.countNonZero(mask_red))
                green_count = int(cv2.countNonZero(mask_green))
                if red_count > 50 and red_count > green_count:
                    det["color"] = "红灯"
                elif green_count > 50 and green_count > red_count:
                    det["color"] = "绿灯"
                else:
                    det["color"] = "黄灯"
    return dets


@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(silent=True) or {}
        image_data = body.get('image')
        if not image_data:
            return jsonify({'error': '未提供图片数据'}), 400

        # 1) 解析 base64
        data = image_data.split(',', 1)[-1] if ',' in image_data else image_data
        img_bytes = base64.b64decode(data)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': '图片解析失败'}), 400

        h, w = img.shape[:2]

        # 2) 先跑“你原来能工作的 yolov8n.pt 写法”，不要改任何默认阈值
        #    ——这一步应该能恢复你“红绿灯检测OK”的旧行为
        results_g = model_general(img, verbose=False)[0]  # 关键：沿用旧写法
        dets = []
        if results_g.boxes is not None and len(results_g.boxes) > 0:
            names_g = results_g.names  # COCO 的 80 类名
            for b in results_g.boxes:
                cls_id = int(b.cls[0])
                cls_name = names_g[cls_id]
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                dets.append({"class": cls_name, "box": [x1, y1, x2, y2], "source": "general"})

        # 3) 再跑你的 best.pt（如果加载成功）
        if model_blind is not None:
            # 为避免阈值过高导致“明明能检却被过滤”，这里也先用默认阈值（与旧版一致）
            results_b = model_blind(img, verbose=False)[0]
            if results_b.boxes is not None and len(results_b.boxes) > 0:
                names_b = results_b.names  # 一般是 ['blind_path']
                for b in results_b.boxes:
                    cls_id = int(b.cls[0])
                    cls_name = names_b[cls_id]
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    dets.append({"class": cls_name, "box": [x1, y1, x2, y2], "source": "blind"})

        # 4) 如果两边都没检到，做一次“降阈值重试”（只重试一次，避免反复）
        if len(dets) == 0:
            # 降 conf + 提大 imgsz；这两步对红绿灯/细长目标很有效
            results_g2 = model_general(img, conf=0.15, imgsz=960, iou=0.5, verbose=False)[0]
            if results_g2.boxes is not None and len(results_g2.boxes) > 0:
                names_g = results_g2.names
                for b in results_g2.boxes:
                    cls_id = int(b.cls[0])
                    cls_name = names_g[cls_id]
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    dets.append({"class": cls_name, "box": [x1, y1, x2, y2], "source": "general"})

            if model_blind is not None:
                results_b2 = model_blind(img, conf=0.35, imgsz=960, iou=0.5, verbose=False)[0]
                if results_b2.boxes is not None and len(results_b2.boxes) > 0:
                    names_b = results_b2.names
                    for b in results_b2.boxes:
                        cls_id = int(b.cls[0])
                        cls_name = names_b[cls_id]
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        dets.append({"class": cls_name, "box": [x1, y1, x2, y2], "source": "blind"})

        # 5) 给 traffic light 加颜色（不影响“是否检测到”的判断）
        dets = enrich_traffic_light_color(img, dets)

        # 6) 生成播报
        text = generate_alert_text(dets, img_w=w)

        return jsonify({
            'result': text,
            'detections': dets,
            'meta': {
                'general_boxes': int(len([d for d in dets if d["source"] == "general"])),
                'blind_boxes':   int(len([d for d in dets if d["source"] == "blind"])),
                'blind_model_loaded': model_blind is not None
            }
        })

    except Exception as e:
        return jsonify({'error': f'处理失败: {e}'}), 500



@app.route('/tts', methods=['POST'])
def tts():
    try:
        body = request.get_json(silent=True) or {}
        text = (body.get('text') or '').strip()
        if not text:
            return jsonify({'error': '缺少 text'}), 400

        sid = os.environ.get("TENCENTCLOUD_SECRET_ID")
        sk  = os.environ.get("TENCENTCLOUD_SECRET_KEY")
        if not sid or not sk:
            return jsonify({'error': '未配置腾讯云密钥(TENCENTCLOUD_SECRET_ID/KEY)'}), 500

        cred = credential.Credential(sid, sk)
        client = tts_client.TtsClient(cred, "ap-guangzhou")

        req = models.TextToVoiceRequest()
        params = {
            "Text": text[:500],
            "SessionId": "s1",
            "ModelType": 1,
            "VoiceType": 101001,
            "Codec": "mp3",
            "SampleRate": 16000,
            "Speed": 0,
            "Volume": 0
        }
        req.from_json_string(json.dumps(params))
        resp = client.TextToVoice(req)

        return jsonify({'audio_base64': 'data:audio/mpeg;base64,' + resp.Audio})
    except Exception as e:
        return jsonify({'error': f'TTS失败: {e}'}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'yolo-detection'})


@app.route('/routes', methods=['GET'])
def routes():
    return jsonify(sorted([f"{sorted(r.methods)} {r.rule}" for r in app.url_map.iter_rules()]))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
