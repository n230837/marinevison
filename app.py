import os
import json
import numpy as np
import pandas as pd
import cv2
import folium
import gradio as gr
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best(1).onnx")
MAP_PATH = os.path.join(BASE_DIR, "marine_debris_offshore_map.html")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

# Same 100-point demo offshore map as our Colab prototype
np.random.seed(42)
m = folium.Map(location=[13.08, 80.36], zoom_start=10)

for i in range(100):
    lat = np.random.uniform(12.98, 13.18)
    lon = np.random.uniform(80.30, 80.42)
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        popup=f"Marine Anomaly #{i + 1}",
        fill=True
    ).add_to(m)

m.save(MAP_PATH)


def detect_and_report(image, confidence):
    if image is None:
        return None, "No image uploaded", "—", "—", None, None

    results = model.predict(
        source=image,
        conf=float(confidence),
        imgsz=640,
        verbose=False
    )
    result = results[0]
    annotated = result.plot()

    if annotated is not None:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf_score = float(box.conf[0])

            detections.append({
                "class": "Crab-Pot",
                "confidence": round(conf_score, 4),
                "bbox": [round(float(x), 2) for x in xyxy]
            })

    count = len(detections)

    if count:
        max_confidence = max(d["confidence"] for d in detections)
        risk = (
            "HIGH" if max_confidence >= 0.50
            else "MEDIUM" if max_confidence >= 0.30
            else "LOW"
        )
    else:
        max_confidence = 0.0
        risk = "NO DETECTION"

    latitude = 13.08
    longitude = 80.36

    report = {
        "system": "MarineVision",
        "problem_statement": "SIH26057",
        "analysis_type": "Side-Scan Sonar Marine Debris Detection",
        "gps_source": "SIMULATED OFFSHORE DEMO GPS",
        "latitude": latitude,
        "longitude": longitude,
        "detection_count": count,
        "maximum_confidence": round(max_confidence, 4),
        "risk_level": risk,
        "detections": detections
    }

    json_path = os.path.join(REPORT_DIR, "marine_debris_anomaly_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    csv_path = os.path.join(REPORT_DIR, "marine_debris_anomaly_report.csv")

    if detections:
        rows = []
        for i, d in enumerate(detections, start=1):
            x1, y1, x2, y2 = d["bbox"]
            rows.append({
                "Anomaly_ID": f"MV-{i:03d}",
                "Class": d["class"],
                "Confidence": d["confidence"],
                "Risk_Level": risk,
                "Latitude": latitude,
                "Longitude": longitude,
                "GPS_Source": "SIMULATED OFFSHORE DEMO GPS",
                "BBox_X1": x1,
                "BBox_Y1": y1,
                "BBox_X2": x2,
                "BBox_Y2": y2
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    else:
        pd.DataFrame([{
            "Anomaly_ID": "NONE",
            "Class": "No Detection",
            "Confidence": 0,
            "Risk_Level": risk,
            "Latitude": latitude,
            "Longitude": longitude,
            "GPS_Source": "SIMULATED OFFSHORE DEMO GPS"
        }]).to_csv(csv_path, index=False)

    return (
        annotated,
        f"{count} marine debris object(s) detected",
        f"{max_confidence:.2f}",
        risk,
        json_path,
        csv_path
    )


with gr.Blocks(title="MarineVision", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🌊 MarineVision
    ### AI-Powered Underwater Marine Debris Detection

    **Side-Scan Sonar • YOLO Detection • Anomaly Reporting • Geospatial Mapping**
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="numpy",
                label="Upload Side-Scan Sonar Image"
            )

            confidence = gr.Slider(
                minimum=0.05,
                maximum=0.80,
                value=0.25,
                step=0.05,
                label="Detection Confidence"
            )

            detect_btn = gr.Button(
                "🔍 Detect Marine Debris",
                variant="primary"
            )

        with gr.Column():
            output_image = gr.Image(label="Detection Result")
            detection_count = gr.Textbox(label="Detections")
            max_conf = gr.Textbox(label="Maximum Confidence")
            risk = gr.Textbox(label="Risk Level")

    gr.Markdown("---")
    gr.Markdown("## 📄 Anomaly Reports")
    gr.Markdown(
        "Generate downloadable structured reports after running detection."
    )

    with gr.Row():
        json_download = gr.File(label="⬇️ JSON Report")
        csv_download = gr.File(label="⬇️ CSV Report")

    detect_btn.click(
        fn=detect_and_report,
        inputs=[input_image, confidence],
        outputs=[
            output_image,
            detection_count,
            max_conf,
            risk,
            json_download,
            csv_download
        ]
    )

    gr.Markdown("---")
    gr.Markdown("""
    ## 🗺️ Geospatial Anomaly Map

    **Prototype note:** the current training dataset does not expose real GPS
    metadata, so the map uses simulated offshore demo coordinates.
    In production, coordinates would come from the sonar platform's
    navigation/INS/GNSS metadata.
    """)

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        gr.HTML(f.read())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
