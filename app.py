import os
import json
import tempfile

import cv2
import numpy as np
import pandas as pd
import folium
import gradio as gr
import onnxruntime as ort


# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best(1).onnx")
IMG_SIZE = 640


# =========================
# LOAD ONNX MODEL DIRECTLY
# =========================
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = session.get_inputs()[0].name


# =========================
# PREPROCESSING
# =========================
def preprocess(image):
    if image is None:
        return None, None

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original = image.copy()

    h, w = image.shape[:2]

    scale = min(IMG_SIZE / w, IMG_SIZE / h)

    nw = int(round(w * scale))
    nh = int(round(h * scale))

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (IMG_SIZE, IMG_SIZE, 3),
        114,
        dtype=np.uint8
    )

    left = (IMG_SIZE - nw) // 2
    top = (IMG_SIZE - nh) // 2

    canvas[
        top:top + nh,
        left:left + nw
    ] = resized

    tensor = canvas.astype(np.float32) / 255.0

    tensor = np.transpose(
        tensor,
        (2, 0, 1)
    )

    tensor = np.expand_dims(
        tensor,
        axis=0
    )

    return tensor, (
        original,
        scale,
        left,
        top
    )


# =========================
# NMS
# =========================
def nms(boxes, scores, iou_threshold=0.45):

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:

        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])

        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        width = np.maximum(0, xx2 - xx1)
        height = np.maximum(0, yy2 - yy1)

        intersection = width * height

        union = (
            areas[i]
            + areas[order[1:]]
            - intersection
        )

        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0
        )

        remaining = np.where(
            iou <= iou_threshold
        )[0]

        order = order[
            remaining + 1
        ]

    return keep


# =========================
# DECODE YOLO OUTPUT
# =========================
def decode_predictions(
    output,
    confidence_threshold
):

    output = np.squeeze(output)

    # Expected:
    # [5, 8400]
    #
    # 0 = x
    # 1 = y
    # 2 = width
    # 3 = height
    # 4 = confidence

    if output.shape[0] == 5:
        output = output.T

    if output.shape[1] != 5:
        raise ValueError(
            f"Unexpected model output shape: {output.shape}"
        )

    boxes = []
    scores = []

    for prediction in output:

        cx, cy, w, h, score = prediction

        score = float(score)

        if score < confidence_threshold:
            continue

        x1 = float(cx - w / 2)
        y1 = float(cy - h / 2)

        x2 = float(cx + w / 2)
        y2 = float(cy + h / 2)

        boxes.append(
            [x1, y1, x2, y2]
        )

        scores.append(score)

    keep = nms(
        boxes,
        scores,
        iou_threshold=0.45
    )

    return [
        (
            boxes[i],
            scores[i]
        )
        for i in keep
    ]


# =========================
# MAIN DETECTION
# =========================
def detect_and_report(
    image,
    confidence
):

    if image is None:

        return (
            None,
            "No image uploaded",
            "—",
            "—",
            None,
            None
        )

    tensor, metadata = preprocess(image)

    try:

        outputs = session.run(
            None,
            {
                INPUT_NAME: tensor
            }
        )

        predictions = decode_predictions(
            outputs[0],
            confidence
        )

    except Exception as e:

        return (
            None,
            f"Inference error: {e}",
            "—",
            "—",
            None,
            None
        )

    original, scale, left, top = metadata

    annotated = original.copy()

    detections = []

    h, w = original.shape[:2]

    for i, (box, score) in enumerate(
        predictions,
        start=1
    ):

        x1, y1, x2, y2 = box

        # Undo letterbox
        x1 = (x1 - left) / scale
        y1 = (y1 - top) / scale

        x2 = (x2 - left) / scale
        y2 = (y2 - top) / scale

        # Keep inside image
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))

        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        # Draw detection
        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 180, 0),
            2
        )

        cv2.putText(
            annotated,
            f"Crab-Pot {score:.2f}",
            (
                int(x1),
                max(
                    20,
                    int(y1) - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2
        )

        detections.append(
            {
                "class": "Crab-Pot",
                "confidence": round(
                    float(score),
                    4
                ),
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            }
        )

    count = len(detections)

    if count > 0:

        maximum_confidence = max(
            d["confidence"]
            for d in detections
        )

        if maximum_confidence >= 0.50:
            risk = "HIGH"

        elif maximum_confidence >= 0.30:
            risk = "MEDIUM"

        else:
            risk = "LOW"

    else:

        maximum_confidence = 0
        risk = "NO DETECTION"


    # =========================
    # DEMO GPS
    # =========================

    latitude = 13.08
    longitude = 80.36

    report = {

        "system": "MarineVision",

        "analysis_type":
            "Side-Scan Sonar Marine Debris Detection",

        "gps_source":
            "SIMULATED OFFSHORE DEMO GPS",

        "latitude":
            latitude,

        "longitude":
            longitude,

        "detection_count":
            count,

        "maximum_confidence":
            round(
                maximum_confidence,
                4
            ),

        "risk_level":
            risk,

        "detections":
            detections
    }


    # =========================
    # JSON REPORT
    # =========================

    temp_dir = tempfile.gettempdir()

    json_path = os.path.join(
        temp_dir,
        "marine_debris_anomaly_report.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )


    # =========================
    # CSV REPORT
    # =========================

    csv_path = os.path.join(
        temp_dir,
        "marine_debris_anomaly_report.csv"
    )

    rows = []

    for i, detection in enumerate(
        detections,
        start=1
    ):

        rows.append(
            {
                "Anomaly_ID":
                    i,

                "Class":
                    detection["class"],

                "Confidence":
                    detection["confidence"],

                "Risk_Level":
                    risk,

                "Latitude":
                    latitude,

                "Longitude":
                    longitude,

                "GPS_Source":
                    "SIMULATED OFFSHORE DEMO GPS",

                "X1":
                    detection["bbox"][0],

                "Y1":
                    detection["bbox"][1],

                "X2":
                    detection["bbox"][2],

                "Y2":
                    detection["bbox"][3]
            }
        )

    if not rows:

        rows.append(
            {
                "Anomaly_ID": "",
                "Class": "",
                "Confidence": "",
                "Risk_Level": risk,
                "Latitude": latitude,
                "Longitude": longitude,
                "GPS_Source":
                    "SIMULATED OFFSHORE DEMO GPS",
                "X1": "",
                "Y1": "",
                "X2": "",
                "Y2": ""
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        csv_path,
        index=False
    )


    return (
        annotated,
        str(count),
        (
            f"{maximum_confidence:.2f}"
            if count
            else "—"
        ),
        risk,
        json_path,
        csv_path
    )


# =========================
# MAP
# =========================

map_path = os.path.join(
    BASE_DIR,
    "marine_debris_offshore_map.html"
)

np.random.seed(42)

marine_map = folium.Map(
    location=[
        13.08,
        80.36
    ],
    zoom_start=10
)

for i in range(100):

    lat = np.random.uniform(
        12.98,
        13.18
    )

    lon = np.random.uniform(
        80.30,
        80.42
    )

    folium.CircleMarker(
        location=[
            lat,
            lon
        ],
        radius=5,
        popup=
            f"Marine Anomaly #{i + 1}",
        fill=True
    ).add_to(
        marine_map
    )

marine_map.save(
    map_path
)


# =========================
# UI
# =========================

with gr.Blocks(
    title="MarineVision"
) as app:

    gr.Markdown(
        """
        # MarineVision

        **AI-Powered Side-Scan Sonar Marine Debris Detection**
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="numpy",
                label=
                    "Upload Side-Scan Sonar Image"
            )

            confidence = gr.Slider(
                minimum=0.05,
                maximum=0.80,
                value=0.25,
                step=0.05,
                label=
                    "Detection Confidence"
            )

            detect_btn = gr.Button(
                "🔎 Detect Marine Debris",
                variant="primary"
            )

        with gr.Column():

            output_image = gr.Image(
                label=
                    "Detection Result",
                type="numpy"
            )

            with gr.Row():

                detection_count = gr.Textbox(
                    label="Detections",
                    value="—",
                    interactive=False
                )

                max_conf = gr.Textbox(
                    label=
                        "Maximum Confidence",
                    value="—",
                    interactive=False
                )

                risk_level = gr.Textbox(
                    label="Risk Level",
                    value="—",
                    interactive=False
                )


    gr.Markdown(
        "## Anomaly Reports"
    )

    gr.Markdown(
        "Download structured reports after running detection."
    )

    with gr.Row():

        json_file = gr.File(
            label="JSON Report"
        )

        csv_file = gr.File(
            label="CSV Report"
        )


    gr.Markdown(
        "## Offshore Anomaly Map"
    )

    gr.HTML(
        f"""
        <iframe
            src="file={map_path}"
            width="100%"
            height="500"
            style="
                border:none;
                border-radius:16px;
            ">
        </iframe>
        """
    )


    detect_btn.click(
        fn=detect_and_report,

        inputs=[
            image_input,
            confidence
        ],

        outputs=[
            output_image,
            detection_count,
            max_conf,
            risk_level,
            json_file,
            csv_file
        ]
    )


# =========================
# RENDER START
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
