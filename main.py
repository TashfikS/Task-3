from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
import zlib

app = Flask(__name__)

net = cv2.dnn.readNet("yolov4-custom_last.weights", "yolov4-custom.cfg")
with open("obj.names", "r") as f:
    classes = f.read().strip().split('\n')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        image = request.files['file'].read()
        img_array = cv2.imdecode(np.frombuffer(image, np.uint8), -1)

        boxes = detect_objects_yolo(img_array)

        annotated_image = annotate_image(img_array, boxes)

        _, img_encoded = cv2.imencode('.png', annotated_image)
        img_encoded_compressed = zlib.compress(img_encoded.tobytes())

        def generate():
            chunk_size = 1024
            for i in range(0, len(img_encoded_compressed), chunk_size):
                yield img_encoded_compressed[i:i + chunk_size]

        headers = {
            'Content-Type': 'application/octet-stream',
            'Content-Disposition': 'inline; filename=image.png'
        }

        return Response(generate(), content_type='application/octet-stream', headers=headers), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def detect_objects_yolo(image):
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    output_layer_names = net.getUnconnectedOutLayersNames()

    detections = net.forward(output_layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes

def annotate_image(image, boxes):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

if __name__ == '__main__':
    app.run(debug=True)
