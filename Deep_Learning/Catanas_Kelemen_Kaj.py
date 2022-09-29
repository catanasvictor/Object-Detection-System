import cv2
import libcamera
import numpy as np
import tflite_runtime.interpreter as tflite

from picamera2 import Picamera2, Preview, MappedArray

# Fisierele de model si label
MODEL_FILE = "./mobilenet_v2.tflite"
LABEL_FILE = "./coco_labels.txt"

# Mod de previzualizare
PREVIEW_TYPE = Preview.QT     # SSH cu X11 FORWARDING
# PREVIEW_TYPE = Preview.QTGL # display
# PREVIEW_TYPE = Preview.DRM  # Fara X Windows

NORMAL_SIZE = (640, 480) # Dimensiunea imaginii la afisare
HALF_SIZE = (320, 240)   # Dimensiunea imaginii la procesare 

# Categorii de dimensiuni pentru previzualizare
CURR_PREVIEW_HALF = "lores"   # HALF_SZIE
CURR_PREVIEW_NORMAL = "main"  # NORMAL_SIZE

# Setari bounding box
BOUND_BOXES = []
BOUNDING_BOXES_COLOR = (255, 0, 0, 0)
BOUNDING_BOXES_WIDTH = 3

# LABEL
LABEL_COLOR = (255, 255, 255)
LABEL_FONT = cv2.FONT_HERSHEY_DUPLEX


def initialize_camera():
    cam = Picamera2()
    cam.start_preview(PREVIEW_TYPE)
    config = cam.preview_configuration(main={"size": NORMAL_SIZE},
                                       lores={"size": HALF_SIZE, "format": "YUV420"},
                                       raw={"size": cam.sensor_resolution},
                                       transform=libcamera.Transform(hflip=0, vflip=1))
    cam.configure(config)
    cam.post_callback = put_overlay
    return cam


def parse_label_file(file_path):
    ret_vals = {}
    text_file = open(file_path, 'r')
    lines = text_file.readlines()
    for line in lines:
        split_vals = line.strip().split(maxsplit=1)
        ret_vals[int(split_vals[0])] = split_vals[1].strip()
    return ret_vals


def process_image(image):
    global BOUND_BOXES

    labels = parse_label_file(LABEL_FILE)

    interpreter = tflite.Interpreter(model_path=MODEL_FILE, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = False

    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    orig_height, orig_width, channels = rgb.shape

    resized_image = cv2.resize(rgb, (width, height))

    input_data = np.expand_dims(resized_image, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    detected_boxes = interpreter.get_tensor(output_details[0]['index'])
    detected_classes = interpreter.get_tensor(output_details[1]['index'])
    detected_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])

    BOUND_BOXES = []
    for i in range(int(num_boxes)):
        top, left, bottom, right = detected_boxes[0][i]
        class_id = int(detected_classes[0][i])
        score = detected_scores[0][i]
        if score > 0.5:
            xmin = left * orig_width
            ymin = bottom * orig_height
            xmax = right * orig_width
            ymax = top * orig_height
            box = [xmin, ymin, xmax, ymax, labels[class_id] + ': ' + str(round(score * 100, 2)) + '%']
            BOUND_BOXES.append(box)


def calc_bound_box_coords(box):
    padding_val = 5
    bottom_left = (int(box[0] * 2) - padding_val, int(box[1] * 2) - padding_val)
    top_right = (int(box[2] * 2) + padding_val, int(box[3] * 2) + padding_val)
    return [bottom_left, top_right]


def put_overlay(request):
    with MappedArray(request, CURR_PREVIEW_NORMAL) as m:
        for b_box in BOUND_BOXES:
            print(b_box)
            [b_box_bottom_left, b_box_top_right] = calc_bound_box_coords(b_box)
            cv2.rectangle(m.array, b_box_bottom_left, b_box_top_right, BOUNDING_BOXES_COLOR, BOUNDING_BOXES_WIDTH)
            cv2.putText(m.array, b_box[4], (int(b_box[0] * 2) + 10, int(b_box[1] * 2) + 10), LABEL_FONT, 1, LABEL_COLOR,2, cv2.LINE_AA)


def main():
    cam = initialize_camera()
    stride_for_resize = cam.stream_configuration(CURR_PREVIEW_HALF)["stride"]
    cam.start()

    while True:
        buffer = cam.capture_buffer(CURR_PREVIEW_HALF)
        resized_buf = buffer[:stride_for_resize * HALF_SIZE[1]].reshape((HALF_SIZE[1], stride_for_resize))
        process_image(resized_buf)


if __name__ == '__main__':
    main()

