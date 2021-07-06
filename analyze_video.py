import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import sys
import random
import colorsys


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def draw_bbox(image, bboxes, game_state, classes=read_class_names("classes.names"), allowed_classes=list(read_class_names("classes.names").values()), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        coor_0 = int(coor[0])
        coor_1 = int(coor[1])
        coor_2 = int(coor[2])
        coor_3 = int(coor[3])
        

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        class_name = classes[class_ind]


        # check if class is in allowed classes
        if class_name not in allowed_classes:
            continue
        else:

            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))

            # update game state 
            game_state["goal_timeout"] -= 1

            if class_name == "basketball":  
                # check shot status
                ball_min = c2[1]
                if ball_min < game_state["hoop_height"]:
                    game_state["is_shot"] = True
                    print("IS SHOT")
                else:
                    if game_state["is_shot"]:
                        game_state["attempts"] += 1
                        game_state["is_shot"] = False
                print(class_name)
                print(f"c1: {c1}\nc2: {c2}\n")

            if class_name == "goal":
                if game_state["goal_timeout"] <= 0:
                    game_state["makes"] += 1
                    game_state["goal_timeout"] = 10
                else:
                    pass
                

            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if show_label:
                bbox_mess = '%s: %.2f' % (classes[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                # cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled
                cv2.rectangle(image, c1, (int(np.float32(c3[0])), int(np.float32(c3[1]))), bbox_color, -1)

                cv2.putText(image, bbox_mess, (c1[0], int(np.float32(c1[1] - 2))), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image

frame_batch = 3
input_size = 416
iou = 0.45
score = 0.60
weights_path = "checkpoints/custom-416"
saved_model = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])
infer = saved_model.signatures["serving_default"]

if len(sys.argv) < 2:
    print("Missing video path, exiting early.")
    exit()

video_path = sys.argv[1]

try:
    video = cv2.VideoCapture(video_path)
except:
    print("Could not load video from path: ", video_path)
    exit()


game_state = {
    "goal_timeout": 0,
    "attempts": 0,
    "makes": 0,
    "hoop_height": 380,
    "is_shot": False
}

skip_count = 0

while True:
    ret, frame = video.read()
    if ret:
        skip_count += 1
        if skip_count < frame_batch:
            continue
    else:
        print('Video has ended or failed, try a different video format!')
        break

    skip_count = 0
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    
    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = draw_bbox(frame, pred_bbox, game_state)
    result = np.asarray(image)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result" ,result)

    if cv2.waitKey(1) == ord('q'):
            break

video.release()
cv2.destroyAllWindows()

makes = game_state["makes"]
attempts = game_state["attempts"]

print(f"makes: {makes}\nattempts: {attempts}")