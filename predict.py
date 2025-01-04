import cv2
from ultralytics import YOLO
import numpy as np
from Field.field_config import SoccerPitchConfiguration
from Field.field_utils import draw_pitch
from utilities import transform_players_det, draw_bboxes

# Load models
keypoints_det = YOLO("./models/keypoints_detector.pt")
player_det = YOLO("./models/player_detector.pt")

# Video capture
cap = cv2.VideoCapture("./videoplayback_cut.mp4")

# Configurations
config = SoccerPitchConfiguration()
static_field_view = draw_pitch(config)

# Parameters
scale = 0.1
padding = 50
heatmap_width = int(static_field_view.shape[1])
heatmap_height = int(static_field_view.shape[0])
heatmap = np.zeros((heatmap_height, heatmap_width))


def get_heatmap_color(value, max_value):
    """Generate a color for the heatmap based on normalized value."""
    normalized = value / max_value
    if normalized <= 0.25:
        r, g, b = 0, int(4 * normalized * 255), 255
    elif normalized <= 0.5:
        r, g, b = 0, 255, int((1 - 4 * (normalized - 0.25)) * 255)
    elif normalized <= 0.75:
        r, g, b = int(4 * (normalized - 0.5) * 255), 255, 0
    else:
        r, g, b = 255, int((1 - 4 * (normalized - 0.75)) * 255), 0
    return b, g, r


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints_results = keypoints_det(frame)
    keypoints = keypoints_results[0].keypoints.xy.cpu().numpy()[0]
    keypoints_confidence = keypoints_results[0].keypoints.conf.cpu().numpy()[0]
    filter = np.where(keypoints_confidence > 0.70)
    valid_keypoints = keypoints[filter]
    mask = (valid_keypoints[:, 0] > 1) & (valid_keypoints[:, 1] > 1)

    if len(valid_keypoints) > 3:
        filtered_static_field_keypoints = np.array(config.vertices)[filter]
        homography_m, _ = cv2.findHomography(valid_keypoints[mask], filtered_static_field_keypoints[mask])

    player_results = player_det.predict(frame, verbose=False)
    player_boxes = player_results[0].boxes.xywh.cpu().numpy()
    player_classes = player_results[0].boxes.cls.cpu().numpy()
    player_boxes = player_boxes[player_classes == 2]

    player_boxes_xyxy = np.zeros_like(player_boxes)
    player_boxes_xyxy[:, 0] = player_boxes[:, 0] - player_boxes[:, 2] / 2
    player_boxes_xyxy[:, 1] = player_boxes[:, 1] - player_boxes[:, 3] / 2
    player_boxes_xyxy[:, 2] = player_boxes[:, 0] + player_boxes[:, 2] / 2
    player_boxes_xyxy[:, 3] = player_boxes[:, 1] + player_boxes[:, 3] / 2

    frame = draw_bboxes(frame, player_boxes_xyxy)

    transformed_points = transform_players_det(player_boxes, homography_m)
    if transformed_points is not None and len(transformed_points) > 0:
        for point in transformed_points.reshape(-1, 2):
            x = int(point[0] * scale) + padding
            y = int(point[1] * scale) + padding
            if 0 <= x < heatmap_width and 0 <= y < heatmap_height:
                heatmap[y:y + 20, x:x + 20] += 1

    max_count = np.max(heatmap)
    for y in range(heatmap.shape[0]):
        for x in range(heatmap.shape[1]):
            if heatmap[y, x] > 0:
                color = get_heatmap_color(heatmap[y, x], max_count)
                cv2.circle(static_field_view, (x, y), 15, color, -1)

    field_resized = cv2.resize(static_field_view, (frame.shape[1] // 4, frame.shape[0] // 4))
    x_offset = (frame.shape[1] - field_resized.shape[1]) // 2
    y_offset = frame.shape[0] - field_resized.shape[0] - 10
    frame[y_offset:y_offset + field_resized.shape[0], x_offset:x_offset + field_resized.shape[1]] = field_resized

    output_width = 1280  
    output_height = 720 
    frame_resized = cv2.resize(frame, (output_width, output_height))

    cv2.imshow("show", frame_resized)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
