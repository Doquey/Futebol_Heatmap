import cv2
import numpy as np

def transform_players_det(boxes: np.array, homography_matrix: np.array) -> np.array:
    try:
        if boxes is None or boxes.shape[0] == 0 or boxes.shape[1] < 2:
            return np.array([])  # Return empty if boxes are invalid
        
        if homography_matrix is None or homography_matrix.shape != (3, 3):
            return np.array([])  # Return empty if homography matrix is invalid
        
        centroids = np.array(boxes[:, :2], dtype=np.float32).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(centroids, homography_matrix)
        return transformed_points
    
    except Exception as e:
        print(f"Error in transform_players_det: {e}")
        print(f"boxes: {boxes}")
        print(f"homography_matrix: {homography_matrix}")
        return np.array([])
    
    
def draw_bboxes(frame: np.ndarray, boxes: np.array, color=(255, 50, 50), thickness=2) -> np.ndarray:
    for box in boxes:
        if len(box) == 4:  
            x, y, x2, y2 = box
            frame = cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), color, thickness)
        else:
            print(f"Skipping invalid box with {len(box)} elements: {box}")
    return frame