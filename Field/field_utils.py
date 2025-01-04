import numpy as np
import cv2

from Field.field_config import SoccerPitchConfiguration
import cv2
import numpy as np

def draw_pitch(
    config,
    background_color: tuple = (34, 139, 34),  
    line_color: tuple = (255, 255, 255),  
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    
    background_color_bgr = background_color[::-1]
    line_color_bgr = line_color[::-1]

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color_bgr, dtype=np.uint8)

    for start, end in config.edges:
        point1 = (
            int(config.vertices[start - 1][0] * scale) + padding,
            int(config.vertices[start - 1][1] * scale) + padding
        )
        point2 = (
            int(config.vertices[end - 1][0] * scale) + padding,
            int(config.vertices[end - 1][1] * scale) + padding
        )
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color_bgr,
            thickness=line_thickness
        )
    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color_bgr,
        thickness=line_thickness
    )

    penalty_spots = [
        (
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        ),
        (
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        )
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color_bgr,
            thickness=-1
        )

    return pitch_image



if __name__ == "__main__":
    CONFIG = SoccerPitchConfiguration()
    annotated_frame = draw_pitch(CONFIG)
    
    cv2.imwrite("field.jpg", annotated_frame)