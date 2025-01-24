import numpy as np

# 각도 계산 함수
def calculate_angle(point_a, point_b, point_c):
    vector_ab = np.array(point_a) - np.array(point_b)
    vector_cb = np.array(point_c) - np.array(point_b)
    dot_product = np.dot(vector_ab, vector_cb)
    magnitude_ab = np.linalg.norm(vector_ab)
    magnitude_cb = np.linalg.norm(vector_cb)
    if magnitude_ab == 0 or magnitude_cb == 0:
        return None
    cosine_angle = dot_product / (magnitude_ab * magnitude_cb)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle