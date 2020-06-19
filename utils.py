
import numpy as np

class2color_dynamic = {
    0  : ( 0,    0,   0), # None
    1  : ( 70,  70,  70), # Buildings
    2  : (190, 153, 153), # Fences
    3  : ( 72,   0,  90), # Other
    4  : (220,  20,  60), # Pedestrians
    5  : (153, 153, 153), # Poles
    6  : (157, 234,  50), # RoadLines
    7  : (128,  64, 128), # Roads
    8  : (244,  35, 232), # Sidewalks
    9  : (107, 142,  35), # Vegetation
    10 : (  0,   0, 255), # Vehicles
    11 : (102, 102, 156), # Walls
    12 : (220, 220,   0)  # TrafficSigns
}

def semseg_image_to_carla_palette(labels_array, static=False):
    map_dict = class2color_dynamic
    result = np.zeros((labels_array.shape[0], labels_array.shape[1], 3))
    for key, value in map_dict.items():
        result[np.where(labels_array == key)] = value
    return result.astype(np.uint8)
