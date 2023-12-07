import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def calculate_voronoi_areas(vor):
    areas = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            areas.append(Polygon(polygon).area)
    return areas

def calculate_area_variance(areas):
    return np.var(areas)


