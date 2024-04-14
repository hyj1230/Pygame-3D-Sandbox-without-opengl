import numpy as np
import pygame
import speedup
import math


def normalize(v):
    # 归一化操作
    unit = np.linalg.norm(v)
    if unit == 0:
        return np.zeros(3, dtype=np.float64)
    return v / unit


def gluPerspective(fov, aspect, near, far):
    ymax = near * math.tan(fov * math.pi / 360)
    ymin = -ymax
    xmin = ymin * aspect
    xmax = -xmin
    return np.array(
        [
            [(2*near)/(xmax-xmin), 0, (xmax+xmin)/(xmax-xmin), 0],
            [0,  (2*near)/(ymax-ymin), (ymax+ymin)/(ymax-ymin), 0],
            [0,  0, -((far+near)/(far-near)), -((2*far*near)/(far-near))],
            [0,  0, -1, 0],
        ], dtype=np.float64
    )


def viewport(x, y, w, h):
    return np.array(
        [
            [w/2, 0, 0, x+w/2],
            [0, h/2, 0, y+h/2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64
    )


def translation(tx, ty, tz):
    return np.array(
        [
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1],
        ], dtype=np.float64
    )


light_dirs = [normalize(np.array([1, 1, 1], dtype=np.float64)),
              normalize(np.array([-1, -1, -1], dtype=np.float64))]


def render_cube(model, height, width, screen, zbuffer, angle_x, angle_y, tx, ty, tz, fov, O2=True):
    # 绘制 3d 模型
    # 初始化矩阵
    transform_matrix = np.dot(
        np.dot(angle_x, angle_y),
        translation(tx, ty, tz)
    )
    Projection = gluPerspective(fov, width / height, 0.1, 60.0)  # 透视矩阵
    Viewport = viewport(0, 0, width, height)

    # 矩阵预乘
    Projection_ModelView = np.dot(Projection, transform_matrix)  # 因为矩阵乘法满足结合律，可以进行预乘，以减少运算量
    Viewport_Projection_ModelView = np.dot(Viewport, Projection_ModelView)

    clip_vert = np.matmul(Projection_ModelView, model.vertices)[:, :, 0]  # 存储模型相机坐标系坐标，且进行透视
    pts = np.matmul(Viewport_Projection_ModelView, model.vertices)[:, :, 0]  # 存储屏幕坐标
    norms = np.zeros(model.norms.shape[0], dtype=np.float64)

    for light_dir in light_dirs:
        norms += np.maximum(np.dot(model.norms, light_dir), 0.0)

    speedup.generate_faces_flat(
        model.indices, model.uv_indices, pts,
        model.uv_vertices, clip_vert, norms,
        model.texture_width, model.texture_height, pygame.surfarray.pixels3d(screen),
        model.texture_array, zbuffer, O2
    )  # 逐个绘制三角形
