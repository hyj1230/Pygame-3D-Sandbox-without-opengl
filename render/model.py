import numpy as np
from PIL import Image
import pygame


font = pygame.font.Font('data/unifont.otf', 17)
font_nums = [font.render(str(i), True, (255,)*3).convert_alpha() for i in range(65)]


def normalize(v):
    # 归一化操作
    unit = np.linalg.norm(v)
    if unit == 0:
        return np.zeros(3, dtype=np.float64)
    return v / unit


class NoneModel:
    def __init__(self, texture, vertices, uv_vertices, norms, uv_indices, indices, w, h):
        self.vertices = vertices
        self.uv_vertices = uv_vertices
        self.norms = norms
        self.uv_indices = uv_indices
        self.indices = indices

        self.texture_array = texture
        self.texture_width, self.texture_height = w, h


class Cube:
    def __init__(self, filename, add, icon, is_drop=False):
        """
        https://en.wikipedia.org/wiki/Wavefront_.obj_file#Vertex_normal_indices
        """
        self.vertices = []
        self.uv_indices = []
        self.indices = []
        self.icon = pygame.image.load(icon).convert_alpha()
        self.icon_array = np.array(Image.open(icon))
        self.add = 12 * add
        self.div_num = 2 if not is_drop else 12
        self.max_num = 64

        with open(filename) as f:
            for line in f:
                line = line.replace('  ', ' ')
                if line.startswith("v "):
                    x, y, z = [float(d) for d in line.strip("v").strip().split(" ")][:3]
                    self.vertices.append(np.array([[x/self.div_num], [y/self.div_num], [z/self.div_num], [1]]))
                elif line.startswith("f "):
                    facet = [d.split("/") for d in line.strip("f").strip().split(" ")]
                    self.indices.append([int(d[0])-1 for d in facet])
                    self.uv_indices.append([int(d[1])-1+self.add for d in facet])
        self.vertices = np.array(self.vertices, dtype=np.float64)
        self.indices = np.array(self.indices, dtype=np.int32)
        self.uv_indices = np.array(self.uv_indices, dtype=np.int32)

        self.norms = np.empty((self.indices.shape[0], 3), dtype=np.float64)
        for i, face in enumerate(self.indices):
            v1, v2, v3 = self.vertices[:, :3, 0][face]
            self.norms[i] = normalize(np.cross(v2 - v1, v3 - v1))

        if not is_drop:
            self.drop = Cube(filename, add, icon, True)

    def render(self, screen, rect, num):
        screen.blit(self.icon, rect)
        if num == 1:
            return
        num_sf = font_nums[num]
        w, h = num_sf.get_size()
        num_x, num_y = rect.bottomright
        num_x += 1 - w
        num_y += 1 - h
        screen.blit(num_sf, (num_x, num_y))


class Texture:
    def __init__(self, filename, texture_filename, texture_filename_focused):
        self.uv_vertices = []

        texture = Image.open(texture_filename)
        texture_focused = Image.open(texture_filename_focused)
        self.texture_array = np.array(texture)
        self.texture_array_focused = np.array(texture_focused)
        self.texture_width, self.texture_height = texture.size
        '''tmp = self.texture_array.shape[2]
        self.texture_array.resize(
            (self.texture_width * self.texture_height, tmp))
        self.texture_array_focused.resize(
            (self.texture_width * self.texture_height, tmp))'''

        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.replace('  ', ' ')
                if line.startswith("vt "):
                    u, v = [float(d) for d in line.strip("vt").strip().split(" ")][:2]
                    self.uv_vertices.append([u, 1-v])
        self.uv_vertices = np.array(self.uv_vertices, dtype=np.float64)
