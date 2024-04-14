import time
from collections import deque
from noise_gen import NoiseGen
import numpy as np
import random
from render import render_cube, NoneModel
import pygame
from numba import njit
from configure import *


@njit
def normalize(position):  # 将 position 坐标取整
    x, y, z = position
    x, y, z = round(x), round(y), round(z)
    return x, y, z


@njit(fastmath=True)
def get_sight_vector(x, y):
    """返回当前视线向量，指示玩家的朝向。"""
    # y 范围从 -90 到 90 ，或从 -pi/2 到 pi/2 ，因此 m 范围从 0 到 1 ，
    # 当平行于地面向前看时， m 为 1 ，正对正上或正下时为 0 。
    rad_y = math.radians(y)
    rad_x = math.radians(x - 90)
    m = math.cos(rad_y)
    # dy范围从-1到1，在正对下方时为-1，在正对上方时为1。
    dy = math.sin(rad_y)
    dx = math.cos(rad_x) * m
    dz = math.sin(rad_x) * m
    return dx, dy, dz


def normalize_vec(v):
    # 归一化操作
    unit = np.linalg.norm(v)
    if unit == 0:
        return np.zeros(3, dtype=np.float64)
    return v / unit


def glRotatef(angle, x, y, z):
    x, y, z = normalize_vec(np.array([x, y, z], dtype=np.float64))
    alpha = angle / 180 * np.pi
    s = np.sin(alpha)
    c = np.cos(alpha)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y + s * z, t * x * z - s * y, 0],
            [t * x * y - s * z, t * y * y + c, t * y * z + s * x, 0],
            [t * x * z + s * y, t * y * z - s * x, t * z * z + c, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64
    ).T


def sectorize(position):   # 返回位置 position 所在区块的坐标。
    x, y, z = normalize(position)  # 将位置取整
    x, y, z = x // SECTOR_SIZE, y // SECTOR_SIZE, z // SECTOR_SIZE  # 根据SECTOR_SIZE将位置转换为区块坐标
    return x, 0, z


class Model(object):

    def __init__(self):
        self.size = 128
        self.world = {}  # 存储某个位置的方块对应的纹理

        self.shown = {}  # 和`world`相同，但只包含显示的方块。

        # 存储某个位置的方块的顶点数据、边数据、纹理数据
        self._shown_vertices = {}
        self._shown_indices = {}
        self._shown_uv_indices = {}

        # 存储某个区块中的所有方块的纹理
        self.sectors = {}

        # 简单的函数队列实现。队列由 _show_block() 和 _hide_block() 方法调用填充。
        self.queue = deque()

        self._initialize()  # 生成世界
        self.model = NoneModel(all_textures.texture_array, None, all_textures.uv_vertices, None, None, None,
                               all_textures.texture_width, all_textures.texture_height)

        self.world_drops = {}
        self.sectors_drops = {}
        self._shown_drops_vertices = {}
        self._shown_drops_indices = {}
        self._shown_drops_uv_indices = {}
        self.model_drop = NoneModel(all_textures.texture_array, None, all_textures.uv_vertices, None, None, None,
                                    all_textures.texture_width, all_textures.texture_height)

    def _initialize(self):  # 生成世界
        print("开始生成世界")
        gen = NoiseGen(452692)
        random.seed(452692)

        range_n = np.arange(self.size)

        print('初始化地形高度')
        height_map = np.zeros((self.size, self.size))
        for x in range_n:
            for z in range_n:
                height_map[x, z] = int(gen.getHeight(x, z))
        print('地形高度初始化完成')

        # 生成世界
        for x in range_n:
            for z in range_n:
                h = height_map[x, z]
                self.add_block((x, 0, z), BEDROCK, immediate=False)
                if h < 15:
                    for y in np.arange(1, max(int(h / 2), 2)):
                        self.add_block((x, y, z), STONE, immediate=False)
                    for y in np.arange(max(int(h / 2), 2), 15):
                        self.add_block((x, y, z), WATER, immediate=False)
                    continue
                self.add_block((x, h, z), GRASS, immediate=False)
                for y in np.arange(h - 1, 0, -1):
                    self.add_block((x, y, z), STONE, immediate=False)
                # 可能在这个位置(x, z)添加树
                if h > 20:
                    if random.randrange(0, 1000) > 990:
                        tree_height = random.randrange(5, 7)
                        # 树干
                        for y in np.arange(h + 1, h + tree_height):
                            self.add_block((x, y, z), WOOD, immediate=False)
                        # 树叶
                        leafh = h + tree_height
                        for lz in np.arange(z + -2, z + 3):
                            for lx in np.arange(x + -2, x + 3):
                                for ly in np.arange(3):
                                    self.add_block((lx, leafh + ly, lz), LEAF, immediate=False)
        print("世界生成完成")

    def add_drop(self, texture, position):
        self.world_drops[position] = texture.drop
        self.sectors_drops.setdefault(sectorize(position), []).append(position)
        self._show_drop(position, texture.drop)

    def remove_drop(self, position):
        del self.world_drops[position]
        self.sectors_drops[sectorize(position)].remove(position)
        self._hide_drop(position)

    def _hide_drop(self, position):
        self._shown_drops_vertices.pop(position)
        self._shown_drops_indices.pop(position)
        self._shown_drops_uv_indices.pop(position)

    def _show_drop(self, position, texture):
        x, y, z = position
        new_vertices = texture.vertices.copy()
        new_vertices[:, 0, 0] += x
        new_vertices[:, 1, 0] += y
        new_vertices[:, 2, 0] += z
        self._shown_drops_vertices[position] = new_vertices
        self._shown_drops_indices[position] = texture.indices.copy()
        self._shown_drops_uv_indices[position] = texture.uv_indices.copy()

    def hit_test(self, position, vector, max_distance=7):
        """从当前位置进行视线搜索。如果遇到方块，则返回该方块以及视线上的前一个方块。
        如果没有找到方块，则返回None，None。

        参数
        ----------
        position : 长度为3的元组
            要检查可见性的位置(x, y, z)。
        vector : 长度为3的元组
            视线向量。
        max_distance : 整数
            搜索碰撞的最大距离（单位：方块）。

        """
        m = 8
        x, y, z = position
        dx, dy, dz = vector
        previous = None
        for _ in range(max_distance * m):
            key = normalize((x, y, z))
            if key != previous and key in self.world:
                return key, previous
            previous = key
            x, y, z = x + dx / m, y + dy / m, z + dz / m
        return None, None

    def exposed(self, position):
        """如果给定的`position`在所有6个面都被方块包围，则返回False，否则返回True。

        """
        x, y, z = position
        for dx, dy, dz in FACES:
            if (x + dx, y + dy, z + dz) not in self.world:
                return True
        return False

    def add_block(self, position, texture, immediate=True):
        """将带有给定纹理和位置的方块添加到世界中。

        参数
        ----------
        position：长度为3的元组
            要添加的方块的(x, y, z)位置。
        texture：长度为3的列表
            纹理方块的坐标。使用`tex_coords()`生成。
        immediate：bool
            是否立即绘制方块。

        """
        if position in self.world:
            self.remove_block(position, immediate)
        self.world[position] = texture
        self.sectors.setdefault(sectorize(position), []).append(position)
        if immediate:
            if self.exposed(position):
                self.show_block(position)
            self.check_neighbors(position)

    def remove_block(self, position, immediate=True):
        """删除给定位置处的方块。

        参数
        ----------
        position：长度为3的元组
            要删除的方块的(x, y, z)位置。
        immediate：bool
            是否立即从画布中移除方块。

        """
        del self.world[position]
        self.sectors[sectorize(position)].remove(position)
        if immediate:
            if position in self.shown:
                self.hide_block(position)
            self.check_neighbors(position)

    def check_neighbors(self, position):
        """检查“position”周围的所有方块，并确保它们的可视状态是准确的。
        这意味着隐藏未显示的方块，并显示所有暴露的方块。通常在添加或删除方块后使用。

        """
        x, y, z = position
        for dx, dy, dz in FACES:
            key = (x + dx, y + dy, z + dz)
            if key not in self.world:
                continue
            if self.exposed(key):
                if key not in self.shown:
                    self.show_block(key)
            else:
                if key in self.shown:
                    self.hide_block(key)

    def show_block(self, position, immediate=True):
        """显示给定位置处的方块。这个方法假设方块已经使用add_block()方法添加了。

        参数
        ----------
        position：长度为3的元组
            要显示的方块的(x, y, z)位置。
        immediate：bool
            是否立即显示方块。

        """
        texture = self.world[position]
        self.shown[position] = texture
        if immediate:
            self._show_block(position, texture)
        else:
            self._enqueue(self._show_block, position, texture)

    def _show_block(self, position, texture):
        """show_block()方法的私有实现。

        参数
        ----------
        position：长度为3的元组
            要显示的方块的(x, y, z)位置。
        texture：长度为3的列表
            纹理方块的坐标。使用`tex_coords()`生成。

        """
        x, y, z = position
        new_vertices = texture.vertices.copy()
        new_vertices[:, 0, 0] += x
        new_vertices[:, 1, 0] += y
        new_vertices[:, 2, 0] += z
        self._shown_vertices[position] = new_vertices
        self._shown_indices[position] = texture.indices.copy()
        self._shown_uv_indices[position] = texture.uv_indices.copy()

    def hide_block(self, position, immediate=True):
        """隐藏给定位置处的方块。隐藏不会从世界中移除方块。

        参数
        ----------
        position：长度为3的元组
            要隐藏的方块的(x, y, z)位置。
        immediate：bool
            是否立即从画布中移除方块。

        """
        self.shown.pop(position)
        if immediate:
            self._hide_block(position)
        else:
            self._enqueue(self._hide_block, position)

    def _hide_block(self, position):
        """hide_block()方法的私有实现。

        """
        self._shown_vertices.pop(position)
        self._shown_indices.pop(position)
        self._shown_uv_indices.pop(position)

    def show_sector(self, sector):
        """确保需要显示的给定区块中的所有方块都绘制到画布上。

        """
        for position in self.sectors.get(sector, []):
            if position not in self.shown and self.exposed(position):
                self.show_block(position, False)
        for position in self.sectors_drops.get(sector, []):
            if position not in self._shown_drops_indices:
                self._show_drop(position, self.world_drops[position])

    def hide_sector(self, sector):
        """确保需要隐藏的给定区块中的所有方块都从画布中移除。

        """
        for position in self.sectors.get(sector, []):
            if position in self.shown:
                self.hide_block(position, False)
        for position in self.sectors_drops.get(sector, []):
            if position in self._shown_drops_indices:
                self._hide_drop(position)

    def change_sectors(self, before, after):
        """从“before”区块移动到“after”区块。区块是世界的连续x、y子区域。区块用于加速世界渲染。

        """
        before_set = set()
        after_set = set()
        pad = 4
        for dx in range(-pad, pad + 1):
            dy = 0
            for dz in range(-pad, pad + 1):
                if dx ** 2 + dy ** 2 + dz ** 2 > (pad + 1) ** 2:
                    continue
                if before:
                    x, y, z = before
                    before_set.add((x + dx, y + dy, z + dz))
                if after:
                    x, y, z = after
                    after_set.add((x + dx, y + dy, z + dz))
        show = after_set - before_set
        hide = before_set - after_set
        for sector in show:
            self.show_sector(sector)
        for sector in hide:
            self.hide_sector(sector)

    def _enqueue(self, func, *args):
        """将“func”添加到内部队列中。

        """
        self.queue.append((func, args))

    def _dequeue(self):
        """从内部队列中弹出顶部函数并调用它。

        """
        func, args = self.queue.popleft()
        func(*args)

    def process_queue(self):
        """在定期休息的同时处理整个队列。这样可以使游戏循环顺利运行。
        队列包含对_show_block()和_hide_block()的调用，所以如果add_block()或remove_block()使用immediate=False调用，应该调用该方法。

        """
        start = time.process_time()
        while self.queue and time.process_time() - start < 1.0 / TICKS_PER_SEC:
            self._dequeue()

    def process_entire_queue(self):
        """处理整个队列，没有休息。

        """
        while self.queue:
            self._dequeue()

    def render(self):
        cube_num = len(self._shown_vertices)
        if cube_num == 0:
            return True
        vertices = np.concatenate(list(self._shown_vertices.values()), axis=0)
        indices = np.concatenate(list(self._shown_indices.values()), axis=0)
        uv_indices = np.concatenate(list(self._shown_uv_indices.values()), axis=0)
        indices += np.repeat(np.arange(0, cube_num * 8, 8), repeats=12)[:, None]
        self.model.vertices = vertices
        self.model.indices = indices
        self.model.uv_indices = uv_indices
        self.model.norms = DIRT.norms
        return False

    def render_drop(self):
        cube_num = len(self._shown_drops_vertices)
        if cube_num == 0:
            return True
        vertices = np.concatenate(list(self._shown_drops_vertices.values()), axis=0)
        vertices_add = np.concatenate(list(self._shown_drops_vertices.keys()), axis=0)
        addx, addz = vertices_add[np.arange(0, cube_num*3, 3)], vertices_add[np.arange(2, cube_num*3, 3)]
        addx = np.repeat(addx, 8)
        addz = np.repeat(addz, 8)
        indices = np.concatenate(list(self._shown_drops_indices.values()), axis=0)
        uv_indices = np.concatenate(list(self._shown_drops_uv_indices.values()), axis=0)
        indices += np.repeat(np.arange(0, cube_num * 8, 8), repeats=12)[:, None]
        vertices[:, 1, 0] += 0.05 * math.sin(time.time())
        vertices[:, 0, 0] -= addx
        vertices[:, 2, 0] -= addz
        angle = time.time()
        vertices[:, 0, 0], vertices[:, 2, 0] = vertices[:, 2, 0] * np.sin(angle) + vertices[:, 0, 0] * np.cos(angle), \
                                               vertices[:, 2, 0] * np.cos(angle) - vertices[:, 0, 0] * np.sin(angle)
        vertices[:, 0, 0] += addx
        vertices[:, 2, 0] += addz
        self.model_drop.vertices = vertices
        self.model_drop.indices = indices
        self.model_drop.uv_indices = uv_indices
        self.model_drop.norms = DIRT.norms
        return False


class Inventory:
    def __init__(self, screen):
        self.screen = screen
        self.w, self.h = screen.get_size()

        self.bg = pygame.transform.scale(pygame.image.load('data/物品栏.png'), (182 * 2, 22 * 2)).convert_alpha()
        self.pic_w, self.pic_h = self.bg.get_size()
        self.bg_rect = pygame.Rect((self.w-self.pic_w)//2, self.h-self.pic_h, self.pic_w, self.pic_h)

        self.set_inventory = pygame.transform.scale(pygame.image.load('data/被选中.png'), (22 * 2 + 4, 22 * 2 + 4)).convert_alpha()
        self.set_w, self.set_h = self.set_inventory.get_size()

        self.inventory = [SAND, WOOD, LEAF, DIRT, STONE, ROCK, WOODPLANK, CRAFTINGTABLE, None]  # 玩家可以放置的方块列表。
        self.inventory_index = 0  # 用户可以放置的当前方块。
        self.inventory_num = [64, 64, 64, 64, 64, 64, 64, 5, 0]  # 每个格子的方块数量

        self.mode = 0  # 是否无限放置
        self.font = pygame.font.Font('data/unifont.otf', 17)
        self.font_nums = [self.font.render(str(i), True, (255,) * 3).convert_alpha() for i in range(65)]

    def update_size(self):
        self.w, self.h = self.screen.get_size()
        self.bg_rect = pygame.Rect((self.w - self.pic_w) // 2, self.h - self.pic_h, self.pic_w, self.pic_h)

    def set_index(self, index):
        self.inventory_index = index

    def get_block(self):
        res = self.inventory[self.inventory_index]
        if not res or self.mode:  # 没有物品或是创造模式
            return res
        self.inventory_num[self.inventory_index] -= 1  # 物品数量减一
        if self.inventory_num[self.inventory_index] == 0:  # 物品取完
            self.inventory[self.inventory_index] = None
        return res

    def get_rect(self, index):
        return pygame.Rect(self.bg_rect.x + (self.set_w - 8) * index - 2, self.bg_rect.y-2, self.set_w, self.set_h)

    def render(self):
        self.screen.blit(self.bg, self.bg_rect)
        for i, texture in enumerate(self.inventory):
            if texture is None:
                continue
            texture.render(self.screen, texture.icon.get_rect(center=self.get_rect(i).center), self.inventory_num[i])
        self.screen.blit(self.set_inventory, self.get_rect(self.inventory_index))


class BackPack:
    def __init__(self, screen, inventory):
        self.screen = screen
        self.inventory = inventory
        self.bg = pygame.transform.scale(pygame.image.load('data/背包.png'), (704//2, 664//2)).convert_alpha()
        self.bg_black = pygame.Surface((1, 1), pygame.SRCALPHA)
        self.bg_black.fill((0, 0, 0, 150))
        self.bg_blacks = {}
        self.bg_rect = self.bg.get_rect(center=self.screen.get_rect().center)
        x, y = self.bg_rect.bottomleft

        self.backpack = [[None for _ in range(9)], [None for _ in range(9)],
                         [None for _ in range(9)], self.inventory.inventory,
                         [None for _ in range(5)]]
        self.backpack_num = [[0 for _ in range(9)], [0 for _ in range(9)],
                             [0 for _ in range(9)], self.inventory.inventory_num,
                             [0 for _ in range(5)]]
        self.backpack_rect = [[pygame.Rect(x + 16 + i * 36, y - 116 - 48, 32, 32) for i in range(9)],
                              [pygame.Rect(x + 16 + i * 36, y - 80 - 48, 32, 32) for i in range(9)],
                              [pygame.Rect(x + 16 + i * 36, y - 44 - 48, 32, 32) for i in range(9)],
                              [pygame.Rect(x + 16 + i * 36, y - 48, 32, 32) for i in range(9)],
                              [pygame.Rect(300 + j, 186 + i, 32, 32) for i in (0, 36) for j in (0, 36)] +
                              [pygame.Rect(412, 206, 32, 32)]]

        self.focus_texture = None
        self.focus_rect = None
        self.focus_num = None
        self.move = False

    def update_size(self):
        self.bg_rect = self.bg.get_rect(center=self.screen.get_rect().center)
        x, y = self.bg_rect.bottomleft
        self.backpack_rect = [[pygame.Rect(x + 16 + i * 36, y - 116 - 48, 32, 32) for i in range(9)],
                              [pygame.Rect(x + 16 + i * 36, y - 80 - 48, 32, 32) for i in range(9)],
                              [pygame.Rect(x + 16 + i * 36, y - 44 - 48, 32, 32) for i in range(9)],
                              [pygame.Rect(x + 16 + i * 36, y - 48, 32, 32) for i in range(9)],
                              [pygame.Rect(300 + j, 186 + i, 32, 32) for i in (0, 36) for j in (0, 36)] +
                              [pygame.Rect(412, 206, 32, 32)]]

    def render(self):
        if self.screen.get_size() not in self.bg_blacks:
            self.bg_blacks[self.screen.get_size()] = pygame.transform.scale(self.bg_black, self.screen.get_size()).convert_alpha()
        self.screen.blit(self.bg_blacks[self.screen.get_size()], (0, 0))
        self.screen.blit(self.bg, self.bg_rect)
        pos = self.get_mouse_index(pygame.mouse.get_pos())
        for i in range(5):
            for j in range(len(self.backpack_num[i])):
                if i == pos[0] and j == pos[1]:
                    pygame.draw.rect(self.screen, (208, 205, 209), self.backpack_rect[i][j])
                if self.backpack[i][j] is None or not self.backpack_num[i][j]:
                    continue
                self.backpack[i][j].render(self.screen, self.backpack_rect[i][j], self.backpack_num[i][j])
        if self.move:
            self.focus_texture.render(self.screen, self.focus_rect, self.focus_num)

    def get_mouse_index(self, mouse_pos):  # 获取鼠标光标在哪一个格子
        x, y = mouse_pos
        for i, rect in enumerate(self.backpack_rect[4]):
            if rect.collidepoint(mouse_pos):
                return 4, i
        for line in range(4):
            if self.backpack_rect[line][0].top-2 <= y < self.backpack_rect[line][0].bottom+2:
                if self.backpack_rect[line][0].left - 2 <= x < self.backpack_rect[line][-1].right + 2:
                    return line, (x - (self.backpack_rect[line][0].left - 2)) // 36
        return None, None

    def get_index_texture(self, i, j, num):  # 获取某个格子上的物品信息
        if i == 4 and j == 4:
            return CRAFTING.pick_res(self.backpack[4], self.backpack_num[4])
        return self.backpack[i][j], min(num, self.backpack_num[i][j])

    def pick_up(self, i, j, num):  # 拿起某个格子上的物品
        # 当前格子没物品，或拿的物品数量为 0
        if self.backpack[i][j] is None or self.backpack_num[i][j] == 0 or not num:
            return

        # 拿起
        self.focus_texture, self.focus_num = self.get_index_texture(i, j, num)
        self.focus_rect = self.backpack_rect[i][j].copy()
        self.move = True

        # 更新原有的物品数量
        self.backpack_num[i][j] -= self.focus_num
        if self.backpack_num[i][j] == 0:
            self.backpack[i][j] = None

    def put_down(self, i, j, num):
        # 手里没有指定数量的物品，或格子放不下指定的数量
        num = min(self.focus_num, num, self.focus_texture.max_num - self.backpack_num[i][j])

        # 放置物品
        self.backpack[i][j] = self.focus_texture
        self.backpack_num[i][j] += num  # 增加物品数

        # 更新手中已拿起的物品数量
        self.focus_num -= num
        if self.focus_num == 0:  # 物品全放完了
            self.focus_texture = self.focus_num = self.focus_rect = None
            self.move = False

    def change(self, i, j):  # 把当前拿的物品和当前格子里的物品交换
        self.focus_texture, self.backpack[i][j] = self.backpack[i][j], self.focus_texture
        self.focus_num, self.backpack_num[i][j] = self.backpack_num[i][j], self.focus_num

    def on_mouse_press(self, button, pos):  # 按下鼠标
        i, j = self.get_mouse_index(pos)
        if i is None or j is None or button not in (1, 3):  # 鼠标不在有效的背包格子中，或者不是有效的按键
            return
        pick_num = {1: self.backpack_num[i][j], 3: max(self.backpack_num[i][j]//2, 1)}
        put_num = {1: self.focus_num, 3: 1}
        if not self.move:  # 不在拖动物品
            self.pick_up(i, j, pick_num[button])
        elif i != 4 or j != 4:  # 不是合成格
            if self.backpack[i][j] is None or self.backpack[i][j].add == self.focus_texture.add:  # 同种物品或无物品
                self.put_down(i, j, put_num[button])  # 放物品
            else:
                self.change(i, j)  # 否则交换物品
        elif self.backpack[4][4] and self.backpack[4][4].add == self.focus_texture.add:
            if self.backpack_num[i][j] + self.focus_num <= self.focus_texture.max_num:
                self.focus_num += CRAFTING.pick_res(self.backpack[4], self.backpack_num[4])[1]

        self.backpack[4][4], self.backpack_num[4][4] = CRAFTING.check(self.backpack[4][:4], self.backpack_num[4][:4])

    def on_mouse_motion(self, rel):
        if self.move:
            self.focus_rect.move_ip(*rel)


class CancelText:
    def __init__(self):
        self.start_time = None
        self._sino = lambda x: math.sin(x * (math.pi / 2))

        w, h = 150 + 50, 40
        self.sf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(self.sf, (0, 0, 0), (0, 0, w, h), border_radius=5)
        text = pygame.font.SysFont('SimHei', int(h // 2.5)).render('按 ESC 可显示鼠标光标', True, (255,) * 3)
        self.sf.blit(text, text.get_rect(center=self.sf.get_rect().center))

    def start_anim(self):
        self.start_time = time.time()

    def end_anim(self):
        self.start_time = None

    def render(self, screen):
        if not self.start_time:
            return
        time_sub = time.time() - self.start_time
        if time_sub < 0.4:
            num = self._sino(time_sub / 0.4)
            w = int(150 + 50 * num)
            h = int(w / 5)
            alpha = int(150 * num)
        elif 0.4 <= time_sub < 3.4:
            w, h, alpha = 150 + 50, 40, 150
        elif 3.4 <= time_sub < 4.4:
            w, h, alpha = 150 + 50, 40, int(150 - 150 * (time_sub - 3.4))
        else:
            self.end_anim()
            return
        self.sf.set_alpha(alpha)
        screen.blit(pygame.transform.smoothscale(self.sf, (w, h)), ((screen.get_width()-w)//2, 30 + (40 - h) // 2))


class Window:
    def __init__(self, screen):
        self.exclusive = False  # 是否独占鼠标的窗口

        self.screen = screen

        # 当飞行时，重力无效且速度增加
        self.flying = False

        # 用于持续跳跃。如果按住空格键，则为True；否则为False
        self.jumping = False

        # 如果玩家实际进行了跳跃，则为True
        self.jumped = False

        # 如果为True，则在最终的glTranslate中添加蹲伏偏移量
        self.crouch = False

        # 玩家冲刺
        self.sprinting = False

        # 这是一个偏移值，因此可以轻松添加速度药水等内容
        self.fov_offset = 0

        self.collision_types = {"top": False, "bottom": False, "right": False, "left": False}

        self.strafe = [0, 0]

        # 当前世界坐标(x、y、z)位置，使用浮点数表示。请注意，
        # 与数学课程不同，y轴是垂直轴。
        self.position = (30, 50, 80)

        # 第一个元素是在x-z平面(地面)上玩家的旋转角度，从z轴向下测量。
        # 第二个元素是从地面到上方的旋转角度。旋转以度为单位。
        #
        # 垂直平面旋转范围从-90(看向直线下方)到90(看向直线上方)。
        # 水平旋转范围无限制。
        self.rotation = (0, 0)

        # 玩家当前所在的区块
        self.sector = None

        # 屏幕中央的准星
        self.reticle = None

        # 在y(向上)方向的速度
        self.dy = 0

        # 便捷的数字键列表。
        self.num_keys = [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                         pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]

        self.interface = 'main'  # 当前界面 main: 主界面， backpack: 背包界面

        # 处理世界的模型实例。
        self.model = Model()
        self.inventory = Inventory(screen)
        self.cancel_text = CancelText()
        self.backpack = BackPack(screen, self.inventory)
        self.last_time = time.time()

    def set_exclusive_mouse(self, exclusive):
        """如果`exclusive`为True，则游戏将捕获鼠标；如果为False，则游戏将忽略鼠标。"""
        pygame.event.set_grab(exclusive)
        pygame.mouse.set_visible(not exclusive)
        self.exclusive = exclusive

    def get_sight_vector(self):
        return get_sight_vector(*self.rotation)

    def get_motion_vector(self):
        """返回当前运动向量，指示玩家的速度。

        返回
        -------
        vector : 长度为3的元组
            包含x、y和z速度的元组。

        """
        if any(self.strafe):
            x, y = self.rotation
            strafe = math.degrees(math.atan2(*self.strafe))
            y_angle = math.radians(y)
            x_angle = math.radians(x + strafe)
            if self.flying:
                m = math.cos(y_angle)
                dy = math.sin(y_angle)
                if self.strafe[1]:
                    # 向左或向右移动。
                    dy = 0.0
                    m = 1
                if self.strafe[0] > 0:
                    # 向后移动。
                    dy *= -1
                # 当你向上或向下飞行时，左右运动更少。
                dx = math.cos(x_angle) * m
                dz = math.sin(x_angle) * m
            else:
                dy = 0.0
                dx = math.cos(x_angle)
                dz = math.sin(x_angle)
        else:
            dy = dx = dz = 0.0
        return dx, dy, dz

    def update(self):
        """这个方法被pyglet的时钟重复调用。

        参数
        ----------
        dt : 浮点数
            自上次调用以来的时间变化量。

        """
        dt = time.time() - self.last_time  # 1.0 / TICKS_PER_SEC
        self.last_time = time.time()
        self.model.process_queue()
        sector = sectorize(self.position)
        if sector != self.sector:
            self.model.change_sectors(self.sector, sector)
            if self.sector is None:
                self.model.process_entire_queue()
            self.sector = sector
        m = 8
        dt = min(dt, 0.2)
        for _ in np.arange(m):
            self._update(dt / m)

    def _update(self, dt):
        """ update() 方法的私有实现。这是大部分运动逻辑、重力和碰撞检测的地方。

        参数
        ----------
        dt : float
            自上次调用以来的时间变化量。

        """
        # 行走
        if self.flying:
            speed = FLYING_SPEED  # 飞行速度
        elif self.sprinting:
            speed = SPRINT_SPEED  # 冲刺速度
        elif self.crouch:
            speed = CROUCH_SPEED  # 蹲伏速度
        else:
            speed = WALKING_SPEED  # 步行速度

        if self.jumping:
            if self.collision_types["top"]:
                self.dy = JUMP_SPEED  # 跳跃速度
                self.jumped = True
        else:
            if self.collision_types["top"]:
                self.jumped = False
        if self.jumped:
            speed += 0.7

        d = dt * speed  # 这一帧移动的距离
        dx, dy, dz = self.get_motion_vector()
        # 新的空间位置，尚未考虑重力。
        dx, dy, dz = dx * d, dy * d, dz * d
        # 重力
        if not self.flying:
            # 更新垂直速度：如果你在下落，加速直到达到极限速度；如果你在跳跃，减速直到开始下落。
            self.dy -= dt * GRAVITY  # 重力加速度
            self.dy = max(self.dy, -TERMINAL_VELOCITY)  # 最大下降速度
            dy += self.dy * dt
        # 碰撞检测
        old_pos = self.position
        x, y, z = old_pos
        x, y, z = self.collide((x + dx, y + dy, z + dz), PLAYER_HEIGHT)
        self.position = (x, y, z)

        # 冲刺相关。如果玩家在 x 和 z 方向上停止移动，则停止冲刺并从视野偏移中减去冲刺视野。
        if old_pos[0] - self.position[0] == 0 and old_pos[2] - self.position[2] == 0:
            disablefov = False
            if self.sprinting:
                disablefov = True
            self.sprinting = False
            if disablefov:
                self.fov_offset -= SPRINT_FOV

    def collide(self, position, height):
        """ 检查给定位置和高度的玩家是否与世界中的任何方块碰撞。

        参数
        ----------
        position : 长度为 3 的元组
            要检查碰撞的 (x, y, z) 位置。
        height : int 或 float
            玩家的高度。

        返回
        -------
        position : 长度为 3 的元组
            考虑到碰撞后玩家的新位置。

        """
        # 需要有多少与周围方块的维度重叠才能算作碰撞。
        # 如果为 0，则任何与地形接触都算作碰撞。
        # 如果为0.49，则你会陷入地面，就像走在高草里一样。
        # 如果 >= 0.5，你会穿过地面。
        pad = 0.25
        p = list(position)  # 当前坐标
        np_ = normalize(position)  # 强转成整数后的
        self.collision_types = {"top": False, "bottom": False, "right": False, "left": False}
        for face in FACES:  # 检查所有周围的方块
            for i in range(3):  # 按照每个维度单独检查
                if not face[i]:  # 当前维度无需检查
                    continue
                # 你在这个维度上的重叠程度。
                d = (p[i] - np_[i]) * face[i]
                if d < pad:
                    continue
                for dy in range(height):  # 检查每个高度
                    op = list(np_)
                    op[1] -= dy
                    op[i] += face[i]
                    if tuple(op) not in self.model.world:
                        continue
                    p[i] -= (d - pad) * face[i]
                    # 如果你与地面或天花板发生碰撞，则停止下落/上升。
                    if face == (0, -1, 0):
                        self.collision_types["top"] = True
                        self.dy = 0
                    if face == (0, 1, 0):
                        self.collision_types["bottom"] = True
                        self.dy = 0
                    break
        return tuple(p)

    def handle_event(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.inventory.inventory_index -= 1
                    self.inventory.inventory_index %= 9
                    continue
                if event.button == 5:
                    self.inventory.inventory_index += 1
                    self.inventory.inventory_index %= 9
                    continue
                self.on_mouse_press(event.button, event.pos)
            elif event.type == pygame.MOUSEMOTION:
                self.on_mouse_motion(*event.rel)
            elif event.type == pygame.KEYDOWN:
                self.on_key_press(event.key)
            elif event.type == pygame.KEYUP:
                self.on_key_release(event.key)
            elif event.type == pygame.VIDEORESIZE:
                self.backpack.update_size()
                self.inventory.update_size()

    @staticmethod
    def check_ctrl_down():
        return pygame.key.get_pressed()[pygame.K_LCTRL] or pygame.key.get_pressed()[pygame.K_RCTRL]

    def calc_length(self, previous):
        length_a = math.sqrt((previous[0] - self.position[0]) ** 2 +
                             (previous[1] - self.position[1]) ** 2 +
                             (previous[2] - self.position[2]) ** 2)
        length_b = math.sqrt((previous[0] - self.position[0]) ** 2 +
                             (previous[1] - self.position[1] + 1) ** 2 +
                             (previous[2] - self.position[2]) ** 2)
        return length_a >= 0.7 and length_b >= 0.7

    def on_mouse_press(self, button, pos):
        """ 当鼠标按下时调用。"""
        if self.interface == 'backpack':
            self.backpack.on_mouse_press(button, pos)
            return
        if self.exclusive:
            vector = self.get_sight_vector()
            block, previous = self.model.hit_test(self.position, vector)
            if button == 3 or (button == 1 and self.check_ctrl_down()):
                if previous and self.calc_length(previous):
                    block = self.inventory.get_block()
                    if block:
                        self.model.add_block(previous, block)
            elif button == 1 and block:
                texture = self.model.world[block]
                if texture.add != BEDROCK.add:
                    self.model.remove_block(block)
                    # self.model.add_drop(texture, block)
        else:
            self.set_exclusive_mouse(True)
            self.cancel_text.start_anim()

    def on_mouse_motion(self, dx, dy):
        """ 当玩家移动鼠标时调用。

        参数
        ----------
        x, y : int
            鼠标点击的坐标。如果鼠标被捕获，则始终为屏幕中心。
        dx, dy : float
            鼠标的移动距离。

        """
        if self.exclusive and self.interface == 'main':
            dy = -dy
            m = 0.15
            x, y = self.rotation
            x, y = x + dx * m, y + dy * m
            y = max(-90, min(90, y))
            self.rotation = (x, y)
        if self.interface == 'backpack':
            self.backpack.on_mouse_motion((dx, dy))

    def on_key_press(self, symbol):
        """ 当玩家按下键时调用。请参阅 pyglet 文档以获取键的映射。

        参数
        ----------
        symbol : int
            表示被按下的键的数字。
        modifiers : int
            表示在鼠标按钮被点击时按下的任何修改键的数字。

        """
        if symbol == pygame.K_w:
            self.strafe[0] -= 1
        elif symbol == pygame.K_s:
            self.strafe[0] += 1
        elif symbol == pygame.K_a:
            self.strafe[1] -= 1
        elif symbol == pygame.K_d:
            self.strafe[1] += 1
        elif symbol == pygame.K_c:
            self.fov_offset -= 60.0  # 望远镜模式（自己编的，作者没玩过mc）
        elif symbol == pygame.K_e:
            self.interface = {'main': 'backpack', 'backpack': 'main'}[self.interface]
            pygame.mouse.set_visible(True if not self.exclusive else self.interface == 'backpack')
            # TODO: 111
            self.backpack.focus_texture = self.backpack.focus_num = self.backpack.focus_rect = None
            self.backpack.move = False
        elif symbol == pygame.K_SPACE:
            self.jumping = True
        elif symbol == pygame.K_ESCAPE:
            self.set_exclusive_mouse(False)
            self.cancel_text.end_anim()
        elif symbol == pygame.K_LSHIFT:
            self.crouch = True  # 潜行
            if self.sprinting:
                self.fov_offset -= SPRINT_FOV
                self.sprinting = False
        elif symbol == pygame.K_r:
            if not self.crouch:
                if not self.sprinting:
                    self.fov_offset += SPRINT_FOV
                self.sprinting = True  # 疾跑
        elif symbol == pygame.K_TAB:
            self.flying = not self.flying
        elif symbol in self.num_keys:
            index = (symbol - 1 - self.num_keys[0]) % 9
            self.inventory.set_index(index)

    def on_key_release(self, symbol):
        """当玩家释放一个键时调用。symbol代表按下的键，modifiers代表其他同时按下的修饰键。

        参数
        ----------
        symbol : int
            表示被按下的键的数字
        modifiers : int
            表示同时按下的修饰键的数字

        """
        if symbol == pygame.K_w:
            self.strafe[0] += 1
        elif symbol == pygame.K_s:
            self.strafe[0] -= 1
        elif symbol == pygame.K_a:
            self.strafe[1] += 1
        elif symbol == pygame.K_d:
            self.strafe[1] -= 1
        elif symbol == pygame.K_SPACE:
            self.jumping = False
        elif symbol == pygame.K_LSHIFT:
            self.crouch = False
        elif symbol == pygame.K_c:
            self.fov_offset += 60.0

    def get_focused_block(self):
        """在准星下方的方块周围绘制黑色边界。

        """
        vector = get_sight_vector(*self.rotation)
        block = self.model.hit_test(self.position, vector)[0]
        if block:
            texture = self.model.world[block]
            x, y, z = block
            new_vertices = texture.vertices.copy()
            new_vertices[:, 0, 0] += x
            new_vertices[:, 1, 0] += y
            new_vertices[:, 2, 0] += z
            return NoneModel(all_textures.texture_array_focused, new_vertices, all_textures.uv_vertices,
                             texture.norms, texture.uv_indices, texture.indices,
                             all_textures.texture_width, all_textures.texture_height)

    def render_ui(self, width, height):
        pygame.draw.line(self.screen, (0,) * 3, (width // 2 - 10, height // 2), (width // 2 + 10, height // 2))
        pygame.draw.line(self.screen, (0,) * 3, (width // 2, height // 2 - 10), (width // 2, height // 2 + 10))
        self.inventory.render()
        if self.interface == 'backpack':
            self.backpack.render()
        self.cancel_text.render(self.screen)

    def on_draw(self, zbuffer):
        """由pyglet调用，绘制画布。

        """
        width, height = self.screen.get_size()
        res = self.model.render()
        res1 = self.model.render_drop()

        x, y = self.rotation
        angle1 = glRotatef(x, 0, 1, 0)
        angle2 = glRotatef(-y, math.cos(math.radians(x)), 0, math.sin(math.radians(x)))
        x, y, z = self.position
        if self.crouch:
            tx, ty, tz = -x, -y + 0.2, -z
        else:
            tx, ty, tz = -x, -y, -z
        if not res:
            render_cube(self.model.model, height, width, self.screen, zbuffer, angle1, angle2,
                        tx, ty, tz, PLAYER_FOV + self.fov_offset)
            focused_model = self.get_focused_block()
            if focused_model:
                render_cube(focused_model, height, width, self.screen, zbuffer, angle1, angle2,
                            tx, ty, tz, PLAYER_FOV + self.fov_offset)
        if not res1:
            render_cube(self.model.model_drop, height, width, self.screen, zbuffer, angle1, angle2,
                        tx, ty, tz, PLAYER_FOV + self.fov_offset)
        self.render_ui(width, height)
