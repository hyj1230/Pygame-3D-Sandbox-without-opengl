from render import Cube, Texture
import math
import crafting


# FPS限制
TICKS_PER_SEC = 60

# 区块大小
SECTOR_SIZE = 2

# 移动变量
WALKING_SPEED = 5  # 步行速度
FLYING_SPEED = 15  # 飞行速度
CROUCH_SPEED = 2  # 蹲伏速度
SPRINT_SPEED = 7  # 冲刺速度
SPRINT_FOV = SPRINT_SPEED / 2  # 冲刺时的视野增大

GRAVITY = 20.0  # 重力加速度
MAX_JUMP_HEIGHT = 1.0  # 最大跳跃高度
JUMP_SPEED = math.sqrt(2 * GRAVITY * MAX_JUMP_HEIGHT)  # 跳跃速度
TERMINAL_VELOCITY = 50  # 终端速度，即下落的最大速度

# 玩家变量
PLAYER_HEIGHT = 2  # 玩家身高
PLAYER_FOV = 80.0  # 玩家视野

FACES = [
    (0, 1, 0),
    (0, -1, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (0, 0, -1),
]

all_textures = Texture('data/cube/texture.obj', 'data/cube/texture_cube.png', 'data/cube/texture_cube_focused.png')

GRASS = Cube(filename="data/cube/cs1.obj", add=6, icon='data/texture/草方块.png')  # 草方块
SAND = Cube(filename="data/cube/cs1.obj", add=2, icon='data/texture/沙子.png')  # 沙子
BRICK = Cube(filename="data/cube/cs1.obj", add=5, icon='data/texture/砖块.png')  # 砖块
STONE = Cube(filename="data/cube/cs1.obj", add=4, icon='data/texture/石头.png')  # 石头
WOOD = Cube(filename="data/cube/cs1.obj", add=3, icon='data/texture/树干.png')  # 树干
LEAF = Cube(filename="data/cube/cs1.obj", add=7, icon='data/texture/树叶.png')  # 叶子
WATER = Cube(filename="data/cube/cs1.obj", add=1, icon='data/texture/彩蛋.png')  # 水
DIRT = Cube(filename="data/cube/cs1.obj", add=0, icon='data/texture/泥土.png')  # 泥土
BEDROCK = Cube(filename="data/cube/cs1.obj", add=8, icon='data/texture/基岩.png')  # 基岩
ROCK = Cube(filename="data/cube/cs1.obj", add=9, icon='data/texture/圆石.png')  # 圆石
WOODPLANK = Cube(filename="data/cube/cs1.obj", add=10, icon='data/texture/木板.png')  # 木板
CRAFTINGTABLE = Cube(filename="data/cube/cs2.obj", add=11, icon='data/texture/工作台.png')  # 工作台


CRAFTING = crafting.Crafting()
CRAFTING.add((WOOD, None, None, None), (1, 0, 0, 0), WOODPLANK, 4)
CRAFTING.add((None, WOOD, None, None), (0, 1, 0, 0), WOODPLANK, 4)
CRAFTING.add((None, None, WOOD, None), (0, 0, 1, 0), WOODPLANK, 4)
CRAFTING.add((None, None, None, WOOD), (0, 0, 0, 1), WOODPLANK, 4)
CRAFTING.add((WOODPLANK,)*4, (1,)*4, CRAFTINGTABLE, 1)
