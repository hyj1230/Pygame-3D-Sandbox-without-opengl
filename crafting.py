class Crafting:
    def __init__(self):
        self.crafting = {}
        self.crafting_num = {}

    def add(self, crafting, crafting_num, crafting_res, crafting_res_num):
        crafting = self.change(crafting)
        self.crafting[crafting] = crafting_res
        self.crafting_num[crafting] = crafting_num, crafting_res_num

    @staticmethod
    def change(texture):
        return tuple([x.add if x is not None else None for x in texture])

    def check(self, texture, num):
        tuple_texture = self.change(texture)
        if tuple_texture not in self.crafting:
            return None, 0
        crafting_num, crafting_res_num = self.crafting_num[tuple_texture]
        for i in range(len(crafting_num)):
            if crafting_num[i] > num[i]:
                return None, 0
        return self.crafting[tuple_texture], crafting_res_num

    def pick_res(self, texture, num):
        res, res_num = self.check(texture[:4], num[:4])
        if res is None:
            return None, None
        tuple_texture = self.change(texture[:4])
        crafting_num, _ = self.crafting_num[tuple_texture]
        for i in range(4):
            num[i] -= crafting_num[i]
            if num[i] == 0:
                texture[i] = None
        return res, res_num
