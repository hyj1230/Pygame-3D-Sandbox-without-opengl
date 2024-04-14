import math


# 噪声参数类，用于存储噪声生成的参数
class NoiseParameters:
    def __init__(self, octaves, amplitude, smoothness, roughness, heightOffset):
        # 噪声的八度数。在噪声生成中，八度指的是生成多个不同频率的噪声来创建复杂的地形特征。较高的八度数会生成更多的细节，但也可能增加计算量和渲染时间。
        self.octaves = octaves

        # 噪声的幅度。幅度决定了噪声对地形高度的影响程度。较大的幅度会产生更大的地形变化，而较小的幅度会产生更平缓的地形。
        self.amplitude = amplitude

        # 噪声的平滑度。平滑度越高，生成的地形变化越平滑。较高的平滑度会使地形变得更加连续，而较低的平滑度会产生更多的尖锐变化。
        self.smoothness = smoothness

        # 噪声的粗糙度。粗糙度越高，地形变化越复杂。较高的粗糙度会生成更多的细节和不规则形状，而较低的粗糙度会产生更平坦的地形。
        self.roughness = roughness

        # 地形高度的偏移量。它可以用来调整整体地形的基本高度。通过增加或减少偏移量，可以改变地形的海拔高度。
        self.heightOffset = heightOffset


# 噪声生成器类
class NoiseGen:
    def __init__(self, seed):
        self.seed = seed
        self.noiseParams = NoiseParameters(
            7, 50, 450, 0.3, 20
        )

    # 内部方法，用于生成伪随机数。它接受一个整数作为输入，并返回一个 0 到 1 之间的浮点数。
    def _getNoise2(self, n):
        n += self.seed
        n = (int(n) << 13) ^ int(n)
        newn = (n * (n * n * 60493 + 19990303) + 1376312589) & 0x7fffffff
        return 1.0 - (float(newn) / 1073741824.0)

    # 内部方法，用于通过调用 _getNoise2 的方式生成噪声值。它接受 x 和 z 坐标作为输入，并返回噪声值。
    def _getNoise(self, x, z):
        return self._getNoise2(x + z * 57)

    # 内部方法，用于进行平滑插值。它接受两个值 a 和 b 以及插值因子 z 作为输入，并返回插值结果。
    def _lerp(self, a, b, z):
        mu2 = (1.0 - math.cos(z * 3.14)) / 2.0
        return a * (1 - mu2) + b * mu2

    # 内部方法，用于生成二维噪声。它接受x和z坐标作为输入，并返回二维噪声值。
    def _noise(self, x, z):
        floorX = float(int(x))
        floorZ = float(int(z))

        s = self._getNoise(floorX, floorZ)
        t = self._getNoise(floorX + 1, floorZ)
        u = self._getNoise(floorX, floorZ + 1)
        v = self._getNoise(floorX + 1, floorZ + 1)

        rec1 = self._lerp(s, t, x - floorX)
        rec2 = self._lerp(u, v, x - floorX)
        rec3 = self._lerp(rec1, rec2, z - floorZ)
        return rec3

    # 公共方法，用于获取指定位置的地形高度。它通过迭代计算多个不同频率和幅度的噪声，并按照指定的参数进行调整，最后返回地形高度值。
    def getHeight(self, x, z):
        totalValue = 0.0

        for a in range(self.noiseParams.octaves - 1):
            freq = math.pow(2.0, a)
            totalValue += self._noise(
                (float(x)) * freq / self.noiseParams.smoothness,
                (float(z)) * freq / self.noiseParams.smoothness
            ) * self.noiseParams.amplitude

        return (totalValue / 5) + self.noiseParams.heightOffset
