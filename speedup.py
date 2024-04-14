import numpy as np
from numba import jit, prange


@jit('UniTuple(int32, 2)(float64,float64,float64)', nopython=True, cache=True)
def get_min_max(a, b, c):
    return int(min(min(a, b), c)), int(max(max(a, b), c))


@jit('int32(int32,int32,int32)', nopython=True, cache=True)
def clip(a, b, c):
    return max(b, min(a, c))


@jit('UniTuple(float64, 2)(float64,float64,float64,float64,float64,float64)', nopython=True, cache=True, fastmath=True)
def cross_product(a0, a1, b2, bc_c, a2b1, a2b0):
    x = a1 * b2 - a2b1
    y = a2b0 - a0 * b2
    return x/bc_c, y/bc_c


@jit('float64(float64[:],float64[:])', nopython=True, cache=True, fastmath=True)
def get_intersect_ratio(prev, curv):
    return (prev[3] - 0.1) / (prev[3] - curv[3])


@jit('uint8(float64[:])', nopython=True, cache=True, fastmath=True)
def is_inside_plane(pts):
    return pts[3] >= 0.1


@jit('float64[:](float64[:],float64[:],float64)', nopython=True, cache=True, fastmath=True)
def lerp_vec4(start, end, alpha):
    return start + (end - start) * alpha


@jit('float64[:](float64[:],float64[:],float64)', nopython=True, cache=True, fastmath=True)
def lerp_vec2(start, end, alpha):
    return start + (end - start) * alpha


@jit('float64(float64,float64,float64)', nopython=True, cache=True, fastmath=True)
def lerp_num(start, end, alpha):
    return start + (end - start) * alpha


@jit(nopython=True, cache=True, fastmath=True, looplift=True)
def render_clip_face(screen, zbuffer, ptsa, ptsb, ptsc, norms, i, uva, uvb, uvc,
                     clip_a, clip_b, clip_c, width, height, texture_array, O2):
    pts2a = ptsa[0] / ptsa[3], ptsa[1] / ptsa[3]
    pts2b = ptsb[0] / ptsb[3], ptsb[1] / ptsb[3]
    pts2c = ptsc[0] / ptsc[3], ptsc[1] / ptsc[3]
    bc_c = (pts2c[0] - pts2a[0]) * (pts2b[1] - pts2a[1]) - (pts2b[0] - pts2a[0]) * (pts2c[1] - pts2a[1])
    if (-bc_c if O2 else abs(bc_c)) < 1e-3:
        return
    ptsa, ptsb, ptsc = 1 / ptsa[3], 1 / ptsb[3], 1 / ptsc[3]
    minx, maxx = get_min_max(pts2a[0], pts2b[0], pts2c[0])
    miny, maxy = get_min_max(pts2a[1], pts2b[1], pts2c[1])
    minx, maxx = clip(minx, 0, screen.shape[0] - 1), clip(maxx + 1, 1, screen.shape[0])
    miny, maxy = clip(miny, 0, screen.shape[1] - 1), clip(maxy + 1, 1, screen.shape[1])
    # intensity = norms[i % 12]
    a, b, c, d = pts2c[0] - pts2a[0], pts2b[0] - pts2a[0], pts2c[1] - pts2a[1], pts2b[1] - pts2a[1]
    uva = uva[0] * width * ptsa, uva[1] * height * ptsa
    uvb = uvb[0] * width * ptsb, uvb[1] * height * ptsb
    uvc = uvc[0] * width * ptsc, uvc[1] * height * ptsc
    clip_a, clip_b, clip_c = clip_a * ptsa, clip_b * ptsb, clip_c * ptsc
    for j in prange(minx, maxx):
        flag = False
        temp = pts2a[0] - float(j)
        m0, m1 = cross_product(a, b, pts2a[1] - miny + 1, bc_c, temp * d, temp * c)
        addm0, addm1 = -b / bc_c, a / bc_c  # 重心坐标使用递推计算
        m2, addm2 = 1 - m1 - m0, -(addm0 + addm1)
        bc_clip_sum = m2 * ptsa + m1 * ptsb + m0 * ptsc
        add_sum = addm2 * ptsa + addm1 * ptsb + addm0 * ptsc
        for k in prange(screen.shape[1] - miny - 1, screen.shape[1] - maxy - 1, -1):
            # 必须显式转换成 double 参与底下的运算，不然结果是错的
            m0 += addm0
            m1 += addm1
            m2 += addm2
            bc_clip_sum += add_sum

            if m0 < 0 or m1 < 0 or m2 < 0:
                if flag:  # 优化：当可以确定超过的是右边界，可以直接换行
                    break
                continue
            flag = True

            bc_clip = (m2 / bc_clip_sum, m1 / bc_clip_sum, m0 / bc_clip_sum)

            frag_depth = clip_a * bc_clip[0] + clip_b * bc_clip[1] + clip_c * bc_clip[2]

            if frag_depth > zbuffer[j, k]:
                continue
            zbuffer[j, k] = frag_depth
            # Blender 导出来的 uv 数据，跟之前的顶点数据有一样的问题，Y轴是个反的，
            # 所以这里的纹理图片要旋转一下才能 work
            u = uva[1] * bc_clip[0] + uvb[1] * bc_clip[1] + uvc[1] * bc_clip[2]
            v = uva[0] * bc_clip[0] + uvb[0] * bc_clip[1] + uvc[0] * bc_clip[2]
            color = texture_array[int(u), int(v)]
            screen[j, k] = color[0], color[1], color[2]


@jit(nopython=True, cache=True, fastmath=True, looplift=True)
def generate_faces_flat(indices, uv_indices, pts, uv_triangle,
                        clip_vert, norms, width, height, screen, texture_array, zbuffer, O2):
    # 使用 z-buffer 算法绘制三角形，以及 flat 着色

    length: int = indices.shape[0]  # 三角形总个数

    for i in prange(length):
        ptsa, ptsb, ptsc = pts[indices[i, 0]], pts[indices[i, 1]], pts[indices[i, 2]]
        uva = uv_triangle[uv_indices[i, 0], 0], uv_triangle[uv_indices[i, 0], 1]
        uvb = uv_triangle[uv_indices[i, 1], 0], uv_triangle[uv_indices[i, 1], 1]
        uvc = uv_triangle[uv_indices[i, 2], 0], uv_triangle[uv_indices[i, 2], 1]
        clip_a = clip_vert[indices[i, 0], 2]
        clip_b = clip_vert[indices[i, 1], 2]
        clip_c = clip_vert[indices[i, 2], 2]
        nums = (ptsa[3] < 0.1) + (ptsb[3] < 0.1) + (ptsc[3] < 0.1)  # 指有几个点在屏幕外
        if nums:  # 透视裁剪
            if nums == 3:
                continue
            out_vert_num = 0
            out_pts = np.empty((4, 4), dtype=np.float64)
            out_uv = np.empty((4, 2), dtype=np.float64)
            out_clip = np.empty(4, dtype=np.float64)
            for j in range(3):
                curv_index = j
                prev_index = (j - 1 + 3) % 3
                curv = pts[indices[i, curv_index]]
                prev = pts[indices[i, prev_index]]
                is_cur_inside = is_inside_plane(curv)
                is_pre_inside = is_inside_plane(prev)
                if is_cur_inside != is_pre_inside:
                    ratio = get_intersect_ratio(prev, curv)
                    out_pts[out_vert_num] = lerp_vec4(prev, curv, ratio)
                    out_uv[out_vert_num] = lerp_vec2(uv_triangle[uv_indices[i, prev_index]],
                                                     uv_triangle[uv_indices[i, curv_index]], ratio)
                    out_clip[out_vert_num] = lerp_num(clip_vert[indices[i, prev_index], 2],
                                                      clip_vert[indices[i, curv_index], 2], ratio)
                    out_vert_num += 1
                if is_cur_inside:
                    out_pts[out_vert_num] = curv
                    out_uv[out_vert_num] = uv_triangle[uv_indices[i, curv_index]]
                    out_clip[out_vert_num] = clip_vert[indices[i, curv_index], 2]
                    out_vert_num += 1
            if out_vert_num == 3:
                uva = out_uv[0, 0], out_uv[0, 1]
                uvb = out_uv[1, 0], out_uv[1, 1]
                uvc = out_uv[2, 0], out_uv[2, 1]
                ptsa, ptsb, ptsc = out_pts[0], out_pts[1], out_pts[2]
                clip_a, clip_b, clip_c = out_clip[0], out_clip[1], out_clip[2]
            elif out_vert_num == 4:
                uva = out_uv[0, 0], out_uv[0, 1]
                uvb = out_uv[1, 0], out_uv[1, 1]
                uvc = out_uv[2, 0], out_uv[2, 1]
                ptsa, ptsb, ptsc = out_pts[0], out_pts[1], out_pts[2]
                clip_a, clip_b, clip_c = out_clip[0], out_clip[1], out_clip[2]
                render_clip_face(screen, zbuffer, ptsa, ptsb, ptsc, norms, i, uva, uvb, uvc,
                                 clip_a, clip_b, clip_c, width, height, texture_array, O2)
                uva = out_uv[0, 0], out_uv[0, 1]
                uvb = out_uv[2, 0], out_uv[2, 1]
                uvc = out_uv[3, 0], out_uv[3, 1]
                ptsa, ptsb, ptsc = out_pts[0], out_pts[2], out_pts[3]
                clip_a, clip_b, clip_c = out_clip[0], out_clip[2], out_clip[3]

        pts2a = ptsa[0] / ptsa[3], ptsa[1] / ptsa[3]
        pts2b = ptsb[0] / ptsb[3], ptsb[1] / ptsb[3]
        pts2c = ptsc[0] / ptsc[3], ptsc[1] / ptsc[3]
        bc_c = (pts2c[0] - pts2a[0]) * (pts2b[1] - pts2a[1]) - (pts2b[0] - pts2a[0]) * (pts2c[1] - pts2a[1])
        if (-bc_c if O2 else abs(bc_c)) < 1e-3:
            continue
        ptsa, ptsb, ptsc = 1 / ptsa[3], 1 / ptsb[3], 1 / ptsc[3]
        minx, maxx = get_min_max(pts2a[0], pts2b[0], pts2c[0])
        miny, maxy = get_min_max(pts2a[1], pts2b[1], pts2c[1])
        minx, maxx = clip(minx, 0, screen.shape[0]-1), clip(maxx+1, 1, screen.shape[0])
        miny, maxy = clip(miny, 0, screen.shape[1]-1), clip(maxy+1, 1, screen.shape[1])
        # intensity = norms[i % 12]
        a, b, c, d = pts2c[0] - pts2a[0], pts2b[0] - pts2a[0], pts2c[1] - pts2a[1], pts2b[1] - pts2a[1]
        uva = uva[0] * width * ptsa, uva[1] * height * ptsa
        uvb = uvb[0] * width * ptsb, uvb[1] * height * ptsb
        uvc = uvc[0] * width * ptsc, uvc[1] * height * ptsc
        clip_a, clip_b, clip_c = clip_a * ptsa, clip_b * ptsb, clip_c * ptsc
        for j in prange(minx, maxx):
            flag = False
            temp = pts2a[0] - float(j)
            m0, m1 = cross_product(a, b, pts2a[1] - miny + 1, bc_c, temp * d, temp * c)
            addm0, addm1 = -b / bc_c, a / bc_c  # 重心坐标使用递推计算
            m2, addm2 = 1 - m1 - m0, -(addm0 + addm1)
            bc_clip_sum = m2 * ptsa + m1 * ptsb + m0 * ptsc
            add_sum = addm2 * ptsa + addm1 * ptsb + addm0 * ptsc
            for k in prange(screen.shape[1]-miny-1, screen.shape[1]-maxy-1, -1):
                # 必须显式转换成 double 参与底下的运算，不然结果是错的
                m0 += addm0
                m1 += addm1
                m2 += addm2
                bc_clip_sum += add_sum

                if m0 < 0 or m1 < 0 or m2 < 0:
                    if flag:  # 优化：当可以确定超过的是右边界，可以直接换行
                        break
                    continue
                flag = True

                bc_clip = (m2 / bc_clip_sum, m1 / bc_clip_sum, m0 / bc_clip_sum)

                frag_depth = clip_a * bc_clip[0] + clip_b * bc_clip[1] + clip_c * bc_clip[2]

                if frag_depth > zbuffer[j, k]:
                    continue
                zbuffer[j, k] = frag_depth
                # Blender 导出来的 uv 数据，跟之前的顶点数据有一样的问题，Y轴是个反的，
                # 所以这里的纹理图片要旋转一下才能 work
                u = uva[1] * bc_clip[0] + uvb[1] * bc_clip[1] + uvc[1] * bc_clip[2]
                v = uva[0] * bc_clip[0] + uvb[0] * bc_clip[1] + uvc[0] * bc_clip[2]
                color = texture_array[int(u), int(v)]
                screen[j, k] = color[0], color[1], color[2]
