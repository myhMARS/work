import os
import sys
import time
import random
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider

BINARY_DIR = './binary'
SOURCE_DIR = './source'
EDGE_DIR = './edge'
RESULT_DIR = './res'

# 椭圆的初始参数
A_MIN = 30
A_MAX = 300
B_MIN = 30
B_MAX = 300
angle_init = 0

SIMILARITY_THRESHOLD = 10


def check_ellipe(ellipe):
    return B_MIN <= ellipe[1][0] <= B_MAX and A_MIN <= ellipe[1][1] <= A_MAX


def get_overlap_percent(image, ellipe, contours, thickness=1):
    img = np.zeros_like(image, np.uint8)
    try:
        cv2.ellipse(img, ellipe, (255, 0, 0), thickness=thickness)
    except:
        return 0.0
    contour_img = np.zeros_like(image, np.uint8)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
    overlap = cv2.bitwise_and(contour_img, img)

    overlap_size = np.sum(overlap > 0)
    target_size = np.sum(img > 0) + 1

    return overlap_size / target_size, contour_img, img, overlap


def show_conf(img, conf, ellipse):
    center = (int(ellipse[0][0]), int(ellipse[0][1]))  # 椭圆中心 (x, y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 0, 0)
    line_type = 2
    if conf < 0.8:
        font_color = (0, 255, 0)
    cv2.putText(img, f'conf: {conf:.3f}', (center[0] - 50, center[1] - 10), font, font_scale, font_color, line_type)


# 欧几里得距离计算
def euclidean_distance(ellipse1, ellipse2):
    # 两个椭圆的参数
    (cx_1, cy_1), (MA_1, ma_1), angle_1 = ellipse1
    (cx_2, cy_2), (MA_2, ma_2), angle_2 = ellipse2
    return np.sqrt((cx_1 - cx_2) ** 2 +
                   (cy_1 - cy_2) ** 2 +
                   (MA_1 - MA_2) ** 2 +
                   (ma_1 - ma_2) ** 2 +
                   (angle_2 - angle_2) ** 2)


def similarity_ellipes(ellipses):
    threshold_similarity = SIMILARITY_THRESHOLD
    if len(ellipses) <= 1:
        return ellipses
    unique_ellipses = [ellipses[0]]
    for i in range(1, len(ellipses)):
        current_ellipse, conf = ellipses[i]
        is_unique = True

        # 比较当前椭圆与已有的所有椭圆的相似度
        for unique_ellipse, conf in unique_ellipses:
            similarity = euclidean_distance(current_ellipse, unique_ellipse)
            if similarity < threshold_similarity:
                is_unique = False
                break

        # 如果当前椭圆与所有已有椭圆的相似度都高于阈值，则将其添加到唯一椭圆列表中
        if is_unique:
            unique_ellipses.append([current_ellipse, conf])
    print(f'del {len(ellipses) - len(unique_ellipses)} similar ellipses')
    return unique_ellipses  #


def set_limit(source):
    global A_MIN, A_MAX, B_MIN, B_MAX, angle_init

    def plot_ellipse(ax, center, width, height, angle, color):
        ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, edgecolor=color, fc='None', lw=2)
        ax.add_patch(ellipse)

    # 创建绘图和滑动条
    fig, ax = plt.subplots()
    plt.imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    plt.grid()
    plt.axis('on')
    plt.title('source')
    plt.subplots_adjust(left=0.1, bottom=0.43)  # 调整底部以容纳滑动条

    # 绘制初始椭圆
    center_init = (source.shape[1] // 2, source.shape[0] // 2)
    plot_ellipse(ax, center_init, A_MIN, B_MIN, angle_init, 'green')
    plot_ellipse(ax, center_init, A_MAX, B_MAX, angle_init, 'red')
    ax.set_xlim(0, source.shape[1])
    ax.set_ylim(source.shape[0], 0)
    ax.set_aspect('equal')

    # 创建滑动条轴
    axcolor = 'lightgoldenrodyellow'
    ax_x = plt.axes((0.1, 0.35, 0.65, 0.02), facecolor=axcolor)
    ax_y = plt.axes((0.1, 0.30, 0.65, 0.02), facecolor=axcolor)
    ax_A_MAX = plt.axes((0.1, 0.25, 0.65, 0.02), facecolor=axcolor)
    ax_B_MAX = plt.axes((0.1, 0.20, 0.65, 0.02), facecolor=axcolor)
    ax_A_MIN = plt.axes((0.1, 0.15, 0.65, 0.02), facecolor=axcolor)
    ax_B_MIN = plt.axes((0.1, 0.10, 0.65, 0.02), facecolor=axcolor)
    ax_angle = plt.axes((0.1, 0.05, 0.65, 0.02), facecolor=axcolor)

    # 创建滑动条
    slider_x = Slider(ax_x, 'x', 0, source.shape[1], valinit=center_init[0])
    slider_y = Slider(ax_y, 'y', 0, source.shape[0], valinit=center_init[1])
    slider_A_MAX = Slider(ax_A_MAX, 'X_MAX', 0.5, 1500, valinit=A_MAX)
    slider_B_MAX = Slider(ax_B_MAX, 'Y_MAX', 0.5, 1500, valinit=B_MAX)
    slider_A_MIN = Slider(ax_A_MIN, 'X_MIN', 0.5, 1500, valinit=A_MIN)
    slider_B_MIN = Slider(ax_B_MIN, 'Y_MIN', 0.5, 1500, valinit=B_MIN)
    slider_angle = Slider(ax_angle, 'Angle', 0, 180, valinit=angle_init)

    # 更新函数
    def update(val):
        global A_MIN, B_MIN, A_MAX, B_MAX, angle_init
        ax.clear()  # 清除当前的绘图
        plot_ellipse(ax, (slider_x.val, slider_y.val), slider_A_MAX.val, slider_B_MAX.val, slider_angle.val, 'red')
        plot_ellipse(ax, (slider_x.val, slider_y.val), slider_A_MIN.val, slider_B_MIN.val, slider_angle.val, 'green')
        ax.imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        ax.set_xlim(0, source.shape[1])
        ax.set_ylim(source.shape[0], 0)
        X_MIN = slider_A_MIN.val
        X_MAX = slider_A_MAX.val
        Y_MIN = slider_B_MIN.val
        Y_MAX = slider_B_MAX.val

        # 区分长短轴
        A_MAX = max(X_MAX, Y_MAX)
        A_MIN = min(X_MIN, Y_MIN)
        B_MAX = min(X_MAX, Y_MAX)
        B_MIN = min(X_MIN, Y_MIN)
        fig.canvas.draw_idle()

    # 连接滑动条事件到更新函数
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_A_MAX.on_changed(update)
    slider_A_MIN.on_changed(update)
    slider_B_MAX.on_changed(update)
    slider_B_MIN.on_changed(update)
    slider_angle.on_changed(update)
    try:
        plt.show()
    except:
        pass
    print(A_MIN, A_MAX, B_MIN, B_MAX)


def detect_ellipses(source_file, debug=False, save=False):
    # global A_MIN, A_MAX, B_MIN, B_MAX, angle_init
    start_time = time.time()

    # 加载图像
    source = cv2.imread(source_file)
    # height, width = source.shape[:2]
    # source = cv2.resize(source, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

    # 转换为灰度图像
    img_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (9, 9), 0)
    img_gray = cv2.medianBlur(img_gray, 5)  # 11
    # cv2.imshow('median', img_gray)
    # cv2.waitKey(0)
    binary = cv2.adaptiveThreshold(src=img_gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY, blockSize=11, C=1)
    # binary = cv2.medianBlur(binary, 11)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((3, 3), np.uint8)
    # binary = cv2.dilate(binary, kernel, iterations=3)
    # binary = cv2.erode(binary, kernel, iterations=1)
    edges = cv2.Canny(binary, 50, 150)
    # edges = cv2.dilate(edges, kernel, iterations=2)
    # edges = cv2.erode(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros_like(img_gray)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imshow('binary', img_gray)
    cv2.imshow('edges', edges)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(BINARY_DIR, os.path.basename(source_file)), img_gray)
    cv2.imwrite(os.path.join(EDGE_DIR, os.path.basename(source_file)), edges)
    start_rhte_time = time.time()
    ellipses = []
    print(os.path.basename(source_file))
    edge_points = np.column_stack(np.where(edges == 255))

    print(len(edge_points))
    # inverted_img = cv2.bitwise_not(binary)
    # cv2.imshow('img', inverted_img)
    # cv2.moveWindow('img', 0, 0)
    # cv2.waitKey(0)
    set_limit(source)
    iterations = 1000
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TimeRemainingColumn(),
                  TimeElapsedColumn()) as progress:
        running = progress.add_task(description="running progress", total=len(contours))
        for contour in contours:
            progress.advance(running, advance=1)
            if len(contour) < 5:
                continue
            # print(contour)
            res_ell = None
            best_overlap = 0
            i = 0
            while i < iterations:
                edge_points = random.sample(contour.reshape(-1, 2).tolist(), 5)
                # 霍夫变换 https://github.com/opencv/opencv/blob/4.x/samples/cpp/fitellipse.cpp
                # https://github.com/opencv/opencv/blob/d9a139f9e85a17b4c47dbca559ed90aef517c279/modules/imgproc/src/shapedescr.cpp#L504
                ellipse = cv2.fitEllipse(np.array(edge_points))
                i += 1
                if check_ellipe(ellipse):
                    overlap, contour_img, img, overlap_img = get_overlap_percent(img_gray, ellipse, contours)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        res_ell = ellipse
            # print(best_overlap)

            # 检查椭圆尺寸是否有效
            if res_ell and best_overlap > 0.6:
                print(best_overlap)
                # cv2.imshow('contour', contour_img)
                # cv2.imshow('ellipse', img)
                # cv2.imshow('overlap', overlap_img)
                # cv2.waitKey(0)
                ellipses.append([res_ell, best_overlap])
            else:
                if debug:
                    print(f"Invalid ellipse dimensions: {ellipse}")
    cv2.waitKey(0)
    # ellipses.append(res_ell)

    ellipses = similarity_ellipes(ellipses)
    end_time = time.time()

    if debug:
        print(f"T0 : {start_rhte_time - start_time}  PREPROCESSING")
        print(f"T1 : {end_time - start_rhte_time}  ELLIPSE DETECTION")

    if save:
        for ellipse, conf in ellipses:
            print(ellipse, type(ellipse))
            cv2.ellipse(source, ellipse, (0, 255, 0), 2)
            show_conf(source, conf, ellipse)
        cv2.imshow('res', source)
        cv2.waitKey(0)
        output_file = os.path.basename(source_file).split('.')[0] + '_pyout.png'
        output_file = os.path.join(RESULT_DIR, output_file)
        cv2.imwrite(output_file, source)

    data = [len(ellipses), end_time - start_rhte_time]
    return data


detect_ellipses('source/e1.PNG', debug=False, save=True)
