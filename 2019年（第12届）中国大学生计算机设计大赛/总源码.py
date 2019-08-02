import numpy as np
import os
import tellopy
import av
import cv2.cv2 as cv2
import time
import six.moves.urllib as urllib
import sys
import tarfile
import traceback

import tensorflow as tf
import zipfile
import math

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'RFCN'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
count = 0

#加载模型进入内存
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  
#无人机飞控与图像传输
import tellopy
import pygame
import pygame.display
import pygame.key
import pygame.locals
import pygame.font
import datetime
import threading
import socket
from subprocess import Popen, PIPE

prev_flight_data = None
video_player = None
video_recorder = None
font = None
wid = None
date_fmt = '%Y-%m-%d_%H%M%S'
flag = False
sign = False

def toggle_recording(drone, speed):
    global video_recorder
    global date_fmt
    if speed == 0:
        return

    if video_recorder:
        # already recording, so stop
        video_recorder.stdin.close()
        status_print('Video saved to %s' % video_recorder.video_filename)
        video_recorder = None
        return

    # start a new recording
    filename = '/home/qiqi/tensorflow/tellopy/examples/Pictures/tello-%s.mp4' % (datetime.datetime.now().strftime(date_fmt))
    video_recorder = Popen([
        'mencoder', '-', '-vc', 'x264', '-fps', '30', '-ovc', 'copy',
        '-of', 'lavf', '-lavfopts', 'format=mp4',
        # '-ffourcc', 'avc1',
        # '-really-quiet',
        '-o', filename,
    ], stdin=PIPE)
    video_recorder.video_filename = filename
    status_print('Recording video to %s' % filename)

def take_picture(drone, speed):
    drone.take_picture()

def palm_land(drone, speed):
    if speed == 0:
        return
    drone.palm_land()

def toggle_zoom(drone, speed):
    # In "video" mode the drone sends 1280x720 frames.
    # In "photo" mode it sends 2592x1936 (952x720) frames.
    # The video will always be centered in the window.
    # In photo mode, if we keep the window at 1280x720 that gives us ~160px on
    # each side for status information, which is ample.
    # Video mode is harder because then we need to abandon the 16:9 display size
    # if we want to put the HUD next to the video.
    #if speed == 0:
    #    return
    drone.set_video_mode(not drone.zoom)
    pygame.display.get_surface().fill((0,0,0))
    pygame.display.flip()

controls = {
    'takeoff':'takeoff',
    'land':'land',
    'w': 'forward 20',
    's': 'backward',
    'a': 'left',
    'd': 'right',
    'turn_left':'ccw 5',
    'turn_right':'cw 5',
    'flip':'flip r',
    'space': 'up',
    'left shift': 'down',
    'right shift': 'down',
    'q': 'counter_clockwise',
    'e': 'clockwise',
    'go':lambda drone, speed:drone.forward(30),
    # arrow keys for fast turns and altitude adjustments
    'left': lambda drone, speed: drone.counter_clockwise(5),
    'right': lambda drone, speed: drone.clockwise(5),
    'turn_right':lambda drone, speed: drone.clockwise(10),
    'up': lambda drone, speed: drone.up(50),
    'down': lambda drone, speed: drone.down(50),
    'tab': lambda drone, speed: drone.takeoff(),
    'backspace': lambda drone, speed: drone.land(),
    'p': palm_land,
    'r': toggle_recording,
    'z': toggle_zoom,
    'enter': take_picture,
    're': take_picture,
}

class FlightDataDisplay(object):
    # previous flight data value and surface to overlay
    _value = None
    _surface = None
    # function (drone, data) => new value
    # default is lambda drone,data: getattr(data, self._key)
    _update = None
    def __init__(self, key, format, colour=(255,255,255), update=None):
        self._key = key
        self._format = format
        self._colour = colour

        if update:
            self._update = update
        else:
            self._update = lambda drone,data: getattr(data, self._key)

    def update(self, drone, data):
        new_value = self._update(drone, data)
        if self._value != new_value:
            self._value = new_value
            self._surface = font.render(self._format % (new_value,), True, self._colour)
        return self._surface

def flight_data_mode(drone, *args):
    return (drone.zoom and "VID" or "PIC")

def flight_data_recording(*args):
    return (video_recorder and "REC 00:00" or "")  # TODO: duration of recording

def update_hud(hud, drone, flight_data):
    (w,h) = (158,0) # width available on side of screen in 4:3 mode
    blits = []
    for element in hud:
        surface = element.update(drone, flight_data)
        if surface is None:
            continue
        blits += [(surface, (0, h))]
        # w = max(w, surface.get_width())
        h += surface.get_height()
    h += 64  # add some padding
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((0,0,0)) # remove for mplayer overlay mode
    for blit in blits:
        overlay.blit(*blit)
    pygame.display.get_surface().blit(overlay, (0,0))
    pygame.display.update(overlay.get_rect())

def status_print(text):
    pygame.display.set_caption(text)

hud = [
    FlightDataDisplay('height', 'ALT %3d'),
    FlightDataDisplay('ground_speed', 'SPD %3d'),
    FlightDataDisplay('battery_percentage', 'BAT %3d%%'),
    FlightDataDisplay('wifi_strength', 'NET %3d%%'),
    FlightDataDisplay(None, 'CAM %s', update=flight_data_mode),
    FlightDataDisplay(None, '%s', colour=(255, 0, 0), update=flight_data_recording),
]

def flightDataHandler(event, sender, data):
    global prev_flight_data
    text = str(data)
    if prev_flight_data != text:
        update_hud(hud, sender, data)
        prev_flight_data = text

def videoFrameHandler(event, sender, data):
    global video_player
    global video_recorder
    if video_player is None:
        cmd = [ 'mplayer', '-fps', '35', '-really-quiet' ]
        if wid is not None:
            cmd = cmd + [ '-wid', str(wid) ]
        video_player = Popen(cmd + ['-'], stdin=PIPE)

    try:
        video_player.stdin.write(data)
    except IOError as err:
        status_print(str(err))
        video_player = None

    try:
        if video_recorder:
            video_recorder.stdin.write(data)
    except IOError as err:
        status_print(str(err))
        video_recorder = None
        
def handleFileReceived(event, sender, data):
    global date_fmt
    global count
    global flag
    path = 'test_images/image.jpeg'
    with open(path, 'wb') as fd:
        fd.write(data)
    status_print('Saved photo to %s' % path)
    flag = True



PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,'image.jpeg')]
IMAGE_SIZE = (12, 8)


#路径规划算法
def coordinate_init(coor,size):
    # 产生坐标字典
    coordinate_dict = {}
    coordinate_dict[0] = (0, 0)
    for i in range(1, size + 1):
        coordinate_dict[i] = (coor[i-1][0],coor[i-1][1])
    coordinate_dict[size + 1] = (coor[len(coor)-1][0], coor[len(coor)-1][1])
    #print(1)
    return coordinate_dict


def distance_matrix(coordinate_dict, size):  # 生成距离矩阵
    d = np.zeros((size + 2, size + 2))
    for i in range(size + 1):
        for j in range(size + 1):
            if (i == j):
                continue
            if (d[i][j] != 0):
                continue
            x1 = coordinate_dict[i][0]
            y1 = coordinate_dict[i][1]
            x2 = coordinate_dict[j][0]
            y2 = coordinate_dict[j][1]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if (i == 0):
                d[i][j] = d[size + 1][j] = d[j][i] = d[j][size + 1] = distance
            else:
                d[i][j] = d[j][i] = distance
    #print(2)
    return d


def path_length(d_matrix, path_list, size):  # 计算路径长度
    length = 0
    for i in range(size + 1):
        length += d_matrix[path_list[i]][path_list[i + 1]]
    #print(3)
    return length
 

def shuffle(my_list):#起点和终点不能打乱
    temp_list=my_list[1:-1]
    np.random.shuffle(temp_list)
    shuffle_list=my_list[:1]+temp_list+my_list[-1:]
    #print(4)
    return shuffle_list

def product_len_probability(my_list,d_matrix,size,p_num):
    len_list=[]
    pro_list=[]
    path_len_pro=[]
    for path in my_list:
        len_list.append(path_length(d_matrix,path,size))
    max_len=max(len_list)+1e-10
    gen_best_length=min(len_list)
    gen_best_length_index=len_list.index(gen_best_length)
    mask_list=np.ones(p_num)*max_len-np.array(len_list)
    sum_len=np.sum(mask_list)
    for i in range(p_num):
        if(i==0):
            pro_list.append(mask_list[i]/sum_len)
        elif(i==p_num-1):
            pro_list.append(1)
        else:
            pro_list.append(pro_list[i-1]+mask_list[i]/sum_len)
    for i in range(p_num):
        path_len_pro.append([my_list[i],len_list[i],pro_list[i]])
    #print(5)
    return my_list[gen_best_length_index],gen_best_length,path_len_pro
 

def choose_cross(population,p_num):
    jump=np.random.random()
    if jump<population[0][2]:
        return 0
    low=1
    high=p_num
    mid=int((low+high)/2)
    #二分搜索
    while(low<high):
        if jump>population[mid][2]:
            low=mid
            mid=int((low+high)/2)
        elif jump<population[mid-1][2]:
            high=mid
            mid = int((low + high) / 2)
        else:
            #print(6)
            return mid

def veriation(my_list,size):#变异
    ver_1=np.random.randint(1,size+1)
    ver_2=np.random.randint(1,size+1)
    while ver_2==ver_1:#直到ver_2与ver_1不同
        ver_2 = np.random.randint(1, size+1)
        #print('死循环')
    my_list[ver_1],my_list[ver_2]=my_list[ver_2],my_list[ver_1]
    #print(7)
    return my_list

def get_path(coor,size):
    p_num=100#种群个体数量
    gen=100#进化代数
    pm=0.1#变异率
    coordinate_dict=coordinate_init(coor,size)
    if len(coordinate_dict) == 3:
        coor = []
        pa = []
        coor.append((0,0))
        pa.append(0)
        for p in range(1,len(coordinate_dict)):
            coor.append(coordinate_dict[p])
            pa.append(p)
        return coor,pa
    d=distance_matrix(coordinate_dict,size)
    #print(coordinate_dict)
    path_list=list(range(size+2))
    #print(path_list)#打印初始化的路径
    population=[0]*p_num#种群矩阵
    #建立初始种群
    for i in range(p_num):
        population[i]=shuffle(path_list)
    gen_best,gen_best_length,population=product_len_probability(population,d,size,p_num)
    #print(population)#这个列表的每一项中有三项，第一项是路径，第二项是长度，第三项是使用时转盘的概率
    son_list=[0]*p_num
    best_path=gen_best#最好路径初始化
    best_path_length=gen_best_length#最好路径长度初始化
    every_gen_best=[]
    every_gen_best.append(gen_best_length)
    for i in range(gen):#迭代gen代
        son_num=0
        while son_num<p_num:#产生p_num数量子代，杂交与变异
            father_index = choose_cross(population, p_num)
            mother_index = choose_cross(population, p_num)
            father=population[father_index][0]
            mother=population[mother_index][0]
            son1=father.copy()
            son2=mother.copy()
            prduct_set=np.random.randint(1,p_num+1)
            father_cross_set=set(father[1:prduct_set])
            mother_cross_set=set(mother[1:prduct_set])
            cross_complete=1
            for j in range(1,size+1):
                if son1[j] in mother_cross_set:
                    son1[j]=mother[cross_complete]
                    cross_complete+=1
                    if cross_complete>prduct_set:
                        break
            if np.random.random()<pm:#变异
                son1=veriation(son1,size)
            son_list[son_num]=son1
            son_num+=1
            if son_num==p_num: break
            cross_complete=1
            for j in range(1,size+1):
                if son2[j] in father_cross_set:
                    son2[j]=father[cross_complete]
                    cross_complete+=1
                    if cross_complete>prduct_set:
                        break
            if np.random.random()<pm:#变异
                son2=veriation(son2,size)
            son_list[son_num]=son2
            son_num+=1
        gen_best, gen_best_length,population=product_len_probability(son_list,d,size,p_num)
        if(gen_best_length<best_path_length):
            best_path=gen_best
            best_path_length=gen_best_length
        every_gen_best.append(gen_best_length)
    x=[]
    y=[]
    for point in best_path:
        x.append(coordinate_dict[point][0])
        y.append(coordinate_dict[point][1])

    #print(coordinate_dict)
    #print(gen_best)#最后一代最优路径
    #print(gen_best_length)#最后一代最优路径长度
    #print(best_path)#史上最优路径
    #print(best_path_length)#史上最优路径长度
    plt.figure(figsize=IMAGE_SIZE)
    plt.subplot(211)
    plt.plot(every_gen_best)
    plt.subplot(212)
    plt.scatter(x,y)
    plt.plot(x,y)
    plt.savefig('test_images/path.png')
    #plt.grid()
    #plt.show()
    #global flag_path = True
    return coordinate_dict,best_path


#图像目标检测，获取目标的坐标与规划好的路线序列
def ob_image():
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                s_boxes = boxes[scores > 0.5]
                s_classes = classes[scores > 0.5]
                s_scores=scores[scores>0.5]
                size = image_np.shape
                coordinate = [] #坐标列表，用于存储当前画面中多个目标的坐标信息
                if 44 in s_classes:
                    print('识别到瓶子')
                    for i in range(0,len(s_classes)):
                        if s_classes[i] == 44:
                            ymin = s_boxes[i][0]*size[0]
                            xmin = s_boxes[i][1]*size[1]
                            ymax = s_boxes[i][2]*size[0]
                            xmax = s_boxes[i][3]*size[1]
                            xcenter = (xmin + xmax) / 2 #目标中心的横坐标
                            ycenter = (ymin + ymax) / 2 #目标中心的纵坐标
                            frame_xcenter = size[1] / 2 #图像中心的横坐标
                            frame_ycenter = size[0] / 2 #图像中心的纵坐标
                            x = math.sqrt((xcenter-frame_xcenter)*(xcenter-frame_xcenter)+(ycenter-frame_ycenter)*(ycenter-frame_ycenter))
                            distance = distance_to_camera(KNOWN_WIDTH,focalLength,xmax-xmin) * 2.54 #乘2.54是将英寸转换为厘米
                            x = x / focalLength
                            if xcenter < frame_xcenter:
                                x = 0 - x
                            y = math.sqrt(distance*distance-x*x)
                            coordinate.append((100*x,y))
                    print(coordinate)
                    if len(coordinate) >= 2:
                        print('开始路径规划')
                        coordinate_dict,best_path = get_path(coordinate,len(coordinate)-1)
                        print('跑完路径规划')
                        coordin=[]
                        path = []
                        for j in range(1,len(best_path)):
                            coordin.append(coordinate_dict[j])
                            path.append(best_path[j])
                        plt.figure(figsize=IMAGE_SIZE)
                        cv2.imwrite('test_images/image_ob.png', image_np)
                        return coordin,path
                    elif len(coordinate) == 1:
                        plt.figure(figsize=IMAGE_SIZE)
                        cv2.imwrite('test_images/image_ob.png', image_np)
                        return coordinate,None
                    else:
                        return None,None
                else:
                    return None,None



def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

KNOWN_WIDTH = 3.15 #目标宽度，单位为英寸
KNOWN_HEIGHT = 10.81 #目标高度，英寸
KNOWN_DISTANCE = 39.37 #初始化已知的从目标到相机的距离，用来计算焦距
focalLength = 2202 #焦距

#主函数，调用其他各个模块。在主函数内根据规划的路线，控制无人机依次作业
def main():
    pygame.init()
    pygame.display.init()
    pygame.display.set_mode((1280, 720))
    pygame.font.init()
    global font
    font = pygame.font.SysFont("dejavusansmono", 32)
    global count
    global wid
    global sign
    global flag
    if 'window' in pygame.display.get_wm_info():
        wid = pygame.display.get_wm_info()['window']
    print("Tello video WID:", wid)
    drone = tellopy.Tello()
    drone.connect()
    drone.start_video()
    time.sleep(3)
    drone.subscribe(drone.EVENT_FLIGHT_DATA, flightDataHandler)
    drone.subscribe(drone.EVENT_VIDEO_FRAME, videoFrameHandler)
    drone.subscribe(drone.EVENT_FILE_RECEIVED, handleFileReceived)
    speed = 0
    #drone.takeoff()
    #time.sleep(3)
    #drone.down(150)
    #time.sleep(3)
    cnt = 0
    while True:
        if not flag:
            key_handler = controls['enter']
            key_handler(drone,speed)
            time.sleep(3)
            print('已拍照',flag)
            coordinate,best_path = ob_image() #coordinate为各点坐标，best_path存储路线点序列，均不包含原点
            if coordinate is not None and best_path is not None:
                #控制无人机遍历目标
                print('有',len(best_path),'个瓶子')
                degree_y_offset = 0.0
                for i in range(0,len(best_path)):
                    node = best_path[i]
                    x,y = coordinate[node-1]
                    tan = y / x
                    tan = abs(tan)
                    degree = math.degrees(math.atan(tan))
                    if int(degree_y_offset) > 0:
                        drone.counter_clockwise(int(degree_y_offset))
                        time.sleep(2)
                    elif int(degree_y_offset) < 0:
                        drone.clockwise(int(degree_y_offset))
                        time.sleep(2)
                    else:
                        pass
                    if x <= 0:
                        drone.counter_clockwise(int(degree))
                        degree_y_offset = degree_y_offset - degree
                        time.sleep(2)
                    else:
                        drone.clockwise(int(degree))
                        degree_y_offset = degree_y_offset + degree
                        time.sleep(2)
                    distance = math.sqrt(x*x+y*y)
                    drone.forward(int(distance))
                    time.sleep(3)
                    #坐标系变换
                    for j in range(i+1,len(coordinate)):
                        a,b = coordinate[j]
                        a = a - coordinate[i][0]
                        b = b - coordinate[i][1]
                        coordinate[j] = (a,b)
                flag = False
            elif coordinate is not None and best_path is None:
                print('有一个瓶子')
                x,y = coordinate[0]
                tan = y / x
                tan = abs(tan)
                degree = math.degrees(math.atan(tan))
                if x <= 0:
                    drone.counter_clockwise(int(degree))
                    time.sleep(2)
                else:
                    drone.clockwise(int(degree))
                    time.sleep(2)
                distance = math.sqrt(x*x+y*y)
                drone.forward(int(distance))
                time.sleep(3)
                flag = False
            else:
                #当前视角下没有目标
                print('没有瓶子')
                drone.clockwise(60)
                time.sleep(3)
                flag = False
                cnt = cnt + 1
                if cnt >= 6:
                    #旋转360度都没有目标
                    drone.land()
                    break
                
if __name__ == '__main__':
    main()