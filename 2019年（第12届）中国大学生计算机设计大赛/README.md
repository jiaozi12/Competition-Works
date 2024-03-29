### 项目简述
基于tello无人机的图像目标检测与路径规划算法

### 设计概要
1.无人机飞控和图像传输模块：用于控制无人机飞行距离、转向角度等行动，传输无人机摄像头拍摄到的实时视频流。

2.图像目标检测模块：用于识别无人机视角中的水面垃圾，并获取各处水面垃圾相对于无人机的坐标信息。

3.路径规划模块：根据图像目标检测模块得到的垃圾坐标进行路径规划，无人机再根据规划好的路线依次完成作业。

各模块关系如下图所示

![image](https://github.com/jiaozi12/Competition-Works/blob/master/2019%E5%B9%B4%EF%BC%88%E7%AC%AC12%E5%B1%8A%EF%BC%89%E4%B8%AD%E5%9B%BD%E5%A4%A7%E5%AD%A6%E7%94%9F%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%AE%BE%E8%AE%A1%E5%A4%A7%E8%B5%9B/Image/%E5%90%84%E6%A8%A1%E5%9D%97%E5%85%B3%E7%B3%BB%E5%9B%BE.png)

工作流程图如下图所示

![image](https://github.com/jiaozi12/Competition-Works/blob/master/2019%E5%B9%B4%EF%BC%88%E7%AC%AC12%E5%B1%8A%EF%BC%89%E4%B8%AD%E5%9B%BD%E5%A4%A7%E5%AD%A6%E7%94%9F%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%AE%BE%E8%AE%A1%E5%A4%A7%E8%B5%9B/Image/%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B%E5%9B%BE.png)

图像目标检测效果如下图所示

![image](https://github.com/jiaozi12/Competition-Works/blob/master/2019%E5%B9%B4%EF%BC%88%E7%AC%AC12%E5%B1%8A%EF%BC%89%E4%B8%AD%E5%9B%BD%E5%A4%A7%E5%AD%A6%E7%94%9F%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%AE%BE%E8%AE%A1%E5%A4%A7%E8%B5%9B/Image/%E6%97%A0%E4%BA%BA%E6%9C%BA%E8%AF%86%E5%88%AB%E6%88%AA%E5%9B%BE.png)

路径规划效果如下图所示

![image](https://github.com/jiaozi12/Competition-Works/blob/master/2019%E5%B9%B4%EF%BC%88%E7%AC%AC12%E5%B1%8A%EF%BC%89%E4%B8%AD%E5%9B%BD%E5%A4%A7%E5%AD%A6%E7%94%9F%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%AE%BE%E8%AE%A1%E5%A4%A7%E8%B5%9B/Image/%E6%97%A0%E4%BA%BA%E6%9C%BA%E8%B7%AF%E5%BE%84%E8%A7%84%E5%88%92%E8%B7%AF%E7%BA%BF%E6%88%AA%E5%9B%BE.png)

### 文件结构
####   —— 路径规划模块源码.py
####   —— 图像目标检测模块源码.py
####   —— 无人机飞控与图像传输模块源码.py
####   —— 总源码.py
####   —— 说明文档.pdf

[完整工程文件可通过百度网盘下载](https://pan.baidu.com/s/1hnzfjEYw-k-a9Je2W-nRIw)    提取码：h498

项目详情以及环境要求参考说明文档.pdf
