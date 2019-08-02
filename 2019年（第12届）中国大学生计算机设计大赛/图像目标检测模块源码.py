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
                        