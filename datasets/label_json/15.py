import os
import xml.etree.ElementTree as ET
import cv2

path = r'E:\pythonProject1\yolov7\datasets\image\huafen\train\labels'  # 数据集路径
img_path = os.path.join(path, 'img1')
txt_path = os.path.join(path, 'det')
annotations_path = os.path.join(path, 'E:\pythonProject1\yolov7\datasets\image\huafen/train/labels/')  # 生成的xml文件的保持路径
if not os.path.exists(annotations_path):
    os.makedirs(annotations_path)


def write_xml(imgname, img_path, filepath, labeldicts, img_shape, T):
    if T == 0:
        root = ET.Element('Annotation')  # 创建Annotation根节点
        ET.SubElement(root, 'folder').text = str(img_path).split('\\')[-2]
        ET.SubElement(root, 'filename').text = str(imgname)  # 创建filename子节点（无后缀）
        ET.SubElement(root, 'path').text = str(img_path)
        sizes = ET.SubElement(root, 'size')  # 创建size子节点
        ET.SubElement(sizes, 'width').text = str(img_shape[0])
        ET.SubElement(sizes, 'height').text = str(img_shape[1])
        ET.SubElement(sizes, 'depth').text = str(img_shape[2])
        for labeldict in labeldicts:
            objects = ET.SubElement(root, 'object')  # 创建object子节点
            ET.SubElement(objects, 'name').text = labeldict['name']  # 文件中的类别名
            ET.SubElement(objects, 'pose').text = 'Unspecified'
            ET.SubElement(objects, 'truncated').text = '0'
            ET.SubElement(objects, 'difficult').text = '0'
            bndbox = ET.SubElement(objects, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(labeldict['xmin']))
            ET.SubElement(bndbox, 'ymin').text = str(int(labeldict['ymin']))
            ET.SubElement(bndbox, 'xmax').text = str(int(labeldict['xmax']))
            ET.SubElement(bndbox, 'ymax').text = str(int(labeldict['ymax']))
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding='utf-8')
        print("成功写入", filepath.split('\\')[-1])
    else:
        tree = ET.parse(filepath)
        root = tree.getroot()
        for labeldict in labeldicts:
            objects = ET.SubElement(root, 'object')
            ET.SubElement(objects, 'name').text = labeldict['name']
            ET.SubElement(objects, 'pose').text = 'Unspecified'
            ET.SubElement(objects, 'truncated').text = '0'
            ET.SubElement(objects, 'difficult').text = '0'
            bndbox = ET.SubElement(objects, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(labeldict['xmin']))
            ET.SubElement(bndbox, 'ymin').text = str(int(labeldict['ymin']))
            ET.SubElement(bndbox, 'xmax').text = str(int(labeldict['xmax']))
            ET.SubElement(bndbox, 'ymax').text = str(int(labeldict['ymax']))
        tree.write(filepath, encoding='utf-8')


def txt_xml(txt_path, img_path, annotations_path):
    # dict = {'1': "person"}
    files = os.listdir(txt_path)
    pre_img_name = 'datasets\image\huafen/train/labels/'
    for i, name in enumerate(files):
        txtFile = open(txt_path + os.sep + name)
        txtList = txtFile.readlines()
        img_name = name.split('.')[0]
        for j in range(len(txtList)):
            labeldicts = []
            img_id = txtList[j].split(',')[0]
            img = cv2.imread(img_path + os.sep + "%06d" % int(img_id) + '.jpg')  # 根据图片的命名规则来改
            l = float(txtList[j].split(',')[2])
            t = float(txtList[j].split(',')[3])
            w = float(txtList[j].split(',')[4])
            h = float(txtList[j].split(',')[5])
            x1 = l
            x2 = l + w
            y1 = t
            y2 = t + h
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            new_dict = {'name': 'person',
                        'difficult': '0',
                        'xmin': x1,  # 坐标转换
                        'ymin': y1,
                        'xmax': x2,
                        'ymax': y2
                        }
            labeldicts.append(new_dict)
            if img_id != pre_img_name:
                T = 0
                write_xml(img_name, img_path + os.sep + "%06d" % int(img_id) + '.jpg',
                          annotations_path + os.sep + "%06d" % int(img_id) + '.xml', labeldicts, T)
                pre_img_name = img_id
            else:
                T = 1
                write_xml(img_name, img_path + os.sep + "%06d" % int(img_id) + '.jpg',
                          annotations_path + os.sep + "%06d" % int(img_id) + '.xml', labeldicts, img_shape, T)
                pre_img_name = img_id
    print("共写入{}张xml文件".format(len(os.listdir(annotations_path))))
    # print("共写入%d张xml文件" % len(os.listdir(annotations_path)))


txt_xml(img_path=img_path, txt_path=txt_path, annotations_path=annotations_path)