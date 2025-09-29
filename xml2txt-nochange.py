import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
#定义没有的为-1，if等于-1，那么舍弃这个标签


classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33"]
map = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33"]

def getFile_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        print(files)
        for file in files:
            if os.path.splitext(file)[1] in ['.jpg','.JPG']:
                L.append(os.path.splitext(file)[0])  # L.append(os.path.join(root, file))
    return L

def printHello():
    print("hello")


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_id):
    in_file = open(xmlRoot + '\\%s.xml' % (image_id),'rb')

    out_file = open(txtRoot + '\\%s.txt' % (image_id), 'w')  # 生成txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    #print(w)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        if w == 0:
            continue
        cls_id = map[int(cls)]
        #print('cls_id: %d' % int(cls_id))
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
# image_ids_train = open('D:/darknet-master/scripts/VOCdevkit/voc/list.txt').read().strip().split('\',\'')  # list格式只有000000 000001


def convert_singleImg(image_path):
    #根据img_path获取xmlpath和txtpath

    #将image_path里的"图片"换成“标注”和“标注txt
    xmlpath = image_path[:14] + "标注" + image_path[16:]
    xmlpath = xmlpath[:-4] + ".xml"

    txtpath = image_path[:14] + "标注txt" + image_path[16:]  #将标注另存到其他文件夹
    txtpath = txtpath[:-4] + ".txt"

    #txtpath = image_path[:-4] + ".txt"      #将标注存放到图片文件夹

    in_file = open(xmlpath, 'rb')

    out_file = open(txtpath, 'w')  # 生成txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # print(w)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = map[int(cls)]
        # print('cls_id: %d' % int(cls_id))
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__=='__main__':

    print(classes)
    xmlRoot = r'E:\ProData\boss\exp_result\dataset\ori\xml'
    txtRoot = r'E:\ProData\boss\exp_result\dataset\ori\label'
    imageRoot = r'E:\ProData\boss\exp_result\dataset\ori\oriimg'


    print(xmlRoot)

    image_ids_train = getFile_name(imageRoot)
    # image_ids_val = open('/home/*****/darknet/scripts/VOCdevkit/voc/list').read().strip().split()
    # print(image_ids_train)
    print(imageRoot + r'\train.txt')

    list_file_train = open(imageRoot + r'\train.txt', 'w', encoding='utf-8')  # train.txt需要自行创建

    print(image_ids_train)

    for image_id in image_ids_train:
        print(image_id)
        list_file_train.write(imageRoot + '\\%s.jpg\n' % (image_id))  # 这个看看有没问题，因为有的是JPG（大写的）
        convert_annotation(image_id)
    list_file_train.close()  # 只生成训练集，自己根据自己情况决定



