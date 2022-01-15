import re
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import yaml
from sklearn.model_selection import train_test_split


def mkdir(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)


def glob_file(root, mode):
    files = [file.as_posix() for file in root.rglob(mode)]
    # 排序
    files.sort(key=lambda e: int(re.sub('[^0-9]', '', e)))
    return files


def get_yolo_format(pic_width, pic_height, x_min, y_min, x_max, y_max):
    x_center = (x_max + x_min) / (2 * pic_width)
    y_center = (y_max + y_min) / (2 * pic_height)
    width = (x_max - x_min) / pic_width
    height = (y_max - y_min) / pic_height
    return x_center, y_center, width, height


def make_label(annotations_files, new_labels_path, labels):
    '''将 xml 文件转换为 txt 文件，因为 yolo 希望 txt 文件具有规范化的边界盒。'''

    infos = []
    for annotations_file in annotations_files:
        annotations_file = Path(annotations_file)
        label_file_name = annotations_file.name.split('.')[0] + '.txt'
        with open(new_labels_path/label_file_name, 'w') as label_file:
            root = ET.parse(annotations_file)
            pic_width = int(root.find('size').findtext('width'))
            pic_height = int(root.find('size').findtext('height'))
            info = [pic_width, pic_height]
            for obj in root.findall('object'):
                #box_info = []
                class_name = obj.findtext('name')
                x_min = int(obj.find('bndbox').findtext('xmin'))
                y_min = int(obj.find('bndbox').findtext('ymin'))
                x_max = int(obj.find('bndbox').findtext('xmax'))
                y_max = int(obj.find('bndbox').findtext('ymax'))
                info.append([labels.index(class_name),
                            x_min, y_min, x_max, y_max])
                yolo_format = get_yolo_format(
                    pic_width, pic_height, x_min, y_min, x_max, y_max)
                label_file.write(str(labels.index(class_name)) +
                                 ' ' + ' '.join(map(str, yolo_format)) + '\n')
            infos.append(info)
            label_file.flush()


def copy_file(paths, new_dir):
    for path in paths:
        path = Path(path)
        new_dir = Path(new_dir)
        shutil.copy(path, new_dir/path.name)


def split(images_files, labels_files, show_count=5):
    '''划分数据集'''
    images_train, images_else, labels_train, labels_else = \
        train_test_split(images_files, labels_files, test_size=0.2)
    images_val, images_test, labels_val, labels_test = \
        train_test_split(images_else, labels_else,
                         test_size=show_count / len(images_else))
    return (images_train, labels_train), (images_val, labels_val), (images_test, labels_test)

def write(save_path):
    with open(save_path, 'w') as fp:
        fp.writelines()
        
def make_yaml(root, labels, yaml_path):
    yaml_data = {
        'path': root,
        'train': 'train',
        'val': 'val',
        'nc': len(labels),
        'names': labels
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, explicit_start=True,
                  default_flow_style=False)


def gen_dataset(labels, images_files, annotations_files, dataset_path):
    new_images_path = dataset_path / 'images/'
    new_labels_path = dataset_path / 'labels/'
    mkdir(new_images_path)
    mkdir(new_labels_path)
    make_label(annotations_files, new_labels_path, labels)
    # 标签制作
    labels_files = glob_file(dataset_path, '*.txt')
    copy_file(images_files, new_images_path)
    new_images = glob_file(new_images_path, '*.png')
    (images_train, labels_train), (images_val, labels_val), (images_test, labels_test) = \
        split(new_images, labels_files)
    sub_directories = [(images_train, 'train/'),
                       (images_val, 'val/'), (images_test, 'test/')]
    # 复制图片
    for paths, sub_directory in sub_directories:
        new_dir = new_images_path/sub_directory
        mkdir(new_dir)
        copy_file(paths, new_dir)
