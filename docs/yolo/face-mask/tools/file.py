import re
# from abc import ABC
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import yaml
from sklearn.model_selection import train_test_split


def mkdir(out_dir, overwrite=False):
    out_dir = Path(out_dir)
    is_exist = out_dir.exists()
    if is_exist and overwrite:
        shutil.rmtree(out_dir)
    if not is_exist:
        out_dir.mkdir(parents=True, exist_ok=True)


def change_name(path, new_dir, suffix='.txt'):
    '''修改 path 的 name 的后缀，并将其父目录变为 new_dir
    '''
    path = Path(path)
    new_dir = Path(new_dir)
    name = path.name
    name = new_dir/name.replace(path.suffix, suffix)
    return name.as_posix()


def get_yolo_format(pic_width, pic_height, x_min, y_min, x_max, y_max):
    x_center = (x_max + x_min) / (2 * pic_width)
    y_center = (y_max + y_min) / (2 * pic_height)
    width = (x_max - x_min) / pic_width
    height = (y_max - y_min) / pic_height
    return x_center, y_center, width, height


def labels_iter(file_object, names):
    tree = ET.parse(file_object)
    pic_width = int(tree.find('size').findtext('width'))
    pic_height = int(tree.find('size').findtext('height'))
    info = [pic_width, pic_height]

    for obj in tree.findall('object'):
        class_name = obj.findtext('name')
        x_min = int(obj.find('bndbox').findtext('xmin'))
        y_min = int(obj.find('bndbox').findtext('ymin'))
        x_max = int(obj.find('bndbox').findtext('xmax'))
        y_max = int(obj.find('bndbox').findtext('ymax'))
        info.append([names.index(class_name),
                    x_min, y_min, x_max, y_max])
        yolo_format = get_yolo_format(pic_width, pic_height,
                                      x_min, y_min, x_max, y_max)
        yield [names.index(class_name), *yolo_format]


class PathMeta:
    def __init__(self, root):
        self.root = Path(root)

    def glob(self, mode):
        # 获取全部路径
        paths = [file.as_posix() for file in self.root.rglob(mode)]
        # 排序
        paths.sort(key=lambda e: int(re.sub('[^0-9]', '', e)))
        return paths

    def copy_file(self, path, new_path):
        '''复制 path 到 new_path
        '''
        shutil.copy(path, new_path)

    def copy_files(self, path_group, new_dir, overwrite=False):
        '''复制 path_group 中的文件到 new_dir 下
        :param: paths 是可迭代的路径组对象
        :param: new_dir 是目标目录
        '''
        new_dir = Path(new_dir)
        mkdir(new_dir, overwrite=overwrite)
        for path in path_group:
            path = Path(path)
            self.copy_file(path, new_dir/path.name)


class ObjectPath(PathMeta):
    def __init__(self, root,
                 names,
                 mode_label='*.xml',
                 mode_image='*.png'):
        super().__init__(root)
        self.names = names
        self.mode_label = mode_label
        self.mode_image = mode_image

    @property
    def annotations(self):
        '''获取标签路径'''
        return self.glob(self.mode_label)

    @property
    def images(self):
        '''获取图片路径'''
        return self.glob(self.mode_image)

    def split(self, show_count=5, val_size=0.2):
        '''划分数据集'''
        assert show_count > 0, "保证至少有一张作为测试"
        self.images_train, images_else, self.labels_train, labels_else = \
            train_test_split(self.images, self.annotations, test_size=val_size)
        self.images_val, self.images_test, self.labels_val, self.labels_test = \
            train_test_split(images_else, labels_else,
                             test_size=show_count / len(images_else))

    def copy_images(self, new_dir):
        new_dir = Path(new_dir)
        mkdir(new_dir, overwrite=True)
        self.copy_files(self.images_train, new_dir/'train')
        self.copy_files(self.images_val, new_dir/'val')
        self.copy_files(self.images_test, new_dir/'test')

    def copy_labels(self, new_dir):
        new_dir = Path(new_dir)
        mkdir(new_dir, overwrite=True)
        self.copy_files(self.labels_train, new_dir/'train')
        self.copy_files(self.labels_val, new_dir/'val')
        self.copy_files(self.labels_test, new_dir/'test')

    def write_label(self, label_path, new_dir):
        with open(label_path) as fp:
            labels = [' '.join(map(str, label))
                      for label in labels_iter(fp, self.names)]
        labels = '\n'.join(labels)
        label_path = change_name(label_path, new_dir)
        label_path = Path(label_path)
        mkdir(label_path.parent)
        with open(label_path, 'w') as fp:
            fp.write(labels)

    def _write_labels(self, label_paths, new_dir):
        for label_path in label_paths:
            self.write_label(label_path, new_dir)

    def write_labels(self, new_dir):
        new_dir = Path(new_dir)
        mkdir(new_dir, overwrite=True)
        self._write_labels(self.labels_train, new_dir/'train')
        self._write_labels(self.labels_val, new_dir/'val')
        self._write_labels(self.labels_test, new_dir/'test')


def make_yaml(root, names, config_name):
    root = Path(root).absolute()
    yaml_data = {
        'path': root.as_posix(),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(names),
        'names': names
    }
    yaml_path = root/f'{config_name}.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, explicit_start=True,
                  default_flow_style=False)
