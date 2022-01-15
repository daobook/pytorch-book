# # import re
# # from pathlib import Path
# # import xml.etree.ElementTree as ET
# # import shutil
# # import yaml
# # from sklearn.model_selection import train_test_split

        

# def gen_labels(annotations_files, new_labels_path, labels):
#     '''将 xml 文件转换为 txt 文件，因为 yolo 希望 txt 文件具有规范化的边界盒。'''
#     infos = []
#     for annotations_file in annotations_files:
#         annotations_file = Path(annotations_file)
#         label_file_name = annotations_file.name.split('.')[0] + '.txt'
#         with open(new_labels_path/label_file_name, 'w') as label_file:
#             root = ET.parse(annotations_file)
#             pic_width = int(root.find('size').findtext('width'))
#             pic_height = int(root.find('size').findtext('height'))
#             info = [pic_width, pic_height]
#             for obj in root.findall('object'):
#                 #box_info = []
#                 class_name = obj.findtext('name')
#                 x_min = int(obj.find('bndbox').findtext('xmin'))
#                 y_min = int(obj.find('bndbox').findtext('ymin'))
#                 x_max = int(obj.find('bndbox').findtext('xmax'))
#                 y_max = int(obj.find('bndbox').findtext('ymax'))
#                 info.append([labels.index(class_name),
#                             x_min, y_min, x_max, y_max])
#                 yolo_format = get_yolo_format(
#                     pic_width, pic_height, x_min, y_min, x_max, y_max)
#                 label_file.write(str(labels.index(class_name)) +
#                                  ' ' + ' '.join(map(str, yolo_format)) + '\n')
#             infos.append(info)
#             label_file.flush()


# # def make_yaml(root, labels, yaml_path):
# #     yaml_data = {
# #         'path': root,
# #         'train': 'train',
# #         'val': 'val',
# #         'nc': len(labels),
# #         'names': labels
# #     }

# #     with open(yaml_path, 'w') as f:
# #         yaml.dump(yaml_data, f, explicit_start=True,
# #                   default_flow_style=False)


# # def gen_dataset(labels, images_files, annotations_files, dataset_path):
# #     new_images_path = dataset_path / 'images/'
# #     new_labels_path = dataset_path / 'labels/'
# #     mkdir(new_images_path)
# #     mkdir(new_labels_path)
# #     make_label(annotations_files, new_labels_path, labels)
# #     # 标签制作
# #     labels_files = glob_file(dataset_path, '*.txt')
# #     copy_file(images_files, new_images_path)
# #     new_images = glob_file(new_images_path, '*.png')
# #     (images_train, labels_train), (images_val, labels_val), (images_test, labels_test) = \
# #         split(new_images, labels_files)
# #     sub_directories = [(images_train, 'train/'),
# #                        (images_val, 'val/'), (images_test, 'test/')]
# #     # 复制图片
# #     for paths, sub_directory in sub_directories:
# #         new_dir = new_images_path/sub_directory
# #         mkdir(new_dir)
# #         copy_file(paths, new_dir)
