import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('xml_folder', '', 'Path to xml')
flags.DEFINE_string('type_data', '', 'train or test')
flags.DEFINE_string('csv_out', '', 'out_path')

FLAGS = flags.FLAGS


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    xml_df = xml_to_csv(FLAGS.xml_folder)
    xml_df.to_csv((FLAGS.csv_out + '/' + FLAGS.type_data + '_labels.csv'), index=None)
    print('Successfully converted xml to csv.')


main()
