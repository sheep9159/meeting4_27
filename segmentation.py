import os
import mne
import data_dir
import re
import pandas as pd


WINDOWS = 150  # 指定窗口长度，用于分割序列，观察在相同的时间跨度上的相似性,这里的是300ms


def get_file_name(file_dir, file_type):
    """
    :遍历指定目录下的所有指定类型的数据文件
    :file_dir: 此目录下包含.eeg原始数据文件，.vhdr文件(含mark)和.vmrk文件
    :file_type: 指定需要找到的文件类型

    :返回
    :file_names: 指定文件的绝对路径
    """

    file_names = []

    for root, dirs, files in os.walk(file_dir, topdown=False):
        for file in files:
            if file_type in file:
                file_names.append(os.path.join(root, file))

    return file_names


def read_mark_txt(file_dir):
    """
    :读取.vmrk文件中mark信息
    :file_dir: 是.vmrk文件的绝对路径

    :返回
    :mark: 点击数字时发送到脑电帽的一个mark标记
    :point: 代表点击数字时已经记录了多少次脑电信号
    """
    mark = []
    point = []
    with open(file_dir, 'r') as f:
        for line in f:
            if line[0:2] == 'MK':
                mark.append(int(line[3:].split(',')[1]))
                point.append(int(line[3:].split(',')[2]))

    return mark, point


set_file_name = get_file_name(data_dir.segmentation_dir+r'\preprocess2', '.set')

channels_index = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 'FC6', 'T7', 'T8', 'C3',
             'Cz', 'C4', 'CP5', 'CP6', 'P7', 'P8', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'PO3', 'PO4', 'O1',
             'O2']

if __name__ == '__main__':

    for index in range(len(set_file_name)):  # 处理被试者某个子试验的数据
        nums = 0
        pattern = re.compile('\w+\.set')
        current_name_index = pattern.findall(set_file_name[index])[0][:-4]
        current_name = current_name_index[:-2]
        name_dir = r'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\meeting4_27' + '\\' + current_name
        if not os.path.exists(name_dir): os.makedirs(name_dir)

        mark, point = read_mark_txt(data_dir.segmentation_dir + r'\section_raw_data' + '\\' + current_name_index + '.vmrk')
        raw = mne.io.read_raw_eeglab(set_file_name[index], preload=True)

        for i in range(len(mark) - 1):  # 依次处理这次子试验下的所有点击动作
            Section_dir = []  # 时间窗口截取的片段存储目录
            raw_eeg = raw.get_data(start=point[i] - 1, stop=point[i + 1])

            duration = len(raw_eeg[0])
            sliding_window = duration // WINDOWS  # 以300ms秒为一个窗口截取数据计算psd，末尾不足300ms的舍去，
            # 由于后续是从点击动作倒着往前取数据的，所以舍去的是被试者刚开始搜索数字的那一段eeg数据

            for seg in range(sliding_window):
                section_dir = name_dir + '\\prepocessed\\{0:0>2}'.format(seg)

                if not os.path.exists(section_dir): os.makedirs(section_dir)
                Section_dir.append(section_dir)

            for seg in range(sliding_window):
                start = seg * WINDOWS
                end = (seg + 1) * WINDOWS if ((seg + 1) * WINDOWS) < duration else duration
                df = pd.DataFrame(raw_eeg[:, -end:-(start + 1)], index=channels_index)  # 先取离点击动作最近的eeg数据
                df.to_csv(Section_dir[seg] + '\\{0:0>3}.csv'.format(nums))
                nums += 1

