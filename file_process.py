import os


def get_gnene(args, tdir):
    doc, cellt, celln, c = args
    spath = os.path.join(doc, cellt, 'cell_' + celln + '_chr' + c + '.txt')
    tpath = os.path.join(tdir, cellt, 'cell_' + celln + '_chr' + c + '.txt')

    # 读取输入文件
    with open(spath, 'r') as file:
        lines = file.readlines()

    # 提取前两列数据并找出最大值
    max_value = float('-inf')  # 初始化最大值为负无穷
    for line in lines:
        values = line.split()
        if len(values) >= 2:
            col1, col2 = float(values[0]), float(values[1])
            max_value = max(max_value, col1, col2)

    # 将最大值写入输出文件的第一行
    lines.insert(0, str(int(max_value)) + '\n')

    # 写入输出文件
    with open(tpath, 'w') as file:
        file.writelines(lines)


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


if __name__ == '__main__':
    root = 'contact_626\contact_626'
    tdir = 'contact_626_processed'
    chr_num = 23

    # 获取一级子文件夹名
    subdirectories = get_subdirectories(root)

    cell_counts = []

    for subdirectory in subdirectories:
        # 文件夹路径
        folder_path = os.path.join(root, subdirectory)
        # 获取文件夹下的所有文件
        files = os.listdir(folder_path)
        # 统计文件数量
        cell_count = 0
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                cell_count += 1
        cell_counts.append(int(cell_count / chr_num))

    for i, cell_type in enumerate(subdirectories):
        for celln in range(1, cell_counts[i]):
            for chrn in range(1, chr_num):
                args = [root, cell_type, str(celln), str(chrn)]
                get_gnene(args, tdir)
            args = [root, cell_type, str(celln), 'X']
            get_gnene(args, tdir)
