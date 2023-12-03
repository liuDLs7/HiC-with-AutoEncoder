"""
将每个染色体的接触矩阵拉伸为一维向量，再将一个细胞中所有染色体的向量拼接在一起，作为数据输入
预处理使每个染色体对其
"""

import os
import re
import time

import numpy as np
from scipy.sparse import csr_matrix


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def get_max_chr_len(root_dir, chr_num):
    # 获取每条染色体的最大长度用于对齐
    max_len = [0] * chr_num
    subdirectories = get_subdirectories(root_dir)

    for subdirectory in subdirectories:
        # 文件夹路径
        folder_path = os.path.join(root_dir, subdirectory)
        # 获取文件夹下的所有文件
        files = os.listdir(folder_path)

        for file_name in files:
            # 根据文件名获取染色体信息
            match = re.search(r'cell_(\d+)_chr([0-9XY]+).txt', file_name)
            cell_number = int(match.group(1))
            chromosome_number = int(match.group(2)) if match.group(2) != 'X' else 23

            file_path = os.path.join(root_dir, subdirectory, file_name)
            # 打开文件
            with open(file_path, 'r') as file:
                # 读取第一行
                first_line = file.readline().strip()
                # 转换为整数
                ngene = int(float(first_line)) + 1
                max_len[chromosome_number - 1] = max(ngene, max_len[chromosome_number - 1])

    return max_len


class Matrix2Arr1D:

    def __init__(self, root_dir: str = 'contact_626_processed',
                 target_dir: str = 'contact_626_vector',
                 chr_num: int = 23,
                 is_write: bool = False,
                 process_pattern: str = 'diag',
                 m: int = 3,  # m=-1代表包含所有信息
                 chr_len: list = None):

        if chr_len is None:
            chr_len = []

        if not chr_len:
            self.chr_len = get_max_chr_len(root_dir, chr_num)
        else:
            assert len(chr_len) == chr_num, print('给出的chr_len数组或chr_num有误!')
            self.chr_len = chr_len

        assert process_pattern in ['diag', 'row'], print('illegal process_pattern')
        assert -1 <= m <= max(self.chr_len), print('m should in range(-1,max_chr_len)')
        print(self.chr_len)

        self.root_dir = root_dir
        self.target_dir = target_dir
        self.chr_num = chr_num

        self.flatten(iswrite=is_write, process_pattern=process_pattern, m=m)

    def flatten(self, iswrite: bool, process_pattern: str, m: int):
        # 获得一个细胞所有染色体拼接后的数据并写入文件
        subdirectories = get_subdirectories(self.root_dir)

        for subdirectory in subdirectories:
            # 文件夹路径
            folder_path = os.path.join(self.root_dir, subdirectory)
            # 获取文件夹下的所有文件
            files = sorted(os.listdir(folder_path))
            # 定义一个字典，用于存储细胞和染色体编号以及对应的文件名
            cell_chromosomes = {}

            for file_name in files:
                # 根据文件名获取染色体信息
                match = re.search(r'cell_(\d+)_chr([0-9XY]+).txt', file_name)
                cell_number = int(match.group(1))
                chromosome_number = int(match.group(2)) if match.group(2) != 'X' else 23

                # 将细胞和染色体编号及文件名存储到字典中
                if cell_number not in cell_chromosomes:
                    cell_chromosomes[cell_number] = {}
                cell_chromosomes[cell_number][chromosome_number] = file_name

            for cell_number in sorted(cell_chromosomes):
                # 处理后的数据将保存到target_file中（以二进制形式.npy文件保存)
                # 每个细胞信息保存到一个文件中
                target_file = os.path.join(self.target_dir, subdirectory, 'cell_' + str(cell_number) + '.npy')
                # 用于拼接一个细胞的所有染色体
                cell_array = []
                for chromosome_number in sorted(cell_chromosomes[cell_number]):
                    file_name = cell_chromosomes[cell_number][chromosome_number]
                    file_path = os.path.join(self.root_dir, subdirectory, file_name)

                    # 打开文件
                    D = np.loadtxt(file_path, skiprows=1)
                    ngene = self.chr_len[chromosome_number - 1]
                    A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()

                    if process_pattern == 'row':
                        # 按行取
                        # 获取上三角矩阵的索引
                        if m != -1:
                            A = A[:m, :]
                        # 拉伸为一维向量
                        indices = np.triu_indices_from(A)
                        B = A[indices].flatten()

                    elif process_pattern == 'diag':
                        # 按对角线取，只取靠近主对角线的m条（含主对角线)
                        if m != -1:
                            upper_diags = [np.diagonal(A, offset=i) for i in range(0, m)]
                        else:
                            upper_diags = [np.diagonal(A, offset=i) for i in range(0, A.shape[0])]
                        B = np.concatenate(upper_diags)

                    # 将数组B的内容拼接到cell_array中
                    cell_array.append(B)

                cell_array = np.concatenate(cell_array)

                # 将处理后的数据写入文件
                if iswrite:
                    # 可以先用这个创建文件夹
                    os.makedirs(os.path.join(self.target_dir, subdirectory), exist_ok=True)
                    np.save(target_file, cell_array)
                else:
                    print('data_process complete!')


if __name__ == '__main__':
    start = time.time()
    root_dir: str = 'contact_626_processed'
    target_dir: str = 'contact_626_diagvector4'
    chr_len = [250, 244, 198, 192, 181, 171, 160, 147, 142, 136, 135, 134, 116, 108, 103, 91, 82, 79, 60, 63,
               49, 52, 155]
    t = Matrix2Arr1D(target_dir=target_dir, is_write=False, m=4)
    print('timing:\t' + str(time.time() - start))
