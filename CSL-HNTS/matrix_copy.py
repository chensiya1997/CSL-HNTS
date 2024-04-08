import numpy as np
import torch

def matrix_trans(matrix,change_list):
    change_list=np.abs(change_list)
    change_list=np.argsort(change_list)

    matrix=torch.triu(matrix)
    matrix=(1. - torch.eye(matrix.shape[0])) * matrix
    copy_matrix = torch.zeros(matrix.shape[0], matrix.shape[1])

    for i in range(len(change_list)):
        for j in range(len(change_list)):
            copy_matrix[i][j]=matrix[change_list[i]][change_list[j]]

    return copy_matrix







