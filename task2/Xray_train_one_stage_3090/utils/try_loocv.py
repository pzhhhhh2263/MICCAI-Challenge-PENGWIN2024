import numpy as np
from sklearn.model_selection import LeaveOneOut
import csv

def readCsv(csvfname):
    # 读取CSV文件，并返回列表格式的数据
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def get_loocv_filelist(csv_file, num_id=1):
    """
    获取留一法结果的API
    :param csv_file: 带有ID、CATE、size的文件
    :param fold: 返回第几折，从1开始
    :return: 指定折的训练集和测试集
    """
    csvlines = readCsv(csv_file)
    header = csvlines[0]
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    patient_id = []
    for file in data_id:
        file_id = file.split("_")[0]
        patient_id.append(int(file_id))

    patient_num = list(set(patient_id))  # 按病人分折

    loo = LeaveOneOut()
    results = []
    for train_index, test_index in loo.split(patient_num):
        if test_index[0]+1 != num_id:
            continue
        train_id = np.array(patient_num)[train_index]
        test_id = np.array(patient_num)[test_index]
        print('train_id:'+str(train_id)+'\nvalid_id:'+str(test_id))

        train_set = []
        test_set = []

        for h5_file in data_id:
            if int(h5_file.split('_')[0]) in test_id:
                test_set.append(h5_file)
            else:
                train_set.append(h5_file)

        results.append([train_set, test_set])

    return results

# 示例使用
csv_file = "/home/siat/pzh/pengwin/two_stage/csv_data/file_names.csv"
fold_index = 1  # 选择第几个病人留1
results =  get_loocv_filelist(csv_file, num_id=fold_index)
train_set = results[0][0]
test_set = results[0][1]
# print(f"Fold {fold_index}")
print(f"Train Set Length: {len(train_set)}")
print(f"Test Set Length: {len(test_set)}")
# print(f"Train Set: {train_set}")
# print(f"Test Set: {test_set}")
