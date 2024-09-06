# 在源代码上面改动，改为数字和字母的二分类 数字和字母二分类，在iccad数据集和FAB数据集上，各自5个节点进行联邦学习训练，
import os
import errno
import argparse
import sys
import pickle

import numpy as np
from tensorflow.keras.models import load_model
from data_utils import load_MNIST_data, load_ConvConsistFedMD_data, generate_bal_private_data, \
    generate_partial_data

from FedProx  import  FedProx

import time


def parseArg():
    parser = argparse.ArgumentParser(description='FedAvgAndLocal, a federated learning framework. \
    Participants are training collaboratively. ')  # FedAvgAndLocal，一个联邦学习框架。参与者正在合作培训。
    parser.add_argument('-conf', metavar='conf_file', nargs=1,
                        help='the config file for FedAvgAndLocal.'
                        )  # 用于训练的配置文件，默认的 conf_file 是 ./conf/ConvConsistFedMD_balance_conf.json, \

    conf_file = os.path.abspath("conf/ConvConsistFedMD_balance_conf.json")

    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file


if __name__ == "__main__":
    conf_file = parseArg()
    with open(conf_file, "r", encoding='utf-8') as f:
        conf_dict = eval(f.read())
        N_parties = conf_dict["N_parties"]  # N_parties = {int} 10
        private_classes = conf_dict["private_classes"]  # private_classes = {list: 2} [0,1]
        N_samples_per_class = conf_dict["N_samples_per_class"]  # N_samples_per_class = {int} 3

        N_rounds = conf_dict["N_rounds"]  # N_rounds = {int} 13 总的训练轮次
        N_alignment = conf_dict["N_alignment"]  # N_alignment联合 = {int} 5000
        N_private_training_round = conf_dict["N_private_training_round"]  # N_private_training_round = {int} 10
        private_training_batchsize = conf_dict["private_training_batchsize"]  # private_training_batchsize = {int} 5
        asynchronousRates = conf_dict["asynchronousRates"]  # asynchronousRate 异步更新率，一般为50%
        N_logits_matching_round = conf_dict["N_logits_matching_round"]  # N_logits_matching_round = {int} 1
        logits_matching_batchsize = conf_dict["logits_matching_batchsize"]  # {int} 128
        model_saved_dir = conf_dict["model_saved_dir"]  # {str} './pretrained_from_MNIST/' 储存10个模型
        result_save_dir = conf_dict["result_save_dir"]  # {str} './FConvConsistFedMD_balanced/'
        dataSet_reduct_para = conf_dict["dataSet_reduct_para"] #数据集缩小参数 目前为2
        print("系统配置信息为：",conf_dict["configuration_description"])
        fedprox_mu = conf_dict["fedprox_mu"]#
        N_private_learning_rate = conf_dict["N_private_learning_rate"]#

    del conf_dict, conf_file


    # MNIST数据集加载
    # X_train_ICCAD, y_train_ICCAD, X_test_MNIST, y_test_MNIST \
    # = load_MNIST_data(standarized = True, verbose = True) #X_train_ICCAD = {ndarray: (60000, 28, 28)}；y_train_ICCAD = {ndarray: (60000,)}
    #
    # #做分级化处理
    # X_train_ICCAD = X_train_ICCAD[y_train_ICCAD < 2]
    # y_train_ICCAD=y_train_ICCAD[y_train_ICCAD <2]
    #
    # public_dataset = {"X": X_train_ICCAD, "y": y_train_ICCAD}#public_dataset = {dict: 2}
    # del  X_train_ICCAD, y_train_ICCAD, X_test_MNIST, y_test_MNIST

    #
    # X_train_ConvConsistFedMD, y_train_ConvConsistFedMD, X_test_ConvConsistFedMD, y_test_ConvConsistFedMD \
    # = load_ConvConsistFedMD_data(ConvConsistFedMD_data_dir,
    #                    standarized = True, verbose = True)
    #
    # #做分级化处理
    # X_train_ConvConsistFedMD = X_train_ConvConsistFedMD[y_train_ConvConsistFedMD < 2]
    # y_train_ConvConsistFedMD = y_train_ConvConsistFedMD[y_train_ConvConsistFedMD < 2]
    # #做分级化处理
    # X_test_ConvConsistFedMD = X_test_ConvConsistFedMD[y_test_ConvConsistFedMD < 2]
    # y_test_ConvConsistFedMD = y_test_ConvConsistFedMD[y_test_ConvConsistFedMD < 2]
    #
    #
    # #generate private data
    # #私有数据和所有的私有数据private_data = {list: 10}，其中每一个字典元素00 = {'X'= {ndarray: (18, 28, 28)},'y'= {ndarray: (18,)},'idx' = {ndarray: (18,)}}   total_private_data是字典元素 = {'X'= {ndarray: (180, 28, 28)},'y'= {ndarray: (180,)},'idx' = {ndarray: (180,)}}
    # private_data, total_private_data \
    # = generate_bal_private_data(X_train_ConvConsistFedMD, y_train_ConvConsistFedMD,
    #                         N_parties = N_parties,
    #                         classes_in_use = private_classes,
    #                         N_samples_per_class = N_samples_per_class,
    #                         data_overlap = False) #data_overlap = False数据不可重叠
    #
    #
    #
    # X_tmp, y_tmp = generate_partial_data(X = X_test_ConvConsistFedMD, y= y_test_ConvConsistFedMD,
    #                                      class_in_use = private_classes, verbose = True)
    #
    # private_test_data = {"X": X_tmp, "y": y_tmp}#X_tmp = {ndarray: (4800, 28, 28)}；y_tmp = {ndarray: (4800,)}
    # del X_tmp, y_tmp

    #
    X_train_ICCAD = np.load(
        file="../OriginalHotspotDataset/X_train_ICCAD.npy")
    y_train_ICCAD = np.load(
        file="../OriginalHotspotDataset/y_train_ICCAD.npy")

    # 缩小到原来总体数据集的的 1/dataSet_reduct_para
    dataSet_reduct_para = dataSet_reduct_para  #nametodo可以弄到公共参数配置文件里面取 可以分2份数据出来 dataSet_reduct_para>=2，目前还不使用这个参数，dataSet_reduct_para = 1代表取全部数据
    # iccad训练集，取第一部分
    X_train_ConvConsistFedMD_iccad = X_train_ICCAD[:X_train_ICCAD.shape[0] // dataSet_reduct_para, :, :]
    y_train_ConvConsistFedMD_iccad = y_train_ICCAD[:y_train_ICCAD.shape[0] // dataSet_reduct_para]

    # iccad公共数据
    # ICCADnumsEach表示每一份有多少数据
    ICCADnumsEach = X_train_ICCAD.shape[0] // dataSet_reduct_para
    if (dataSet_reduct_para > 1):
        X_train_ICCAD = X_train_ICCAD[ICCADnumsEach: ICCADnumsEach * 2, :, :]
        y_train_ICCAD = y_train_ICCAD[ICCADnumsEach: ICCADnumsEach * 2]

    # 存储iccad测试集
    X_test_ConvConsistFedMD_iccad = np.load(
        file="../OriginalHotspotDataset/X_test_ConvConsistFedMD_iccad.npy")
    y_test_ConvConsistFedMD_iccad = np.load(
        file="../OriginalHotspotDataset/y_test_ConvConsistFedMD_iccad.npy")

    # 存储FAB1训练集
    X_train_ConvConsistFedMD_FAB1 = np.load(
        file="../OriginalHotspotDataset/X_train_ConvConsistFedMD_FAB1.npy")
    y_train_ConvConsistFedMD_FAB1 = np.load(
        file="../OriginalHotspotDataset/y_train_ConvConsistFedMD_FAB1.npy")

    # FAB1训练集，取第一部分
    X_train_ConvConsistFedMD_FAB1 = X_train_ConvConsistFedMD_FAB1[:X_train_ConvConsistFedMD_FAB1.shape[0] // dataSet_reduct_para, :, :]
    y_train_ConvConsistFedMD_FAB1 = y_train_ConvConsistFedMD_FAB1[:y_train_ConvConsistFedMD_FAB1.shape[0] // dataSet_reduct_para]


    # 存储FAB1测试集
    X_test_ConvConsistFedMD_FAB1 = np.load(
        file="../OriginalHotspotDataset/X_test_ConvConsistFedMD_FAB1.npy")
    y_test_ConvConsistFedMD_FAB1 = np.load(
        file="../OriginalHotspotDataset/y_test_ConvConsistFedMD_FAB1.npy")

    public_dataset = {"X": X_train_ICCAD, "y": y_train_ICCAD}  # public_dataset = {dict: 2}

    # 前一半的节点用于装载iccad数据
    private_data, total_private_data \
        = generate_bal_private_data(X_train_ConvConsistFedMD_iccad, y_train_ConvConsistFedMD_iccad,
                                    N_parties=N_parties // 2,
                                    classes_in_use=private_classes,
                                    N_samples_per_class=N_samples_per_class,
                                    # data_overlap = False数据不可重叠 需要重复采样，因为正样本太少了，正负样本极度不均衡
                                    data_overlap=True)
    # 后一半的节点用于装载FAB1数据
    private_data1, total_private_data1 \
        = generate_bal_private_data(X_train_ConvConsistFedMD_FAB1, y_train_ConvConsistFedMD_FAB1,
                                    N_parties=N_parties // 2,
                                    classes_in_use=private_classes,
                                    N_samples_per_class=N_samples_per_class,
                                    data_overlap=True)  # data_overlap = False数据不可重叠
    private_data = private_data + private_data1
    # total_private_data = total_private_data + total_private_data1 报错可以不要

    # iccad和FAB1的测试集做拼接
    # X_test_ConvConsistFedMD_iccad=np.concatenate([X_test_ConvConsistFedMD_iccad,X_test_ConvConsistFedMD_FAB1])
    # y_test_ConvConsistFedMD_iccad=np.concatenate([y_test_ConvConsistFedMD_iccad,y_test_ConvConsistFedMD_FAB1])

    X_tmp_iccad, y_tmp_iccad = generate_partial_data(X=X_test_ConvConsistFedMD_iccad, y=y_test_ConvConsistFedMD_iccad,
                                                     class_in_use=private_classes, verbose=True)
    private_test_data_iccad = {"X": X_tmp_iccad,
                               "y": y_tmp_iccad}  # X_tmp = {ndarray: (4800, 28, 28)}；y_tmp = {ndarray: (4800,)}
    del X_tmp_iccad, y_tmp_iccad

    X_tmp_FAB1, y_tmp_FAB1 = generate_partial_data(X=X_test_ConvConsistFedMD_FAB1, y=y_test_ConvConsistFedMD_FAB1,
                                                     class_in_use=private_classes, verbose=True)
    private_test_data_FAB1 = {"X": X_tmp_FAB1,
                               "y": y_tmp_FAB1}  # X_tmp = {ndarray: (4800, 28, 28)}；y_tmp = {ndarray: (4800,)}
    del X_tmp_FAB1, y_tmp_FAB1

    for asynchronousRate in asynchronousRates:
        if model_saved_dir is not None:
            parties = []
            dpathICCAD = os.path.abspath(model_saved_dir)+"/ICCAD_Model"
            model_names = os.listdir(dpathICCAD)
            print("按照顺序读取和训练模型，一定要对准model_names",model_names)#一定要对准model_names ['CNN_16_16_32_32_1SorAid_250_2_iccad1.h5', 'CNN_16_16_32_32_250_250_2_FAB1.h5']
            for name in model_names:
                tmp = None
                tmp = load_model(os.path.join(dpathICCAD ,name))
                parties.append(tmp)
            dpathFAB = os.path.abspath(model_saved_dir)+"/FAB_Model"
            model_names = os.listdir(dpathFAB)
            print("按照顺序读取和训练模型，一定要对准model_names",model_names)#一定要对准model_names ['CNN_16_16_32_32_1SorAid_250_2_iccad1.h5', 'CNN_16_16_32_32_250_250_2_FAB1.h5']
            for name in model_names:
                tmp = None
                tmp = load_model(os.path.join(dpathFAB ,name))
                parties.append(tmp)


        fedProx = FedProx(parties,
                      public_dataset=public_dataset,
                      private_data=private_data,
                      total_private_data=total_private_data,
                      private_test_data=private_test_data_iccad,
                      private_test_data_FAB1=private_test_data_FAB1,
                      N_rounds=N_rounds,
                      N_alignment=N_alignment,
                      N_logits_matching_round=N_logits_matching_round,
                      logits_matching_batchsize=logits_matching_batchsize,
                      N_private_training_round=N_private_training_round,
                      private_training_batchsize=private_training_batchsize,
                      asynchronousRate=asynchronousRate,
                      fedprox_mu=fedprox_mu,
                      N_private_learning_rate=N_private_learning_rate)

        initialization_result = fedProx.init_result
        pooled_train_result = fedProx.pooled_train_result

        collaboration_performance = fedProx.collaborative_training()

        # result_save_dir 保存目录的路径
        if result_save_dir is not None:
            save_dir_path = os.path.abspath(result_save_dir)
            # make dir
            try:
                os.makedirs(save_dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        # 将初始化结果保存到 'init_result.pkl' 文件中
        with open(os.path.join(save_dir_path, "asynchronousRate-" + str(asynchronousRate) + 'init_result.pkl'), 'wb') as f:
            # 使用 pickle 模块的 dump 方法将 initialization_result 变量保存到文件中
            # wb 模式表示以二进制写入模式打开文件
            # protocol=pickle.HIGHEST_PROTOCOL 表示使用 pickle 模块的最高协议版本进行序列化
            pickle.dump(initialization_result, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 将训练结果保存到 'pooled_train_result.pkl' 文件中
        with open(os.path.join(save_dir_path, "asynchronousRate-" + str(asynchronousRate) + 'pooled_train_result.pkl'),
                  'wb') as f:
            # 使用 pickle 模块的 dump 方法将 pooled_train_result 变量保存到文件中
            # wb 模式表示以二进制写入模式打开文件
            # protocol=pickle.HIGHEST_PROTOCOL 表示使用 pickle 模块的最高协议版本进行序列化
            pickle.dump(pooled_train_result, f, protocol=pickle.HIGHEST_PROTOCOL)
        # 将协作性能结果保存到 'col_performance.pkl' 文件中
        with open(os.path.join(save_dir_path, "asynchronousRate-" + str(asynchronousRate) + 'col_performance.pkl'), 'wb') as f:
            # 使用 pickle 模块的 dump 方法将 collaboration_performance 变量保存到文件中
            # wb 模式表示以二进制写入模式打开文件
            # protocol=pickle.HIGHEST_PROTOCOL 表示使用 pickle 模块的最高协议版本进行序列化
            pickle.dump(collaboration_performance, f, protocol=pickle.HIGHEST_PROTOCOL)