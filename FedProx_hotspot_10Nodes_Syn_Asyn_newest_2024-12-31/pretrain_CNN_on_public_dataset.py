import os
import errno
import sys
import argparse
import pickle
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from data_utils import load_MNIST_data #从data_utils文件调入函数load_MNIST_data
from Neural_Networks import cnn_2layer_fc_model, cnn_3layer_fc_model#从data_utils文件调入函数load_MNIST_data



def parseArg():
    # 创建ArgumentParser对象，用于处理命令行参数
    parser = argparse.ArgumentParser(description='Train an array of Neural Networks on either ICCAD or CIFAR')
    # 添加一个命令行参数 -conf，用于指定训练的配置文件
    parser.add_argument('-conf', metavar='conf_file', nargs=1,
                        help='the config file for training, \
                        for training on ICCAD, the default conf_file is ./conf/pretrain_ICCAD.json, \
                        for training on CIFAR, the default conf_file is ./conf/pretrain_CIFAR.json.'
                        )

    # 设置默认的配置文件路径
    conf_file = os.path.abspath("conf/pretrain_ICCAD_conf.json")

    # 如果提供了命令行参数
    if len(sys.argv) > 1:
        # 解析命令行参数
        args = parser.parse_args(sys.argv[1:])
        # 如果提供了 -conf 参数，更新配置文件路径
        if args.conf:
            conf_file = args.conf[0]
    return conf_file


def train_models(models, X_train, y_train, X_test, y_test,
                 is_show=False, save_dir="./", save_names=None,
                 early_stopping=True,
                 min_delta=0.001, patience=3, batch_size=128, epochs=2, is_shuffle=True, verbose=1,
                 ):
    '''
    在同一数据集上训练一组模型。
    我们使用提前终止来加速训练。
    '''
    resulting_val_acc = []  # 用于存储每个模型的验证准确率
    record_result = []  # 用于存储训练过程中的结果

    # 依次训练每个模型
    for n, model in enumerate(models):
        print("Training model ", n)
        # 如果启用提前终止
        if early_stopping:
            # 训练模型，设置验证数据和提前终止回调
            model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=min_delta, patience=patience)],
                      batch_size=batch_size, epochs=epochs, shuffle=is_shuffle, verbose=verbose
                      )
        else:
            # 训练模型，不使用提前终止
            model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      batch_size=batch_size, epochs=epochs, shuffle=is_shuffle, verbose=verbose
                      )

        # 记录模型的最终验证准确率
        resulting_val_acc.append(model.history.history["val_accuracy"][-1])
        # 记录训练过程中的结果，包括训练准确率、验证准确率、训练损失和验证损失
        record_result.append({"train_acc": model.history.history["accuracy"],
                              "val_acc": model.history.history["val_accuracy"],
                              "train_loss": model.history.history["loss"],
                              "val_loss": model.history.history["val_loss"]})

        # 获取保存目录的绝对路径
        save_dir_path = os.path.abspath(save_dir)
        # 创建保存目录
        try:
            os.makedirs(save_dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # 设置保存模型的文件名
        if save_names is None:
            file_name = save_dir + "model_{0}".format(n) + ".h5"
        else:
            file_name = save_dir + save_names[n] + ".h5"
        # 保存模型
        model.save(file_name)

    # 如果启用显示，打印每个模型的验证准确率
    if is_show:
        print("pre-train accuracy: ")
        print(resulting_val_acc)

    # 返回训练过程中记录的结果
    return record_result


# 定义两种模型的字典，后面会用到
models = {"2_layer_CNN": cnn_2layer_fc_model,
          "3_layer_CNN": cnn_3layer_fc_model}

# 主函数
if __name__ == "__main__":

    print("导入包没问题")

    # 解析配置文件路径
    conf_file = parseArg()

    # 读取配置文件内容并解析为字典
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())  # 字典，10个模型的详细信息

    dataset = conf_dict["data_type"]  # 数据集类型，'ICCAD'
    n_classes = conf_dict["n_classes"]  # 类别数，16
    model_config = conf_dict["models"]  # 模型配置列表
    train_params = conf_dict["train_params"]  # 训练参数
    save_dir = conf_dict["save_directory"]  # 模型保存目录
    save_names = conf_dict["save_names"]  # 模型保存名称列表
    early_stopping = conf_dict["early_stopping"]  # 是否启用提前终止

    # 删除配置字典，释放内存
    del conf_dict

    # 加载数据集
    if dataset == "ICCAD":
        input_shape = (144, 32)
        # 存储公共数据集和iccad训练集 是原来iccad训练集的前1/10 用来产生初始模型
        X_train_ICCAD = np.load(
            file="../OriginalHotspotSmallDataset_2Nodes/X_train_ICCAD.npy")
        y_train_ICCAD = np.load(
            file="../OriginalHotspotSmallDataset_2Nodes/y_train_ICCAD.npy")
        # X_train_ConvConsistFedMD_iccad = X_train_ICCAD
        # y_train_ConvConsistFedMD_iccad = y_train_ICCAD
        # X_test_ConvConsistFedMD_iccad = np.load(file="..\OriginalHotspotDataset\X_test_ConvConsistFedMD_iccad.npy")
        # y_test_ConvConsistFedMD_iccad = np.load(file="..\OriginalHotspotDataset\y_test_ConvConsistFedMD_iccad.npy")
        # X_train_ConvConsistFedMD_FAB1 = np.load(file="..\OriginalHotspotDataset\X_train_ConvConsistFedMD_FAB1.npy")
        # y_train_ConvConsistFedMD_FAB1 = np.load(file="..\OriginalHotspotDataset\y_train_ConvConsistFedMD_FAB1.npy")
        # X_test_ConvConsistFedMD_FAB1 = np.load(file="..\OriginalHotspotDataset\X_test_ConvConsistFedMD_FAB1.npy")
        # y_test_ConvConsistFedMD_FAB1 = np.load(file="..\OriginalHotspotDataset\y_test_ConvConsistFedMD_FAB1.npy")
        X_train, y_train, X_test, y_test = X_train_ICCAD, y_train_ICCAD, X_test_ConvConsistFedMD_iccad, y_test_ConvConsistFedMD_iccad
    else:
        print("Unknown dataset. Program stopped.")
        sys.exit()

    # 初始化预训练模型列表
    pretrain_models = []
    for i, item in enumerate(model_config):
        name = item["model_type"]  # 模型类型
        model_params = item["params"]  # 模型参数
        tmp = models[name](n_classes=n_classes, input_shape=input_shape, **model_params)

        print("model {0} : {1}".format(i, save_names[i]))
        print(tmp.summary())
        pretrain_models.append(tmp)  # 添加模型到列表中

    # 训练模型
    record_result = train_models(pretrain_models, X_train, y_train, X_test, y_test,
                                 save_dir=save_dir, save_names=save_names, is_show=True,
                                 early_stopping=early_stopping, **train_params)

    # 保存训练结果
    with open('pretrain_result.pkl', 'wb') as f:
        pickle.dump(record_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('完成')



#以下是源代码
#
# def parseArg():
#     parser = argparse.ArgumentParser(description='Train an array of Neural Networks on either MNIST or CIFAR')#在 MNIST 上训练一系列神经网络
#     parser.add_argument('-conf', metavar='conf_file', nargs=1,
#                         help='the config file for training, \
#                         for training on MNIST, the default conf_file is ./conf/pretrain_MNIST.json, \
#                         for training on CIFAR, the default conf_file is ./conf/pretrain_CIFAR.json.'
#                        )#用于训练的配置文件，对于 MNIST 的训练，默认的 conf_file 是 ./conf/pretrain_MNIST.json, \
#
#     conf_file = os.path.abspath("conf/pretrain_ICCAD_conf.json")
#
#     if len(sys.argv) > 1:
#         args = parser.parse_args(sys.argv[1:])
#         if args.conf:
#             conf_file = args.conf[0]
#     return conf_file
#
#
#
# def train_models(models, X_train, y_train, X_test, y_test,
#                  is_show = False, save_dir = "./", save_names = None,
#                  early_stopping = True,
#                  min_delta = 0.001, patience = 3, batch_size = 128, epochs = 2, is_shuffle=True, verbose = 1,  #1、原代码epochs = 20
#                  ):
#     '''
#     Train an array of models on the same dataset.
#     We use early termination to speed up training.
#     在同一数据集上训练一组模型。
#      我们使用提前终止来加速训练。
#     '''
#     # epochs = 20  # 4、源代码 无
#     resulting_val_acc = []
#     record_result = []
#
#
#
#     for n, model in enumerate(models):
#         print("Training model ", n)
#         if early_stopping:
#             model.fit(X_train, y_train,
#                       validation_data = (X_test, y_test), #2、源代码validation_data = [X_test, y_test]
#                       callbacks=[EarlyStopping(monitor='val_acc', min_delta=min_delta, patience=patience)], #6、val_acc更改为val_accuracy
#                       batch_size = batch_size, epochs = epochs, shuffle=is_shuffle, verbose = verbose
#                      )
#         else:
#             model.fit(X_train, y_train,
#                       validation_data = (X_test, y_test),#3、源代码validation_data = [X_test, y_test]
#                       batch_size = batch_size, epochs = epochs, shuffle=is_shuffle, verbose = verbose
#                      )
#         # print(model.history.history.keys())  #可删除
#         #5、修改val_acc相关
#         resulting_val_acc.append(model.history.history["val_accuracy"][-1])
#         record_result.append({"train_acc": model.history.history["accuracy"],
#                               "val_acc": model.history.history["val_accuracy"],
#                               "train_loss": model.history.history["loss"],
#                               "val_loss": model.history.history["val_loss"]})
#
#
#         save_dir_path = os.path.abspath(save_dir)
#         #make dir
#         try:
#             os.makedirs(save_dir_path)
#         except OSError as e:
#             if e.errno != errno.EEXIST:
#                 raise
#
#         if save_names is None:
#             file_name = save_dir + "model_{0}".format(n) + ".h5"
#         else:
#             file_name = save_dir + save_names[n] + ".h5"
#         model.save(file_name)
#
#     if is_show:
#         print("pre-train accuracy: ")
#         print(resulting_val_acc)
#
#     return record_result
#
#
# models = {"2_layer_CNN": cnn_2layer_fc_model,    #2种模型，后面会用到
#           "3_layer_CNN": cnn_3layer_fc_model}


# if __name__ == "__main__":
#
#     print("导入包没问题")
#
#
#     conf_file =  parseArg() #'D:\\SoftwareInstallation\\Pycharm(学习资料)\\federated-learning-public-code-master\\FedMD-master_hotspot\\conf\\pretrain_ICCAD_conf.json'
#     with open(conf_file, "r") as f:
#         conf_dict = eval(f.read()) #字典，10个模型的详细信息
#     dataset = conf_dict["data_type"] #'MNIST'
#     n_classes = conf_dict["n_classes"] #16
#     model_config = conf_dict["models"] #list:10 [{'model_type': '2_layer_CNN', 'params': {'n1': 128, 'n2': 256, 'dropout_rate': 0.2}}, {'model_type': '2_layer_CNN', 'params': {'n1': 128, 'n2': 384, 'dropout_rate': 0.2}}, {'model_type': '2_layer_CNN', 'params': {'n1': 128, 'n2': 512, 'dropout_rate': 0.2}}, {'model_type': '2_layer_CNN', 'params': {'n1': 256, 'n2': 256, 'dropout_rate': 0.3}}, {'model_type': '2_layer_CNN', 'params': {'n1': 256, 'n2': 512, 'dropout_rate': 0.4}}, {'model_type': '3_layer_CNN', 'params': {'n1': 64, 'n2': 128, 'n3': 256, 'dropout_rate': 0.2}}, {'model_type': '3_layer_CNN', 'params': {'n1': 64, 'n2': 128, 'n3': 192, 'dropout_rate': 0.2}}, {'model_type': '3_layer_CNN', 'params': {'n1': 128, 'n2': 192, 'n3': 256, 'dropout_rate': 0.2}}, {'model_type': '3_layer_CNN', 'params': {'n1': 128, 'n2': 128, 'n3': 128, 'dropout_rate': 0.3}}, {'model_type': '3_layer_CNN', 'params': {'n1': 128, 'n2': 128, 'n3': 192, 'dropout_rate': 0.3}}]
#     train_params = conf_dict["train_params"]#{'min_delta': 0.001, 'patience': 3, 'batch_size': 128, 'epochs': 20, 'is_shuffle': True, 'verbose': 1}
#     save_dir = conf_dict["save_directory"]#'./pretrained_from_MNIST/'
#     save_names = conf_dict["save_names"]#list:10  ['CNN_128_256', 'CNN_128_384', 'CNN_128_512', 'CNN_256_256', 'CNN_256_512', 'CNN_64_128_256', 'CNN_64_128_192', 'CNN_128_192_256', 'CNN_128_128_128', 'CNN_128_128_192']
#     early_stopping = conf_dict["early_stopping"]  #True
#
#
#     del conf_dict
#
#
#     if dataset == "MNIST":
#         input_shape = (144,32)
#         # 存储公共数据集和iccad训练集
#         X_train_ICCAD = np.load(
#             file="..\OriginalHotspotDataset\X_train_ICCAD.npy")
#         y_train_ICCAD = np.load(
#             file="..\OriginalHotspotDataset\y_train_ICCAD.npy")
#         X_train_ConvConsistFedMD_iccad = X_train_ICCAD
#         y_train_ConvConsistFedMD_iccad = y_train_ICCAD
#         # 存储iccad测试集
#         X_test_ConvConsistFedMD_iccad = np.load(
#             file="..\OriginalHotspotDataset\X_test_ConvConsistFedMD_iccad.npy")
#         y_test_ConvConsistFedMD_iccad = np.load(
#             file="..\OriginalHotspotDataset\y_test_ConvConsistFedMD_iccad.npy")
#         # 存储FAB1训练集
#         X_train_ConvConsistFedMD_FAB1 = np.load(
#             file="..\OriginalHotspotDataset\X_train_ConvConsistFedMD_FAB1.npy")
#         y_train_ConvConsistFedMD_FAB1 = np.load(
#             file="..\OriginalHotspotDataset\y_train_ConvConsistFedMD_FAB1.npy")
#         # 存储FAB1测试集
#         X_test_ConvConsistFedMD_FAB1 = np.load(
#             file="..\OriginalHotspotDataset\X_test_ConvConsistFedMD_FAB1.npy")
#         y_test_ConvConsistFedMD_FAB1 = np.load(
#             file="..\OriginalHotspotDataset\y_test_ConvConsistFedMD_FAB1.npy")
#         X_train, y_train, X_test, y_test = X_train_ICCAD,y_train_ICCAD,X_test_ConvConsistFedMD_iccad,y_test_ConvConsistFedMD_iccad
#
#     else:
#         print("Unknown dataset. Program stopped.")
#         sys.exit()
#
#     pretrain_models = []  #预训练的模型，目前为空
#     for i, item in enumerate(model_config):  #i=0~9 item是model_config里面的10个字典
#         name = item["model_type"]  #2_layer_CNN 或者3_layer_CNN
#         model_params = item["params"] #第0个是{'n1': 128, 'n2': 256, 'dropout_rate': 0.2}
#         tmp = models[name](n_classes=n_classes,
#                            input_shape=input_shape,
#                            **model_params)
#
#         print("model {0} : {1}".format(i, save_names[i]))
#         print(tmp.summary())
#         pretrain_models.append(tmp)   #给空模型里面加载模型
# #下面是模型运行
#     record_result = train_models(pretrain_models, X_train, y_train, X_test, y_test,  #pretrain_models:list,存了10个模型的地址；X_train:ndarray(60000,28,28), y_train:(60000,), X_test:(10000,28,28), y_test:(10000,)
#                                  save_dir = save_dir, save_names = save_names, is_show=True,  #save_dir:'./pretrained_from_MNIST/'; save_names:['CNN_128_256', 'CNN_128_384', 'CNN_128_512', 'CNN_256_256', 'CNN_256_512', 'CNN_64_128_256', 'CNN_64_128_192', 'CNN_128_192_256', 'CNN_128_128_128', 'CNN_128_128_192'];
#                                  early_stopping = early_stopping,  #true
#                                  **train_params  #{'min_delta': 0.001, 'patience': 3, 'batch_size': 128, 'epochs': 20, 'is_shuffle': True, 'verbose': 1}
#                                 )
#
#     with open('pretrain_result.pkl', 'wb') as f:
#         pickle.dump(record_result, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print('完成')
