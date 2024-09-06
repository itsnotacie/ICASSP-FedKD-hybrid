import time

import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random
from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


class FedProx():

    def l2_fedprox(self,model, server_model):
        """计算FedProx的正则项"""
        fedprox_reg = 0.0
        for w, w_t in zip(model.trainable_weights, server_model.trainable_weights):
            fedprox_reg += tf.reduce_sum(tf.square(w - w_t))
        return fedprox_reg


    def Get_Average(self, list):
        sum = 0
        for item in list:
            sum += item
        return sum / len(list)

    # parties, 里面存的模型。N_alignment,=5000需要对齐的公共数据集数量。N_rounds, =13总的循环轮数。N_logits_matching_round=1，每一轮后都进行逻辑匹配。N_private_training_round=10，私有数据训练10轮。
    def __init__(self, parties, public_dataset,
                 private_data, total_private_data,
                 private_test_data, private_test_data_FAB1, N_alignment,
                 N_rounds,
                 N_logits_matching_round, logits_matching_batchsize,
                 #默认使用同步策略
                 N_private_training_round, private_training_batchsize, asynchronousRate=1
                 ,fedprox_mu=0.01,N_private_learning_rate=1e-3):

        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.private_test_data_FAB1 = private_test_data_FAB1
        self.N_alignment = N_alignment

        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize

        self.collaborative_parties = []  # 收集了10个训练好的模型和权重，以及这10个模型去掉顶层的模型
        self.init_result = []  # 目前不知道是什么

        self.asynchronousRate = asynchronousRate
        self.fedprox_mu=fedprox_mu  #fedProx的关键参数
        self.N_private_learning_rate=N_private_learning_rate

        self.server_model=None #聚合的中心模型

        print("start model initialization: ")
        for i in range(self.N_parties):  # 10个模型的初始化
            print("model ", i)
            model_A_twin = None
            model_A_twin = clone_model(parties[i])
            model_A_twin.set_weights(parties[i].get_weights())
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                                 loss="sparse_categorical_crossentropy",
                                 metrics=["accuracy"])

            print("无预训练start full stack training ... ")
            # model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
            #                  batch_size=32, epochs=25, shuffle=True, verbose=0,
            #                  validation_data=(private_test_data["X"], private_test_data["y"]),
            #                  # 1、源代码validation_data = [private_test_data["X"], private_test_data["y"]],
            #                  callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5)]
            #                  # 6、val_acc更改为val_accuracy
            #                  )

            print("无预训练full stack training done")


            model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")  # model_A是model_A_twin去掉softmax 激活的模型

            self.collaborative_parties.append({"model_logits": model_A,
                                               "model_classifier": model_A_twin,
                                               "model_weights": model_A_twin.get_weights()})
            # 3、修改val_acc和acc相关
            # self.init_result.append({"val_acc": model_A_twin.history.history['val_accuracy'],
            #                          "train_acc": model_A_twin.history.history['accuracy'],
            #                          "val_loss": model_A_twin.history.history['val_loss'],
            #                          "train_loss": model_A_twin.history.history['loss'],
            #                          })

            # print()
            del model_A, model_A_twin
        # END FOR LOOP

        # print("calculate the theoretical upper bounds for participants: ")
        #
        self.upper_bounds = []
        self.pooled_train_result = []


    def collaborative_training(self):  # 开始联邦学习，异步从这里改造
        acc_iccad_allRounds = []  # 收集iccad所有轮次的平均精度
        acc_FABindus_allRounds = []  # 收集FABindus所有轮次的平均精度

        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)

            print("这是第",r,"轮,round： ", r)
            start_timeRound = time.time()

            r += 1
            if r > self.N_rounds:
                break


            # nametodo start
            print("update logits ... ")
            # update logits

            # self.asynchronousRate=0.5 #异步率默认50%
            asynchronousNoteNumber = int(self.N_parties * self.asynchronousRate)  # 异步节点数量
            index_random = random.sample(range(0, self.N_parties), asynchronousNoteNumber) #取值范围为（0，self.N_parties-1）
            # 将模型聚合合在一起
            weights_sum = None
            for index, d in enumerate(self.collaborative_parties):
                # for d in self.collaborative_parties:
                if index in index_random:
                    # 提取模型的权重
                    weights = d["model_weights"]
                    # 累加所有层的权重 和偏置参数
                    if weights_sum is None:
                        weights_sum = [w.copy() for w in weights]
                    else:
                        for i in range(len(weights)):
                            weights_sum[i] += weights[i]
            # 将累加后的卷积层权重除以模型的数量，进行平均
            weights_average = [w / asynchronousNoteNumber for w in weights_sum]
            # self.server_model = weights_average #记录中心聚合模型

            if self.server_model is None:
                # 使用第一个参与方模型进行初始化
                example_model = self.collaborative_parties[0]["model_classifier"]
                self.server_model = clone_model(example_model)
                # 为新克隆的模型构建网络结构
                self.server_model.build(example_model.input_shape)
                self.server_model.set_weights(weights_average)
            else:
                self.server_model.set_weights(weights_average)

            # 将聚合后的参数赋值给每一个模型, prox不追求模型强一致性
            # for d in self.collaborative_parties:
            #     d["model_weights"] = weights_average
            #     d["model_logits"].set_weights(d["model_weights"])
            #     d["model_classifier"].set_weights(d["model_weights"])


            # test performance 测试联邦学习模型性能，在两个测试集上面测试
            print("test performance ... ")
            TPR_sum, FPR_sum, acc_sum = [], [], []
            for index, d in enumerate(self.collaborative_parties):
                # 两个数据集一半一半测试，这里直接测的所有模型精度
                if index < self.N_parties // 2:
                    y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose=0).argmax(axis=1)
                    collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
                    TPR = sum(self.private_test_data["y"] + y_pred == 2) / sum(self.private_test_data["y"])
                    FPR = sum(self.private_test_data["y"] - y_pred == -1) / (
                                len(self.private_test_data["y"]) - sum(self.private_test_data["y"]))
                    print("模型", index, "的acc性能：", collaboration_performance[index][-1])
                    acc_sum.append(collaboration_performance[index][-1])
                    print("模型", index, "的TPR性能：", TPR)
                    TPR_sum.append(TPR)
                    print("模型", index, "的FPR性能：", FPR)
                    FPR_sum.append(FPR)
                    del y_pred
                if index >= self.N_parties // 2:
                    y_pred = d["model_classifier"].predict(self.private_test_data_FAB1["X"], verbose=0).argmax(axis=1)
                    collaboration_performance[index].append(np.mean(self.private_test_data_FAB1["y"] == y_pred))
                    TPR = sum(self.private_test_data_FAB1["y"] + y_pred == 2) / sum(self.private_test_data_FAB1["y"])
                    FPR = sum(self.private_test_data_FAB1["y"] - y_pred == -1) / (
                                len(self.private_test_data_FAB1["y"]) - sum(self.private_test_data_FAB1["y"]))
                    print("模型", index, "的acc性能：", collaboration_performance[index][-1])
                    acc_sum.append(collaboration_performance[index][-1])
                    print("模型", index, "的TPR性能：", TPR)
                    TPR_sum.append(TPR)
                    print("模型", index, "的FPR性能：", FPR)
                    FPR_sum.append(FPR)
                    del y_pred
            print("模型总的acc性能：", acc_sum)
            print("模型总的TPR性能：", TPR_sum)
            print("模型总的FPR性能：", FPR_sum)
            # print("acc_sum[0:self.N_parties//2]：", acc_sum[0:self.N_parties//2])
            acc_iccad_allRounds.append(self.Get_Average(acc_sum[0:self.N_parties // 2]))
            acc_FABindus_allRounds.append(self.Get_Average(acc_sum[self.N_parties // 2:]))
            print("异步率为：",self.asynchronousRate,"时","第", r, '轮的测试结果：')
            # print(acc_sum[0:self.N_parties//2])
            # print(acc_sum[self.N_parties//2:])
            print('acc_iccad_allRounds:', acc_iccad_allRounds)
            print('acc_FABindus_allRounds', acc_FABindus_allRounds)

            # nametodo end

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                if index in index_random:
                    #这里每个节点用自己的私有数据去做训练
                    print("model {0} starting training with private data... ".format(index))
                    weights_to_use = None #定义
                    weights_to_use = d["model_weights"]    #d["model_weights"]是模型训练后的具体数值，模型在推理时使用这些权重进行计算，具体权重值
                    d["model_classifier"].set_weights(weights_to_use)  #模型结构 (model_classifier) 定义了模型的架构，即模型的网络拓扑和层次结构。包含结构和权重
                    # 添加FedProx正则项
                    # 初始化优化器Adam和损失函数CategoricalCrossentropy
                    optimizer = Adam(learning_rate=self.N_private_learning_rate) #learning_rate默认为 e-3
                    # criterion = tf.keras.losses.CategoricalCrossentropy()
                    criterion = SparseCategoricalCrossentropy(from_logits=False)
                    # 开始本地模型的训练，进行指定轮数的私有训练
                    for epoch in range(self.N_private_training_round):
                        # 对于每个批次的数据，进行训练
                        for batch in range(0, len(self.private_data[index]["X"]), self.private_training_batchsize):
                            # 获取当前批次的输入数据X_batch和标签y_batch
                            X_batch = self.private_data[index]["X"][batch:batch + self.private_training_batchsize]
                            y_batch = self.private_data[index]["y"][batch:batch + self.private_training_batchsize]

                            # 使用tf.GradientTape来跟踪计算梯度
                            with tf.GradientTape() as tape:
                                # 前向传播，通过模型d["model_classifier"]获取输出
                                outputs = d["model_classifier"](X_batch, training=True)

                                # 计算分类损失，即实际标签与预测值之间的交叉熵损失
                                loss = criterion(y_batch, outputs)

                                # 计算FedProx正则项，该正则项是当前模型与服务器模型之间的L2距离
                                fedprox_reg = self.l2_fedprox(d["model_classifier"], self.server_model)

                                # 总损失为分类损失加上FedProx正则项乘以正则化系数self.fedprox_mu
                                total_loss = loss + self.fedprox_mu * fedprox_reg

                            # 计算损失函数对模型可训练参数的梯度
                            grads = tape.gradient(total_loss, d["model_classifier"].trainable_weights)

                            # 使用优化器更新模型的可训练参数
                            optimizer.apply_gradients(zip(grads, d["model_classifier"].trainable_weights))

                        # 保存训练后的模型权重
                        d["model_weights"] = d["model_classifier"].get_weights()
                        print("model {0} done private training. \n".format(index))


                # END FOR LOOP
                end_timeRound = time.time()
                # 统计每一轮训练时间
                train_timeRound = end_timeRound-start_timeRound
                print( f"第{r}轮的训练时间是: {end_timeRound-start_timeRound:.6f} 秒")

            if r==1 and self.asynchronousRate==1:
                print("在同步训练的情况下，第一轮每个节点的私有训练结果即为LocalTraining的训练结果：")
                print("test performance ... ")
                TPRLocal_sum, FPRLocal_sum, accLocal_sum = [], [], []
                accLocal_iccad_first_round = []  # 收集iccad第一轮次本地训练的平均精度
                accLocal_FABindus_first_round = []  # 收集FABindus第一轮次本地训练的平均精度
                for index, d in enumerate(self.collaborative_parties):
                    # 两个数据集一半一半测试，这里直接测的所有模型精度
                    if index < self.N_parties // 2:
                        y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose=0).argmax(axis=1)
                        collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
                        TPRLocal = sum(self.private_test_data["y"] + y_pred == 2) / sum(self.private_test_data["y"])
                        FPRLocal = sum(self.private_test_data["y"] - y_pred == -1) / (
                                len(self.private_test_data["y"]) - sum(self.private_test_data["y"]))
                        print("模型", index, "的accLocal性能：", collaboration_performance[index][-1])
                        accLocal_sum.append(collaboration_performance[index][-1])
                        print("模型", index, "的TPRLocal性能：", TPRLocal)
                        TPRLocal_sum.append(TPRLocal)
                        print("模型", index, "的FPRLocal性能：", FPRLocal)
                        FPRLocal_sum.append(FPRLocal)
                        del y_pred
                    if index >= self.N_parties // 2:
                        y_pred = d["model_classifier"].predict(self.private_test_data_FAB1["X"], verbose=0).argmax(
                            axis=1)
                        collaboration_performance[index].append(np.mean(self.private_test_data_FAB1["y"] == y_pred))
                        TPRLocal = sum(self.private_test_data_FAB1["y"] + y_pred == 2) / sum(
                            self.private_test_data_FAB1["y"])
                        FPRLocal = sum(self.private_test_data_FAB1["y"] - y_pred == -1) / (
                                len(self.private_test_data_FAB1["y"]) - sum(self.private_test_data_FAB1["y"]))
                        print("模型", index, "的accLocal性能：", collaboration_performance[index][-1])
                        accLocal_sum.append(collaboration_performance[index][-1])
                        print("模型", index, "的TPRLocal性能：", TPRLocal)
                        TPRLocal_sum.append(TPRLocal)
                        print("模型", index, "的FPRLocal性能：", FPRLocal)
                        FPRLocal_sum.append(FPRLocal)
                        del y_pred
                print("所有本地节点本地训练模型总的accLocal性能：", accLocal_sum)
                print("所有本地节点本地训练模型总的TPRLocal性能：", TPRLocal_sum)
                print("所有本地节点本地训练模型总的FPRLocal性能：", FPRLocal_sum)
                # print("accLocal_sum[0:self.N_parties//2]：", accLocal_sum[0:self.N_parties//2])
                accLocal_iccad_first_round.append(self.Get_Average(accLocal_sum[0:self.N_parties // 2]))
                accLocal_FABindus_first_round.append(self.Get_Average(accLocal_sum[self.N_parties // 2:]))
                print("异步率为：", self.asynchronousRate, "时", "第", r, '轮时每个节点本地训练LocalTraining的测试结果：')
                # print(accLocal_sum[0:self.N_parties//2])
                # print(accLocal_sum[self.N_parties//2:])
                print('accLocal_iccad_first_round:', accLocal_iccad_first_round)
                print('accLocal_FABindus_first_round', accLocal_FABindus_first_round)


        # END WHILE LOOP
        return collaboration_performance


