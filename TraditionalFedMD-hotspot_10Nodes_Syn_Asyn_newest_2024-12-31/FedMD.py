import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random
from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer
import time
import datetime

class FedMD():
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
                 N_private_training_round, private_training_batchsize, asynchronousRate=1):
        self.asynchronousRate = asynchronousRate
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
            print("当前时间：",datetime.datetime.now())


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
        # for model in parties:
        #     model_ub = clone_model(model)
        #     model_ub.set_weights(model.get_weights())
        #     model_ub.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3),
        #                      loss = "sparse_categorical_crossentropy",
        #                      metrics = ["acc"])
        #
        #     model_ub.fit(total_private_data["X"], total_private_data["y"],
        #                  batch_size = 32, epochs = 50, shuffle=True, verbose = 1, #7、源代码epochs = 50
        #                  validation_data = (private_test_data["X"], private_test_data["y"]),#2、源代码validation_data = [private_test_data["X"], private_test_data["y"]]
        #                  callbacks=[EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5)])#6、val_acc更改为val_accuracy
        #     # 3、修改val_acc和acc相关
        #     self.upper_bounds.append(model_ub.history.history["val_acc"][-1])
        #     self.pooled_train_result.append({"val_acc": model_ub.history.history["val_acc"],
        #                                      "acc": model_ub.history.history["acc"]})
        #
        #     del model_ub
        # print("the upper bounds are:", self.upper_bounds)

    def collaborative_training(self):  # 开始联邦蒸馏学习，异步从这里改造
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

            print("round ", r)
            print("当前时间：",datetime.datetime.now())
            start_timeRound = time.time()

            # test performance
            print("test performance ... ")
            TPR_sum, FPR_sum, acc_sum = [], [], []
            for index, d in enumerate(self.collaborative_parties):
                # 两个数据集一半一半测试，这里直接测的所有模型精度，就算异步也没有做选择，注意对称
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
            r += 1
            if r > self.N_rounds:
                break
            # nametodo start
            print("update logits ... ")
            # update logits
            logits = 0

            # self.asynchronousRate=0.6 #异步率60%
            asynchronousNoteNumber = int(self.N_parties * self.asynchronousRate)  # 异步节点数量
            # 确保 self.N_parties 是偶数,两类节点异步选取
            assert self.N_parties % 2 == 0, "N_parties must be an even number."
            # 每一半的数量
            half = self.N_parties // 2
            half_asynchronousNoteNumber = asynchronousNoteNumber // 2
            # 前一半随机取值
            first_half_indexes = random.sample(range(0, half), half_asynchronousNoteNumber)
            # 后一半随机取值
            second_half_indexes = random.sample(range(half, self.N_parties), half_asynchronousNoteNumber)
            # 合并结果
            index_random = first_half_indexes + second_half_indexes
            #特殊情况，在总数量中的异步
            if len(index_random)==0:
                index_random = random.sample(range(0, self.N_parties), asynchronousNoteNumber) #取值范围为（0，self.N_parties-1）
            print("异步率为",self.asynchronousRate,"时，第 ", r,"轮选取的没模型是",index_random)


            # 蒸馏
            for index, d in enumerate(self.collaborative_parties):
                # for d in self.collaborative_parties:
                if index in index_random:
                    d["model_logits"].set_weights(d["model_weights"])
                    logits += d["model_logits"].predict(alignment_data["X"], verbose=0)

            logits /= asynchronousNoteNumber

            # nametodo end

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                if index in index_random:
                    print("model {0} starting alignment with public logits... ".format(index))

                    weights_to_use = None
                    weights_to_use = d["model_weights"]
                    d["model_logits"].set_weights(weights_to_use)
                    '''
d["model_logits"]
这是要训练的模型对象。它可能是一个 Keras 模型或者其他类型的模型。
alignment_data["X"]
这是用于训练的输入数据。它可能是一个 NumPy 数组或 TensorFlow 张量。
logits
这是用于训练的目标输出数据。它可能也是一个 NumPy 数组或 TensorFlow 张量。
batch_size = self.logits_matching_batchsize
这个参数指定了每次训练时使用的样本数量。它决定了在一个训练步骤中,模型会同时处理多少个样本。较小的批量大小可能会导致训练更慢,但可能会提高模型的性能。较大的批量大小可以加快训练速度,但可能会影响模型的收敛性。
epochs = self.N_logits_matching_round
这个参数指定了训练的轮数。也就是说,整个数据集将被传递给模型的次数。更多的训练轮数通常意味着模型有更多的机会学习和优化,但也可能导致过拟合。
shuffle=True
这个参数告诉 Keras 在每个训练轮次之前对数据进行随机打乱。这通常有助于防止模型过度拟合特定顺序的数据。
verbose = True
这个参数控制是否在训练过程中打印出进度信息。设置为 True 可以让您更好地监控训练过程。
总的来说,这行代码是在使用 model_logits 模型,通过 alignment_data["X"] 输入数据和 logits 目标数据,进行 N_logits_matching_round 轮次的训练,每轮使用 logits_matching_batchsize 大小的批量,并在训练过程中打印出进度信息。

这种训练方式通常用于模型微调、对抗训练或者知识蒸馏等任务。具体的用途取决于您的问题和模型架构。如果您有任何其他疑问,欢迎继续询问。
'''
                    d["model_logits"].fit(alignment_data["X"], logits,
                                          batch_size=self.logits_matching_batchsize,
                                          epochs=self.N_logits_matching_round,
                                          shuffle=True, verbose=False)  #verbose=False 表示不打印显示进度信息
                    d["model_weights"] = d["model_logits"].get_weights()
                    print("model {0} done alignment".format(index))

                    print("model {0} starting training with private data... ".format(index))
                    weights_to_use = None
                    weights_to_use = d["model_weights"]
                    d["model_classifier"].set_weights(weights_to_use)
                    d["model_classifier"].fit(self.private_data[index]["X"],
                                              self.private_data[index]["y"],
                                              batch_size=self.private_training_batchsize,
                                              epochs=self.N_private_training_round,
                                              shuffle=True, verbose=False)  #verbose=False 表示不打印显示进度信息

                    d["model_weights"] = d["model_classifier"].get_weights()
                    print("model {0} done private training. \n".format(index))
                # END FOR LOOP
                end_timeRound = time.time()
                # 统计每一轮训练时间
                train_timeRound = end_timeRound-start_timeRound
                print( f"第{r}轮的训练时间是: {end_timeRound-start_timeRound:.6f} 秒")


        # END WHILE LOOP
        return collaboration_performance


