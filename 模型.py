# coding=utf-8
import os
from multiprocessing import cpu_count
import numpy as np
import paddl
import paddle.fluid as fluid
class classify():
    data_root_path = ""
    dict_path = "C:\Users\DELL\Desktop\下载\Test.txt"
    model_save_dir = ""
    test_data_path = "C:\Users\DELL\Desktop\下载\Test_IDs.txt"
    save_path = ''
    # 获取字典长度
    def get_dict_len(d_path):
        with open(d_path, 'r', encoding='utf-8') as f:
            line = eval(f.readlines()[0])

        return len(line.keys())

    # 1、创建train reader 和 test_reader

    def data_mapper(sample):
        data, label = sample
        data = [int(data) for data in data.split(',')]
        return data, int(label)

    # 创建数据读取器train_reader
    def train_reader(train_data_path):
        def reader():
            with open(train_data_path, 'r') as f:
                lines = f.readlines()
                np.random.shuffle(lines)
                for line in lines:
                    # print (line)
                    data, label = line.split('\t')
                    yield data, label

        return paddle.reader.xmap_readers(classify.data_mapper, reader, cpu_count(), 1024)

    #  创建数据读取器val_reader
    def val_reader(val_data_path):
        def reader():
            with open(val_data_path, 'r') as f:
                lines = f.readlines()
                np.random.shuffle(lines)
                for line in lines:
                    data, label = line.split('\t')
                    yield data, label

        return paddle.reader.xmap_readers(classify.data_mapper, reader, cpu_count(), 1024)
    def test_reader(test_data_path):
        def reader():
            with open(test_data_path, 'r') as f:
                lines = f.readlines()
                # 打乱
                np.random.shuffle(lines)
                for line in lines:
                    data = line
                    yield data.strip(), -1


 # 创建lstm
    def lstm_net(data,
                   dict_dim,
                   class_dim=14,
                   emb_dim=128,
                   hid_dim=128,
                   hid_dim2=96,
                   ):
        """
        Lstm net
        """
        # embedding layer
        emb = fluid.layers.embedding(
            input=data,
            size=[dict_dim, emb_dim])
        fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)
        lstm_h, c = fluid.layers.dynamic_lstm(
            input=fc0, size=hid_dim * 4, is_reverse=False)
        # extract last layer
        lstm_last = fluid.layers.sequence_last_step(input=lstm_h)
        # full connect layer
        fc1 = fluid.layers.fc(input=lstm_last, size=hid_dim2, act='tanh')
        # softmax layer
        prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')
        return prediction

    def train(self):
        # 获取训练数据读取器和测试数据读取器
        train_reader = paddle.batch(reader=self.train_reader(os.path.join(self.data_root_path, "data/data9658/shuffle_Train_IDs.txt")),
            batch_size=128)
        val_reader = paddle.batch(reader=self.val_reader(os.path.join(self.data_root_path, "data/data9658/Val_IDs.txt")),
                                  batch_size=128)
        # 定义输入数据， lod_level不为0指定输入数据为序列数据
        words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        dict_dim = self.get_dict_len(self.dict_path)

        # 获取分类器
        model = self.lstm_net(words, dict_dim)

        # 获取损失函数和准确率
        cost = fluid.layers.cross_entropy(input=model, label=label)
        avg_cost = fluid.layers.mean(cost)
        acc = fluid.layers.accuracy(input=model, label=label)
        # 获取预测程序
        val_program = fluid.default_main_program().clone(for_test=True)
        # 定义优化方法
        optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.0001)
        opt = optimizer.minimize(avg_cost)

        # 创建一个执行器，CPU训练速度比较慢,此处选择gpu还是cpu
        #place = fluid.CPUPlace()
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        # 进行参数初始化
        exe.run(fluid.default_startup_program())

        # 定义数据映射器
        feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

        EPOCH_NUM = 1

        # 开始训练

        for pass_id in range(EPOCH_NUM):
            # 进行训练
            for batch_id, data in enumerate(train_reader()):
                # print(batch_id,len(data))
                train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                                feed=feeder.feed(data),
                                                fetch_list=[avg_cost, acc])
                if batch_id % 100 == 0:
                    print('Pass:%d, Batch:%d, Cost:%0.5f, Acc:%0.5f' % (pass_id, batch_id, train_cost[0], train_acc[0]))
                    # 进行测试
                    val_costs = []
                    val_accs = []
                    for batch_id, data in enumerate(val_reader()):
                        val_cost, val_acc = exe.run(program=val_program,
                                                    feed=feeder.feed(data),
                                                    fetch_list=[avg_cost, acc])
                        val_costs.append(val_cost[0])
                        val_accs.append(val_acc[0])
            # 计算每个epoch平均预测损失在和准确率
            val_cost = (sum(val_costs) / len(val_costs))
            val_acc = (sum(val_accs) / len(val_accs))
            print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, val_cost, val_acc))
            # 保存预测模型
            if not os.path.exists(self.model_save_dir):
                os.makedirs(self.model_save_dir)
            fluid.io.save_inference_model(self.model_save_dir,
                                          feeded_var_names=[words.name],
                                          target_vars=[model],
                                          executor=exe)
        print('训练模型保存完成！')
        self.test(self)
        print('测试输出已生成！')

# 获取数据
    def get_data(self, sentence):
        # 读取数据字典
        with open(self.dict_path, 'r', encoding='utf-8') as f_data:
            dict_txt = eval(f_data.readlines()[0])
        dict_txt = dict(dict_txt)
        # 把字符串数据转换成列表数据
        keys = dict_txt.keys()
        data = []
        for s in sentence:
            # 判断是否存在未知字符
            if not s in keys:
                s = '<unk>'
            data.append(int(dict_txt[s]))
        return data

    def test(self):
        data = []
        # 获取预测数据
        with open(self.test_data_path, 'r', encoding='utf-8') as test_data:
            lines = test_data.readlines()
        print('test start')
        for line in lines:
            tmp_sents = []
            for word in line.strip().split(','):
                tmp_sents.append(int(word))
            data.append(tmp_sents)
        '''
        a=self.get_data(self, 'w我是共产主义接班人！')
        data=[a]
        '''
        print(len(data))
        def load_tensor(data):
            # 获取每句话的单词数量
            base_shape = [[len(c) for c in data]]
            # 创建一个执行器，CPU训练速度比较慢
            #place = fluid.CPUPlace()
            #GPU
            place = fluid.CUDAPlace(0)
            print('loading tensor')
            # 生成预测数据
            tensor_words = fluid.create_lod_tensor(data, base_shape, place)
            #infer_place = fluid.CPUPlace()
            infer_place = fluid.CUDAPlace(0)
            # 执行预测
            infer_exe = fluid.Executor(infer_place)
            # 进行参数初始化
            infer_exe.run(fluid.default_startup_program())
            # 从模型中获取预测程序、输入数据名称列表、分类器
            print('load_model')
            [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=self.model_save_dir,
                                                                                          executor=infer_exe)
            print('getting_ans')
            result = infer_exe.run(program=infer_program,
                                   feed={feeded_var_names[0]: tensor_words},
                                   fetch_list=target_var)
    
            names = ["财经", "彩票", "房产", "股票", "家居", "教育", "科技",
                     "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]
            print('output')
            # 输出结果
            for i in range(len(data)):
                lab = np.argsort(result)[0][i][-1]
                # print('预测结果标签为：%d， 名称为：%s， 概率为：%f' % (lab, names[lab], result[0][i][lab]))
                with open(self.save_path, 'a', encoding='utf-8') as ans:
                    ans.write(names[lab] + "\n")
            ans.close()
        print('loading 1/4 data')
        load_tensor(data[:int(len(data)/4)])
        print('loading 2/4 data')
        load_tensor(data[int(len(data)/4):2*int(len(data)/4)])
        print('loading 3/4 data')
        load_tensor(data[2*int(len(data)/4):3*int(len(data)/4)])
        print('loading 4/4 data')
        load_tensor(data[3*int(len(data)/4):])
        print('测试输出已生成！')
    
if __name__ == "__main__":
    classify.train(classify)
