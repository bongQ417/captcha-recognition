# coding:utf-8

from captcha.image import ImageCaptcha
import numpy as np
import cv2
import tensorflow as tf
import random, os, sys

from flask import request
from flask import Flask
import json

app = Flask(__name__)


class CnnOcr:
    def __init__(self):
        self.epoch_max = 80  # 最大迭代epoch次数
        self.batch_size = 64  # 训练时每个批次参与训练的图像数目，显存不足的可以调小
        self.lr = 1e-3  # 初始学习率
        self.save_epoch = 10  # 每相隔多少个epoch保存一次模型

        # self.im_width = 244
        # self.im_height = 244
        self.im_width = 192
        self.im_height = 96
        self.im_total_num = 6400  # 总共生成的验证码图片数量
        self.train_max_num = self.im_total_num  # 训练时读取的最大图片数目
        self.val_num = 20 * self.batch_size  # 不能大于self.train_max_num  做验证集用
        self.words_num = 4  # 每张验证码图片上的数字个数
        self.words = '123456789abcdefghijklmnopqrstuvwxyz'
        self.label_num = self.words_num * len(self.words)
        self.keep_drop = tf.placeholder(tf.float32)
        self.x = None
        self.y = None

    def captchaOcr(self, img_path):
        """
        验证码识别
        :param img_path:
        :return:
        """
        im = cv2.imread(img_path)
        im = cv2.resize(im, (self.im_width, self.im_height))
        im = [im]
        im = np.array(im, dtype=np.float32)
        im -= 147
        output = self.sess.run(self.max_idx_p, feed_dict={self.x: im, self.keep_drop: 1.})
        ret = ''
        for i in output.tolist()[0]:
            ret = ret + self.words[int(i)]
        return ret

    def test(self, img_path):
        """
        测试接口
        :param img_path:
        :return:
        """
        self.x = tf.placeholder(tf.float32, [None, self.im_height, self.im_width, 3])  # 输入数据
        self.pred = self.cnnNet()
        self.output = tf.nn.sigmoid(self.pred)
        self.predict = tf.reshape(self.pred, [-1, self.words_num, len(self.words)])
        self.max_idx_p = tf.argmax(self.predict, 2)

        saver = tf.train.Saver()
        # tfconfig = tf.ConfigProto(allow_soft_placement=True)
        # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3  # 占用显存的比例
        # self.ses = tf.Session(config=tfconfig)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  # 全局tf变量初始化

        # 加载w,b参数
        saver.restore(self.sess, './model/CnnOcr-6')
        im = cv2.imread(img_path)
        im = cv2.resize(im, (self.im_width, self.im_height))
        im = [im]
        im = np.array(im, dtype=np.float32)
        im -= 147
        output = self.sess.run(self.max_idx_p, feed_dict={self.x: im, self.keep_drop: 1.})
        ret = ''
        for i in output.tolist()[0]:
            ret = ret + self.words[int(i)]
        print(ret)

    def train(self):
        x_train_list, y_train_list, x_val_list, y_val_list = self.getTrainDataset()

        print('开始转换tensor队列')
        x_train_list_tensor = tf.convert_to_tensor(x_train_list, dtype=tf.string)
        y_train_list_tensor = tf.convert_to_tensor(y_train_list, dtype=tf.float32)

        x_val_list_tensor = tf.convert_to_tensor(x_val_list, dtype=tf.string)
        y_val_list_tensor = tf.convert_to_tensor(y_val_list, dtype=tf.float32)

        x_train_queue = tf.train.slice_input_producer(tensor_list=[x_train_list_tensor], shuffle=False)
        y_train_queue = tf.train.slice_input_producer(tensor_list=[y_train_list_tensor], shuffle=False)

        x_val_queue = tf.train.slice_input_producer(tensor_list=[x_val_list_tensor], shuffle=False)
        y_val_queue = tf.train.slice_input_producer(tensor_list=[y_val_list_tensor], shuffle=False)

        train_im, train_label = self.dataset_opt(x_train_queue, y_train_queue)
        train_batch = tf.train.batch(tensors=[train_im, train_label], batch_size=self.batch_size, num_threads=2)

        val_im, val_label = self.dataset_opt(x_val_queue, y_val_queue)
        val_batch = tf.train.batch(tensors=[val_im, val_label], batch_size=self.batch_size, num_threads=2)

        print('开启训练')
        self.learning_rate = tf.placeholder(dtype=tf.float32)  # 动态学习率
        self.x = tf.placeholder(tf.float32, [None, self.im_height, self.im_width, 3])  # 训练数据
        self.y = tf.placeholder(tf.float32, [None, self.label_num])  # 标签
        self.pred = self.cnnNet()
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.predict = tf.reshape(self.pred, [-1, self.words_num, len(self.words)])
        self.max_idx_p = tf.argmax(self.predict, 2)

        self.y_predict = tf.reshape(self.y, [-1, self.words_num, len(self.words)])
        self.max_idx_l = tf.argmax(self.y_predict, 2)

        self.correct_pred = tf.equal(self.max_idx_p, self.max_idx_l)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        with tf.Session() as self.sess:
            # 全局tf变量初始化
            self.sess.run(tf.global_variables_initializer())
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coordinator)

            # 模型保存
            saver = tf.train.Saver()

            batch_max = len(x_train_list) // self.batch_size
            total_step = 1
            for epoch_num in range(self.epoch_max):
                lr = self.lr * (1 - (epoch_num / self.epoch_max) ** 2)  # 动态学习率
                for batch_num in range(batch_max):
                    x_train_tmp, y_train_tmp = self.sess.run(train_batch)
                    # print(x_train_tmp.shape, y_train_tmp.shape)
                    # sys.exit()

                    self.sess.run(self.optimizer,
                                  feed_dict={self.x: x_train_tmp, self.y: y_train_tmp, self.learning_rate: lr,
                                             self.keep_drop: .5})

                    # 输出评价标准
                    if total_step % 50 == 0 or total_step == 1:
                        print()
                        print('epoch:%d/%d batch:%d/%d step:%d lr:%.10f' % (
                            (epoch_num + 1), self.epoch_max, (batch_num + 1), batch_max, total_step, lr))

                        # 输出训练集评价
                        train_loss, train_acc = self.sess.run([self.loss, self.accuracy],
                                                              feed_dict={self.x: x_train_tmp, self.y: y_train_tmp,
                                                                         self.keep_drop: 1.})
                        print('train_loss:%.10f  train_acc:%.10f' % (np.mean(train_loss), train_acc))

                        # 输出验证集评价
                        val_loss_list, val_acc_list = [], []
                        for i in range(int(self.val_num / self.batch_size)):
                            x_val_tmp, y_val_tmp = self.sess.run(val_batch)
                            val_loss, val_acc = self.sess.run([self.loss, self.accuracy],
                                                              feed_dict={self.x: x_val_tmp, self.y: y_val_tmp,
                                                                         self.keep_drop: 1.})
                            val_loss_list.append(np.mean(val_loss))
                            val_acc_list.append(np.mean(val_acc))
                        print('  val_loss:%.10f    val_acc:%.10f' % (np.mean(val_loss), np.mean(val_acc)))

                    total_step += 1

                # 保存模型
                if (epoch_num + 1) % self.save_epoch == 0:
                    print('正在保存模型：')
                    saver.save(self.sess, './model/CnnOcr', global_step=(epoch_num + 1))
            coordinator.request_stop()
            coordinator.join(threads)

    def cnnNet(self):
        """
        cnn网络
        :return:
        """
        weight = {
            # 输入 batch_size*224*224*3   192  96

            # 第一层
            'wc1_1': tf.get_variable('wc1_1', [3, 3, 3, 64]),  # 卷积 输出：batch_size*224*224*64
            'wc1_2': tf.get_variable('wc1_2', [3, 3, 64, 64]),  # 卷积 输出：batch_size*224*224*64
            # 池化 输出：112*112*64

            # 第二层
            'wc2_1': tf.get_variable('wc2_1', [3, 3, 64, 128]),  # 卷积 输出：batch_size*112*112*128
            'wc2_2': tf.get_variable('wc2_2', [3, 3, 128, 128]),  # 卷积 输出：batch_size*112*112*128
            # 池化 输出：56*56*128

            # 第三层
            'wc3_1': tf.get_variable('wc3_1', [3, 3, 128, 256]),  # 卷积 输出：batch_size*56*56*256
            'wc3_2': tf.get_variable('wc3_2', [3, 3, 256, 256]),  # 卷积 输出：batch_size*56*56*256
            'wc3_3': tf.get_variable('wc3_3', [3, 3, 256, 256]),  # 卷积 输出：batch_size*56*56*256
            # 池化 输出：28*28*256

            # 第四层
            'wc4_1': tf.get_variable('wc4_1', [3, 3, 256, 512]),  # 卷积 输出：batch_size*28*28*512
            'wc4_2': tf.get_variable('wc4_2', [3, 3, 512, 512]),  # 卷积 输出：batch_size*28*28*512
            'wc4_3': tf.get_variable('wc4_3', [3, 3, 512, 512]),  # 卷积 输出：batch_size*28*28*512
            # 池化 输出：14*14*512

            # 第五层
            'wc5_1': tf.get_variable('wc5_1', [3, 3, 512, 512]),  # 卷积 输出：batch_size*14*14*512
            'wc5_2': tf.get_variable('wc5_2', [3, 3, 512, 512]),  # 卷积 输出：batch_size*14*14*512
            'wc5_3': tf.get_variable('wc5_3', [3, 3, 512, 512]),  # 卷积 输出：batch_size*14*14*512
            # 池化 输出：7*7*512

            # 全链接第一层
            # 'wfc_1': tf.get_variable('wfc_1', [7 * 7 * 512, 4096]),
            'wfc_1': tf.get_variable('wfc_1', [6 * 3 * 512, 4096]),

            # 全链接第二层
            'wfc_2': tf.get_variable('wfc_2', [4096, 4096]),

            # 全链接第三层

            'wfc_3': tf.get_variable('wfc_3', [4096, self.label_num]),

        }

        biase = {
            # 第一层
            'bc1_1': tf.get_variable('bc1_1', [64]),
            'bc1_2': tf.get_variable('bc1_2', [64]),

            # 第二层
            'bc2_1': tf.get_variable('bc2_1', [128]),
            'bc2_2': tf.get_variable('bc2_2', [128]),

            # 第三层
            'bc3_1': tf.get_variable('bc3_1', [256]),
            'bc3_2': tf.get_variable('bc3_2', [256]),
            'bc3_3': tf.get_variable('bc3_3', [256]),

            # 第四层
            'bc4_1': tf.get_variable('bc4_1', [512]),
            'bc4_2': tf.get_variable('bc4_2', [512]),
            'bc4_3': tf.get_variable('bc4_3', [512]),

            # 第五层
            'bc5_1': tf.get_variable('bc5_1', [512]),
            'bc5_2': tf.get_variable('bc5_2', [512]),
            'bc5_3': tf.get_variable('bc5_3', [512]),

            # 全链接第一层
            'bfc_1': tf.get_variable('bfc_1', [4096]),

            # 全链接第二层
            'bfc_2': tf.get_variable('bfc_2', [4096]),

            # 全链接第三层
            'bfc_3': tf.get_variable('bfc_3', [self.label_num]),
        }
        # 第一层
        net = tf.nn.conv2d(input=self.x, filter=weight['wc1_1'], strides=[1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc1_1']))  # 加b 然后 激活
        net = tf.nn.conv2d(net, filter=weight['wc1_2'], strides=[1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc1_2']))  # 加b 然后 激活
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 池化

        # 第二层
        net = tf.nn.conv2d(net, weight['wc2_1'], [1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc2_1']))  # 加b 然后 激活
        net = tf.nn.conv2d(net, weight['wc2_2'], [1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc2_2']))  # 加b 然后 激活
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 池化

        # 第三层
        net = tf.nn.conv2d(net, weight['wc3_1'], [1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc3_1']))  # 加b 然后 激活
        net = tf.nn.conv2d(net, weight['wc3_2'], [1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc3_2']))  # 加b 然后 激活
        net = tf.nn.conv2d(net, weight['wc3_3'], [1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc3_3']))  # 加b 然后 激活
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 池化

        # 第四层
        net = tf.nn.conv2d(net, weight['wc4_1'], [1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc4_1']))  # 加b 然后 激活
        net = tf.nn.conv2d(net, weight['wc4_2'], [1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc4_2']))  # 加b 然后 激活
        net = tf.nn.conv2d(net, weight['wc4_3'], [1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc4_3']))  # 加b 然后 激活
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 池化

        # 第五层
        net = tf.nn.conv2d(net, weight['wc5_1'], [1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc5_1']))  # 加b 然后 激活
        net = tf.nn.conv2d(net, weight['wc5_2'], [1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc5_2']))  # 加b 然后 激活
        net = tf.nn.conv2d(net, weight['wc5_3'], [1, 1, 1, 1], padding='SAME')  # 卷积
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc5_3']))  # 加b 然后 激活
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 池化
        print('last-net', net)

        # 拉伸flatten，把多个图片同时分别拉伸成一条向量
        net = tf.reshape(net, shape=[-1, weight['wfc_1'].get_shape()[0]])
        print(weight['wfc_1'].get_shape()[0])
        print('拉伸flatten', net)

        # 全链接层
        # fc第一层
        net = tf.matmul(net, weight['wfc_1']) + biase['bfc_1']
        net = tf.nn.dropout(net, self.keep_drop)
        net = tf.nn.relu(net)
        print('fc第一层', net)

        # fc第二层
        net = tf.matmul(net, weight['wfc_2']) + biase['bfc_2']
        net = tf.nn.dropout(net, self.keep_drop)
        net = tf.nn.relu(net)
        print('fc第二层', net)

        # fc第三层
        net = tf.matmul(net, weight['wfc_3']) + biase['bfc_3']
        print('fc第三层', net)
        return net

    def getTrainDataset(self):
        """
        整理数据集，把图像resize为128*64*3，训练集做成self.im_total_num*128*64*3，把label做成0,1向量形式
        :return:
        """
        train_data_list = os.listdir('./dataset/train/')
        print('共有%d张训练图片， 读取%d张：' % (len(train_data_list), self.train_max_num))
        random.shuffle(train_data_list)  # 打乱顺序

        y_val_list, y_train_list = [], []
        x_val_list = train_data_list[:self.val_num]
        for x_val in x_val_list:
            words_tmp = x_val.split('.')[0].split('_')[1]
            y_val_list.append([1 if _w == w else 0 for w in words_tmp for _w in self.words])

        x_train_list = train_data_list[self.val_num:self.train_max_num]
        for x_train in x_train_list:
            words_tmp = x_train.split('.')[0].split('_')[1]
            y_train_list.append([1 if _w == w else 0 for w in words_tmp for _w in self.words])

        return x_train_list, y_train_list, x_val_list, y_val_list

    def createCaptchaDataset(self):
        """
        生成训练用图片数据集
        :return:
        """
        image = ImageCaptcha(width=self.im_width, height=self.im_height, font_sizes=(56,))
        for i in range(self.im_total_num):
            words_tmp = ''
            for j in range(self.words_num):
                words_tmp = words_tmp + random.choice(self.words)
            print(words_tmp, type(words_tmp))
            im_path = './dataset/train/%d_%s.png' % (i, words_tmp)
            print(im_path)
            image.write(words_tmp, im_path)
        return True

    def dataset_opt(self, x_train_queue, y_train_queue):
        """
        处理图片和标签
        :param queue:
        :return:
        """
        queue = x_train_queue[0]
        contents = tf.read_file('./dataset/train/' + queue)
        im = tf.image.decode_jpeg(contents)
        im = tf.image.resize_images(images=im, size=[self.im_height, self.im_width])
        im = tf.reshape(im, tf.stack([self.im_height, self.im_width, 3]))
        # im -= 147  # 去均值化
        im /= 255  # 将像素处理在0~1之间，加速收敛
        im -= 0.5  # 将像素处理在-0.5~0.5之间
        return im, y_train_queue[0]


if __name__ == '__main__':
    opt_type = sys.argv[1:][0]

    instance = CnnOcr()

    if opt_type == 'create_dataset':
        instance.createCaptchaDataset()
    elif opt_type == 'train':
        instance.train()
    elif opt_type == 'test':
        instance.test('./dataset/test/0_HZDZ.png')
    elif opt_type == 'start':
        # 将session持久化到内存中
        instance.test('./dataset/test/0_HZDZ.png')


        # 启动web服务
        # http://127.0.0.1:5050/captchaOcr?img_path=./dataset/test/2_SYVD.png
        @app.route('/captchaOcr', methods=['GET'])
        def captchaOcr():
            img_path = request.args.to_dict().get('img_path')
            print(img_path)
            ret = instance.captchaOcr(img_path)
            print(ret)
            return json.dumps({'img_path': img_path, 'ocr_ret': ret})


        app.run(host='0.0.0.0', port=5050, debug=False)
