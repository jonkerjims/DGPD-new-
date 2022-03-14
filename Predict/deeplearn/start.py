# -*- coding: utf-8 -*-
import queue
import sys
import os
import time

import numpy
import pandas as pd
from numpy import array
from dbworm.settings import BASE_DIR as SYS_BASE_DIR

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SYS_BASE_DIR, r'DBA'))
sys.path.append(os.path.join(BASE_DIR, r'iLearn-master'))

# sys.path.append(r'C:\Users\Shi Haoyuan\Desktop\deeplearn\deeplearn\iLearn-master')

# print(os.path.exists(r'C:\Users\Shi Haoyuan\Desktop\deeplearn\iLearn-master'))
# os.path.exists(r'C:\Users\80934\Desktop\iLearnResultCSVFile\input_file\protein_sequences.txt')
# print(os.path.abspath(__file__))
from iLearn_protein_basic import *
from .GraphConstruction import *
from .model_GCN import *
from GlobeUtils import send_email


def pro_entry(email, name, File_path, q: queue.Queue):
    try:
        t1 = time.time()
        # h=1/0
        header = iLearnStart(File_path)

        # 读取生成特征
        path = os.path.join(BASE_DIR, 'Predict','deeplearn','iLearnResultCSVFile','output_file','feature')
        feature = pd.read_csv(os.path.join(path, r'feature.csv'))
        BasicFeature = pd.read_csv(os.path.join(path, r'data.csv'))
        feature_all = BasicFeature.append(feature)
        pd.DataFrame(feature_all).to_csv(os.path.join(path, r'feature_all.csv'), index=False, header=header,
                                         encoding='utf-8')

        # 读取生成标签
        BasicLabel = pd.read_csv(os.path.join(path, r'protein_label.csv'), names=['ID', '#', 'label'])
        base_num = BasicLabel.shape[0]
        label = pd.read_csv(os.path.join(path, r'label.csv'))
        predict_num = label.shape[0]
        label_all = BasicLabel.append(label)
        id_ = []
        for i in range(1, base_num + predict_num + 1):
            id_.append(i)
        label_all['ID'] = id_
        pd.DataFrame(label_all).to_csv(os.path.join(path, r'label_all.csv'), index=False, encoding='utf-8')

        # 生成图
        construct(predict_num)

        # 图卷积
        result = convolution()
        label = array(label)
        result = array(result)
        result = numpy.column_stack((label[:, 1], result))
        t2 = time.time()

        result = result.tolist()
        print(result)

        ###################################【邮件内容编辑】###################################
        resultNeirong = r'<thead style="width: 100%;"><tr><th class="text-center" style="width: 33.3%">标签</th><th class="text-center" style="width: 33.3%">不是致密颗粒蛋白概率</th><th class="text-center" style="width: 33.3%">是致密颗粒蛋白概率</th></tr></thead><tbody style="width: 100%;">'
        for res in result:
            resultNeirong += r'<tr><td class="text-center" style="width: 33.3%">' + str(res[0]) + '</td><td class="text-center" style="width: 33.3%">' + '%0.3f' % res[1] + '</td><td class="text-center" style="width: 33.3%">' + '%0.3f' % res[2] + '</td></tr>'
        resultNeirong += r'</tbody>'
        mail_massage = '<div style="text-align: center;"><h2>The predictions</h2><table  border="1px solid #ccc" cellspacing="0" cellpadding="0" style="text-align: center;margin:auto;">'
        mail_massage += resultNeirong
        mail_massage += '</table></div>'
        ####################################################################################

        massage = mail_massage +'<br>'+ str('耗时:%s' % (str(int(t2-t1)))) + 's'
        subject = '蛋白质序列预测成功提醒'

        # 检测邮件是否发送失败
        backcall = send_email(email, massage, subject, name)
        print(backcall)
        print('耗时：', int(t2 - t1))
        # 取出管道的内容
        q.get()
        # print(1/0)

    except BaseException as err:
        print(err)
        time.sleep(15)
        massage = '你在DGPD提交的蛋白质序列，由于运算超时或者服务器超符合运转，导致预测失败，请联系管理员809341512@qq.com查看详情！'
        subject = '蛋白质序列预测失败提醒'
        backcall = send_email(email, massage, subject, name)
        print(backcall)
        # 取出管道的内容
        q.get()
        print('算法出错')
