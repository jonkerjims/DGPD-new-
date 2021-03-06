import os
import queue
import re
import threading
import time
from .deeplearn import start
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

# Create your views here.
from dbworm.settings import BASE_DIR

q = queue.Queue()

def index(request):
    return render(request, 'Pridict_tmp/index.html')


def txt_upDownload(request):
    """（100服务器正确）（200服务器错误）"""
    state = 200
    verify_res = 'Server Error.'

    if request.method == "POST":
        file = request.FILES.get('file')
        filename = request.FILES['file'].name
        '''
            存文件
            file_path 存储文件路径
        '''
        if filename:
            dir = os.path.join(os.path.join(BASE_DIR, 'static'), 'userUpload')
            file_path = os.path.join(dir, filename)
            destination = open(file_path,'wb+')
            for chunk in file.chunks():
                # print(str(chunk.decode('utf-8')))
                destination.write(chunk)
            destination.close()

            """
            检测文件格式:
                1、总数不能超过10
                2、格式必须为：(必须有)
                    >P31946|1|training
                    MAAAMKAVTEQGHELSNEERNLLSVAYKNVVGARRSS...
            """
            # 需要被正则优化
            try:
                verify_res, state = verification(file_path)  # verify_res文件验证结果
            except BaseException as err:
                # print(err)
                pass

        data = {
            'state': state,
            'verify_res': verify_res,
            'filename': filename,
        }
        # time.sleep(3)

        return JsonResponse(data=data)


def submit(request):
    """
        总需求分析：
            1、用户输入文件，如果格式检测没有问题。则直接返回邮件接收结果。
            2、使用多线程的方式，让子线程去处理邮件收发，以及错误信息处理。

    """
    """
        返回状态码：
            // 100 数据正确
            // 200 数据label存在问题
            // 300 蛋白序列存在问题
            // 400 线程已满
            // 500 服务器错误
    """
    verify_res = 'We will send the result of processing to your email.'
    state = 100

    if request.method == 'GET':
        email = request.GET.get('email')
        filename = request.GET.get('filename')
        name = request.GET.get('name')
        textarea = request.GET.get('textarea')
        File_path = ''

        # print(email,filename,name,textarea)
        '''
            需求分析：
                1、先判断是否有文件上传，如果有则使用上传的文件，否则处理textarea文件
                2、判断textarea输入格式是否正确。
                
        '''
        # 第一步
        if filename == '' or filename == None:
            dir = os.path.join(os.path.join(BASE_DIR, 'static'), 'userUpload','textarea')
            file_path = os.path.join(dir, "Fasta.fa")
            with open(file_path,'w',encoding='utf-8') as f:
                f.write(format_file(textarea))
            # 第二步
            try:
                verify_res, state = verification(file_path) # verify_res文件验证结果
                File_path = file_path
            except BaseException as err:
                # print(err)
                pass
        else:
            dir = os.path.join(os.path.join(BASE_DIR, 'static'), 'userUpload')
            File_path = os.path.join(dir, filename)

        """
            如果文件没有问题：
                1、创建一个队列，把姓名、邮箱、文件路径传到队列里
                2、创建一个函数 函数里开启线程，从队列里拿去数据
                3、限制总提交次数3次
                4、发邮件
        """
        if state == 100:

            if q.qsize()<10:
                item = [email,name,File_path]
                q.put(item)
                t = threading.Thread(target=start.pro_entry,args=(email,name,File_path,q),name='deeplearn')
                # t = threading.Thread(target=deeplearn,args=(email,name,File_path,q),name='deeplearn')
                t.start()
            else:
                verify_res ,state = 'The current number of submissions has reached the limit, please try again later!', 400

        data = {
            'state': state,
            'verify_res': verify_res,
            'queue_size': q.qsize()
        }
        # time.sleep(3)

        return JsonResponse(data=data)


# 文件格式验证
def verification(file_path):
    verify_res, state = 'We will send the prediction result to your email. ', 100

    with open(file_path, 'r',encoding='utf-8') as f:
        records = f.read()
        # print(1)
        # 检测文件大小，以及是否存在label
        count = len(open(file_path, 'r',encoding='utf-8').readlines())
        if count > 200:
            verify_res, state = 'Only 200 rows of data are allowed to be uploaded at a time.', 200
        else:
            if re.search('>', records) == None:
                verify_res, state = 'There does not seem to be ">" at the beginning of each sequence. You can refer to the sample.', 200
            else:
                """
                    此处必须重新打开文件，因为上一次打开的文件已经失效
                """
                with open(file_path, 'r',encoding='utf-8') as h:
                    for line in h:
                        line = line.replace('\n','')
                        if '>' in line:
                            geneName = line.split('>')[1]
                            if geneName:
                                if re.findall(r'''[\*"/:?\\|<>”’\']''', geneName):
                                    verify_res, state = 'Protein name cannot contain special symbols. (*, ",/, :,? , \ | <, > .)', 200
                                    break
                            else:
                                verify_res, state = 'Please fill in the correct protein ID after ">".', 200
                                break
                            # content_1 = line.split('>')[1]
                            # if str.count(content_1,'|') == 2:
                            #     content_2 = content_1.split('|')[1]
                            #     content_3 = content_1.split('|')[2]
                            #     if (content_2 != '' and content_2 != None) and (content_3 != '' and content_3 != None):
                            #         pass
                            #     else:
                            #         verify_res, state = 'The values before and after "|" can`t be empty.(Please refer to the sample.)', 200
                            #         break
                            # else:
                            #     verify_res, state = 'The label seems not have " | ".(Please refer to the sample.)', 200
                            #     break
                        else: # 判断是否是蛋白质序列
                            line = line.replace('\n','')
                            res = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(line).upper())
                            if '-' in res:
                                verify_res, state = 'The protein sequence seems to be wrong. You can refer to the sample.', 300

    return verify_res, state


def format_file(textarea):
    res = ''
    textarea = textarea.split('\n')
    textarea = [i+'\n' for i in textarea if i != '']
    res = res.join(textarea)
    # print(res)
    return res


# 模拟算法耗时
def deeplearn(a,s,d,n):
    print('deeplearn process is starting!')

    # print(t)

    time.sleep(30)
    print(n.get())
    print('deeplearn process is ending!')
    return


