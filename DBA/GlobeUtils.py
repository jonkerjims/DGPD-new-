import os
import smtplib
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
from dbworm.settings import BASE_DIR as SYS_BASE_DIR

def send_email(email,massage,subject,name=None):
    if email!='' and email != None and ('@' in email):
        host = 'smtp.qq.com'
        port = 465
        sender = '809341512@qq.com'
        sender_alias = 'Ahau Bioinformatics laboratory'
        password = 'qlovgviabiztbdjg'
        # receiver = 'zhaohuiyouxiang01@163.com'
        # receiver = '809341512@qq.com'
        # receiver = 'zhaohuiyouxiang01@163.com'
        receiver = [email]
        receiver_alias = name

        body = massage
        msg = MIMEText(body, 'html')
        msg['subject'] = subject
        msg['from'] = sender_alias
        msg['to'] = receiver_alias

        s = smtplib.SMTP_SSL(host, port)
        s.login(sender, password)
        s.sendmail(sender, receiver, msg.as_string())
        return 1
    else:
        return 0

def send_email_file(email,massage,attach_file,subject,name=None):
    if email!='' and email != None and ('@' in email):
        host = 'smtp.qq.com'
        port = 465
        sender = '809341512@qq.com'
        sender_alias = 'Ahau Bioinformatics laboratory'
        password = 'qlovgviabiztbdjg'
        receiver = [email]
        receiver_alias = name
        #======邮件内容=====
        body = massage
        #======附件========
        msg = MIMEMultipart() # 创建多媒体文件对象
        msg.attach(MIMEText(body, 'html')) # 添加邮件内容
        # msg.attach(MIMEText('结果详见附件中的csv格式文件', 'plain', 'utf-8'))
        with open(attach_file,'rb') as f:
            file_name = email +'_'+ os.path.basename(attach_file)
            print(file_name)
            # 设置附件格式及名称
            mime = MIMEBase('application','octet-stream',filename=file_name)
            mime.add_header('Content-Disposition', 'attachment', filename=file_name)
            mime.add_header('Content-ID', '<0>')
            mime.add_header('X-Attachment-Id', '0')
            # 将文件读取进来
            mime.set_payload(f.read())
            # 用base64进行编码
            encoders.encode_base64(mime)
            msg.attach(mime)
            # 设置标题 设置发送对象
            msg['subject'] = subject
            msg['from'] = sender_alias
            msg['to'] = receiver_alias
            #=======邮件发送=======
            s = smtplib.SMTP_SSL(host, port)
            s.login(sender, password)
            s.sendmail(sender, receiver, msg.as_string())
            s.quit()
            return 1
    else:
        return 0

def to_csv(result):
    t = time.time()
    hz = int(round(t * 1000))
    filename = str(hz) + '.csv'
    attach_file = os.path.join(SYS_BASE_DIR, 'static', 'file2userEmail', filename)
    df = pd.DataFrame(result)
    df.columns = ["ID", "Score (Non-GRA)", "Score (GRA)"]
    df.to_csv(attach_file, mode="a+", index=False, header=True)
    return attach_file

def trans_queryset_toJson(List,count):
    dict = {}
    dict['code'] = 0
    dict['msg'] = ''
    dict['count'] = count
    dict['data'] = List
    #print(count,'List===========',List)
    return dict


