import smtplib
from email.mime.text import MIMEText


def send_email(email,massage,subject,name=None):
    if email!='' and email != None and ('@' in email):
        host = 'smtp.qq.com'
        port = 465
        sender = '809341512@qq.com'
        sender_alias = 'Ahau Bioinformatics laboratory'
        password = 'tyaqvrhwvfsdbdia'
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

def trans_queryset_toJson(List,count):
    dict = {}
    dict['code'] = 0
    dict['msg'] = ''
    dict['count'] = count
    dict['data'] = List
    #print(count,'List===========',List)
    return dict


