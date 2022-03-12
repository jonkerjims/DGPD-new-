import smtplib
from email.mime.text import MIMEText
import pandas as pd
import os

def trans_Queryset(List):
    Dict = {}
    gene = []
    orga = []
    evidece_level = []
    intron = []
    signal_peptide = []
    tmhmm = []
    pmid = []
    molecular_weight = []
    domains = []
    date_source = []


    for l in List:
        for key in l:
            if key == 'gene_name':
                gene.append(l[key])
            if key == 'organism':
                orga.append(l[key])
            if key == 'evidece_level':
                evidece_level.append(l[key])
            if key == 'intron':
                intron.append(l[key])
            if key == 'signal_peptide':
                signal_peptide.append(l[key])
            if key == 'tmhmm':
                tmhmm.append(l[key])
            if key == 'pmid':
                pmid.append(l[key])
            if key == 'molecular_weight':
                molecular_weight.append(l[key])
            if key == 'domains':
                domains.append(l[key])
            if key == 'date_source':
                date_source.append(l[key])
    if gene:
        Dict["gene_name"] = Unique(gene)
    if orga:
        Dict["organism"] = Unique(orga)
    if evidece_level:
        Dict["evidece_level"] = evidece_level
    if intron:
        Dict["intron"] = intron
    if signal_peptide:
        Dict["signal_peptide"] = signal_peptide
    if tmhmm:
        Dict["tmhmm"] = tmhmm
    if pmid:
        Dict["pmid"] = pmid
    if molecular_weight:
        Dict["molecular_weight"] = molecular_weight
    if domains:
        Dict["domains"] = domains
    if date_source:
        Dict["date_source"] = date_source

    return Dict

def trans_Queryset_toTable(List,count):

    # print(list(List))
    dict = {}
    dict["code"] = 0
    dict["msg"] = ""
    dict["count"] = count
    dict["data"] = list(List)
    return dict

def send_email(emails,massage,subject,name=None):
    email_list = []
    for email in emails:
        if email != '' and email != None and email!='-':
            email_list.append(email)
    #print(email_list)
    host = 'smtp.qq.com'
    port = 465
    sender = '809341512@qq.com'
    sender_alias = 'Ahau Bioinformatics laboratory'
    password = 'qlovgviabiztbdjg'
    # receiver = 'zhaohuiyouxiang01@163.com'
    # receiver = '809341512@qq.com'
    # receiver = 'zhaohuiyouxiang01@163.com'

    #receiver = email_list#全体成员接收邮件总开关====================================================================================
    receiver = ['809341512@qq.com']
    receiver_alias = name

    body = massage
    msg = MIMEText(body, 'html')
    msg['subject'] = subject
    msg['from'] = sender_alias
    msg['to'] = receiver_alias

    s = smtplib.SMTP_SSL(host, port)
    s.login(sender, password)
    s.sendmail(sender, receiver, msg.as_string())
    # print('发送成功')


def Unique(List):
    L = []
    for str in List:
        if str not in L:
            L.append(str)
    return L


def acquireSquence(Gene):
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    print(BASE_DIR)
    msg = 'The protein sequence is not present.'
    io = os.path.join(os.path.join(BASE_DIR,'dataTable'),'dataTable.xls')
    data = pd.DataFrame(pd.read_excel(io, sheet_name=0))
    for x in range(0, len(data)):
        gene = data.iloc[x][1]
        if gene == Gene:
            msg = data.iloc[x][12]
    return msg