import re

from django.test import TestCase

# Create your tests here.


def verification():
    file_path = r'C:\Users\80934\Desktop\DGPD\dbWorm\static\userUpload\textarea\Fasta.fa'
    # file_path = r'C:\Users\80934\Desktop\DGPD\dbWorm\static\userUpload\protein_sequences6666666.txt'
    verify_res, state = '我们会将处理完的结果发送至您的邮箱！', 100

    with open(file_path, 'r',encoding='utf-8') as f:
        records = f.read()
        # print(1)
        # 检测文件大小，以及是否存在label
        count = len(open(file_path, 'r',encoding='utf-8').readlines())
        if count > 10000:
            verify_res, state = 'The file is too large. Please upload it again.', 200
        else:
            if re.search('>', records) == None:
                verify_res, state = 'The input file seems not have label.(Please refer to the sample.)', 200
            else:
                """
                    此处必须重新打开文件，因为上一次打开的文件已经失效
                """
                with open(file_path, 'r',encoding='utf-8') as h:
                    for line in h:
                        # print(line)
                        if '>' in line:
                            content_1 = line.split('>')[1]
                            if str.count(content_1,'|') == 2:
                                content_2 = content_1.split('|')[1]
                                content_3 = content_1.split('|')[2]
                                if (content_2 != '' and content_2 != None) and (content_3 != '' and content_3 != None):
                                    pass
                                else:
                                    verify_res, state = 'The values before and after "|" can`t be empty.(Please refer to the sample.)', 200
                                    break
                            else:
                                verify_res, state = 'The label seems not have " | ".(Please refer to the sample.)', 200
                                break
                        else: # 判断是否是蛋白质序列
                            line = line.replace('\n','')
                            res = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(line).upper())
                            if '-' in res:
                                verify_res, state = 'The protein sequence seems to be wrong.(Please refer to the sample.)', 200

    return verify_res, state

verify_res, state = verification()
print(verify_res, state)

def read_protein_sequences():
    file = r'C:\Users\80934\Desktop\DGPD\dbWorm\static\userUpload\textarea\Fasta.fa'
    # file = r'C:\Users\80934\Desktop\DGPD\dbWorm\static\userUpload\protein_sequences6666666.txt'
    with open(file,encoding='utf-8') as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        print(array)
        header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        label = header_array[1] if len(header_array) >= 1 else '0'
        label_train = header_array[2] if len(header_array) >= 2 else 'training'
        fasta_sequences.append([name, sequence, label, label_train])
    return fasta_sequences

# res = read_protein_sequences()
# print(res)