from django.core import serializers
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

# Create your views here.
from DB import models
from .GlobeUtils import *
from DBA import models as DBAmodels


def index(request):
    return render(request,'DB_tmp/index.html')

def indexCol(request):
    if request.method == "GET":
        Gene_name = request.GET.get('gene_name')
        Organism = request.GET.get('organism')
        # print('Gname=',Gene_name,'////Or=',Organism,type(Organism))
        try:
            # 如果都为空则查找全部
            if (Gene_name == None or Gene_name == '') and (Organism ==None or Organism == ''):
                result = models.Tbworm.objects.all().values('organism','gene_name').distinct().order_by('gene_name')
                # 调用自己的函数，查询结果集格式化
                result = trans_Queryset(result)
                #print('全部',result)
                return JsonResponse(result)
            else:
                # 定义filter动态查询 参数dict_set
                dict_set = {}
                # 定义values动态查询 参数dict_val
                dict_val = ['gene_name','organism']
                # 判断按谁查询就删除谁
                if Gene_name != None and Gene_name != '':
                    dict_set['gene_name'] = Gene_name
                    dict_val.remove('gene_name')
                if Organism !=None and Organism != '':
                    dict_set['organism'] = Organism
                    dict_val.remove('organism')
                # print('dict_set',dict_set)
                # print(dict_val)

                # 将经过处理的字典参数传入filter
                if dict_val:
                    result = models.Tbworm.objects.filter(**dict_set).values(*dict_val).distinct()
                    # 将结果序列化并转化成json格式
                    # result = serializers.serialize('json', result)
                    # 调用自己的函数，查询结果集格式化
                    result = trans_Queryset(result)
                else:
                    result = {}
                #('按条件查询',result)
                return JsonResponse(result)
        except Exception:
            #print('打印错误信息',Exception)
            pass

def Search(request):
    return render(request, 'DB_tmp/Search.html')

def SearchCol(request):
    if request.method == 'GET':
        page = request.GET.get('page')
        limit = request.GET.get('limit')
        Gene_name = request.GET.get('gene_name')
        Organism = request.GET.get('organism')
        userInput = request.GET.get('userInput')
        #print('=============================Userinput============================',userInput)
        try:
            # 定义filter动态查询 参数dict_set
            dict_set = {}
            # input模糊查询
            if userInput != None and userInput != '':
                #print("========================================================================================================================")
                result = models.Tbworm.objects.filter(Q(gene_name__contains=userInput) | Q(organism__contains=userInput)).values()
                count = result.count()
                paginator = Paginator(result, limit)
                result = paginator.page(number=page)
                result = trans_Queryset_toTable(result, count)
                #print('UserInput=======================', result)
                return JsonResponse(result)

            else:
                if Gene_name != None and Gene_name != '':
                    dict_set['gene_name'] = Gene_name
                if Organism != None and Organism != '':
                    dict_set['organism'] = Organism
                #print('dict_set', dict_set)
                # 将经过处理的字典参数传入filter
                result = models.Tbworm.objects.filter(**dict_set).values()
                count = result.count()
                paginator = Paginator(result, limit)
                result = paginator.page(number=page)

                result = trans_Queryset_toTable(result, count)
                # print('Serach=======================', result)
                return JsonResponse(result)


        except Exception:
            #print('打印错误信息', Exception)
            pass


def Details(request):
    return render(request,'DB_tmp/Details.html')

def DetailsCol(request):
    organism = request.GET.get("organism")
    gene_name = request.GET.get("gene_name")
    evidece_level = request.GET.get("evidece_level")
    intron = request.GET.get("intron")
    signal_peptide = request.GET.get("signal_peptide")
    tmhmm = request.GET.get("tmhmm")
    pmid = request.GET.get("pmid")
    molecular_weight = request.GET.get("molecular_weight")
    domains = request.GET.get("domains")
    date_source = request.GET.get("date_source")
    # print(date_source)
    dict_list = {}
    if organism != "null" and organism != '':
        dict_list['organism'] = organism
    if gene_name != "null" and gene_name != '':
        dict_list['gene_name'] = gene_name
    if evidece_level != "null" and evidece_level != '':
        dict_list['evidece_level'] = evidece_level
    if intron != "null" and intron != '':
        dict_list['intron'] = intron
    if signal_peptide != "null" and signal_peptide != '':
        dict_list['signal_peptide'] = signal_peptide
    if tmhmm != "null" and tmhmm != '':
        dict_list['tmhmm'] = tmhmm
    if pmid != "null" and pmid != '':
        dict_list['pmid'] = pmid
    if molecular_weight != "null" and molecular_weight != '':
        dict_list['molecular_weight'] = molecular_weight
    if domains != "null" and domains != '':
        dict_list['domains'] = domains
    # if date_source != "null" and date_source != '':
    #     dict_list['date_source'] = date_source
    # print(dict_list)
    try:
        result = models.Tbworm.objects.filter(**dict_list).values()
        result = trans_Queryset(result)
        # print(result)
        return JsonResponse(result)
    except Exception:
        pass


def userCommentCol(request):
    result = {}

    if request.method == 'GET':
        email = request.GET.get('email')
        reference = request.GET.get('reference')
        text = request.GET.get('text')

        try:
            if '@' not in email:
                result['state'] = 0
                return JsonResponse(result)
            else:
                query_set = DBAmodels.DBA_manager.objects.all().values('email')
                send_email_list = []
                for item in query_set:
                    send_email_list.append(item['email'])

                models.User_comment.objects.create(email=email, reference=reference, text=text)
                msg = '<div><b>User:</b>'+email+'<br><b>Comment:</b>The Reference is ['+reference+']. This UserComment is :'+text+'</div>'
                subject = 'You have a UserComment!'
                print(1)
                send_email(send_email_list,msg,subject)
                result['state'] = 1
                return JsonResponse(result)
        except Exception:
            print(Exception)
            result['state'] = 0
            return JsonResponse(result)

def About(request):
    return render(request,'DB_tmp/About.html')

def Download(request):
    return render(request,'DB_tmp/Download.html')

def Submit(request):
    return render(request,'DB_tmp/Submit.html')

def SubmitCol(request):
    result = {}
    if request.method == "GET":
        Gene_name = request.GET.get('gene_name')
        Organism = request.GET.get('organism')
        email = request.GET.get('email')
        more = request.GET.get('More')

        if '@' in email:
            #print(Gene_name+Organism+email+more)
            query_set = DBAmodels.DBA_manager.objects.all().values('email')
            send_email_list = []
            for item in query_set:
                send_email_list.append(item['email'])
            result['state'] = 1
            models.New_entries.objects.create(gene_name=Gene_name,organism=Organism,email=email,more=more)
            msg = '<div><b>User:</b>' + email + '<br><b>New entries:</b> Gene name[' + Gene_name + '], Organism[' + Organism + '], More introduction is "'+more+'"</div>'
            subject = 'You have a new entries to the database'
            send_email(send_email_list, msg, subject)
            return JsonResponse(result)
        else:
            result['state'] = 0
            return JsonResponse(result)

def Contact(request):
    return render(request,'DB_tmp/Contact.html')

def acqSeq(request):
    if request.method == "GET":
        Gene = request.GET.get('GeneName')
        sequence = acquireSquence(Gene)
    return HttpResponse(sequence)
