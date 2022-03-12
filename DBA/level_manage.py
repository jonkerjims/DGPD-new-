# _*_ coding: utf-8 _*_
"""
Time:     2021/6/21 10:30
Author:   Jason_Xue(薛伟-vx：xw809341512)
Version:  V 1.0
File:     level_manage.py
Describe: 这块主要写后台管理系统
          1、这块主要写超级管理员增删改查普通管理员。
"""
import hashlib
import json
import time

from django.core.paginator import Paginator
from django.http import JsonResponse

from DBA import models
from DBA.GlobeUtils import trans_queryset_toJson


def level_manage_data(request):
    try:
        if request.method == 'GET':
            page = request.GET.get('page')
            limit = request.GET.get('limit')
            print(limit,page)
            query_set = models.DBA_manager.objects.filter(is_delete=1).values('id', 'email', 'code', 'age', 'level',
                                                                              'name', 'original_id','password').order_by('id')
            count = query_set.count()
            paginator = Paginator(query_set, limit)
            query_set = paginator.page(number=page)
            print(query_set)
            query_set = trans_queryset_toJson(list(query_set), count)
            return JsonResponse(query_set)
    except Exception:
        return


def level_manage_data_save(request):
    try:
        if request.method == 'GET':
            name = request.GET.get('name')
            email = request.GET.get('email')
            code = request.GET.get('code')
            age = request.GET.get('age')
            level = request.GET.get('level')
            original_id = request.GET.get('original_id')
            password = request.GET.get('password')
            save_by = request.GET.get('save_by')
            query_set = models.DBA_manager.objects.filter(original_id=original_id)
            print('前',password)
            print('数据库',query_set[0].password)
            hash_pwd = hashlib.new('md5', query_set[0].password.encode('utf-8')).hexdigest()
            if password == hash_pwd:
                password = query_set[0].password

            if level != "0" and level != "1" :
                data = {
                    'msg': '等级：0为超级管理员, 1为普通管理员,请重新选择~'
                }
                return JsonResponse(data)

            query_set.update(name=name, email=email, code=code,age=age, level=level,save_by=save_by,password=password)

            data = {
                'msg': '保存成功！'
            }
            return JsonResponse(data)

    except Exception:
        data = {
            'msg': '保存失败！'
        }
        return JsonResponse(data)


def level_manage_data_delete(request):
    try:
        if request.method == 'GET':
            is_delete = request.GET.get('is_delete')
            original_id = request.GET.get('original_id')
            del_by =request.GET.get('del_by')
            models.DBA_manager.objects.filter(original_id=original_id).update(is_delete=is_delete,del_by=del_by)
            data = {
                'msg': '删除成功！'
            }
            return JsonResponse(data)

    except Exception:
        data = {
            'msg': '删除失败！'
        }
        return JsonResponse(data)


def level_manage_data_add(request):
    try:
        t = time.time()
        hz = 'ID_%s' % int(round(t * 1000))
        add_by = request.GET.get('add_by')
        models.DBA_manager.objects.create(original_id=hz,add_by=add_by)
        data = {
            'msg': '原始ID自动创建成功！'
        }
        return JsonResponse(data)

    except Exception:
        print(Exception)
        data = {
            'msg': '创建失败！'
        }
        return JsonResponse(data)


def level_manage_data_del(request):
    try:
        if request.method == "GET":
            original_id = request.GET.get('chose_data')
            del_by = request.GET.get('del_by')

            for item in json.loads(original_id):
                models.DBA_manager.objects.filter(original_id=item['original_id']).update(is_delete=0,del_by=del_by)
                print('删除成功')

            data = {
                'msg': '批量删除成功！'
            }
            return JsonResponse(data)

    except Exception:
        print(Exception)
        data = {
            'msg': '批量删除失败！'
        }
        return JsonResponse(data)


