# _*_ coding: utf-8 _*_
"""
Time:     2021/6/21 10:30
Author:   Jason_Xue(薛伟-vx：xw809341512)
Version:  V 1.0
File:     level_manage.py
Describe: 这块主要写后台管理系统
          1、个人中心部分的个人信息修改。

"""
import hashlib

from django.http import HttpResponse, JsonResponse

from DBA import models


def self_center_update(request):
    if request.method == 'GET':
        userId = request.GET.get('userId')
        username = request.GET.get('username')
        age = request.GET.get('age')
        password = request.GET.get('password')
        print(password,type(password))
        try:
            query_set = models.DBA_manager.objects.filter(email=userId)
            hash_pwd = hashlib.new('md5', query_set[0].password.encode('utf-8')).hexdigest()
            if password == hash_pwd:
                password = query_set[0].password
            query_set.update(name=username, age=age, password=password)

            request.session['username'] = username
            print(request.session['username'])
            request.session['age'] = age
            data = {
                'username': username,
                'msg': '修改成功,记得刷新页面~'
            }
            return JsonResponse(data)
        except Exception:
            print(Exception)
            data = {
                'username': username,
                'msg': '修改失败！'
            }
            return JsonResponse(data)
