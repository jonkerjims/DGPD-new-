# _*_ coding: utf-8 _*_
"""
Time:     2021/6/21 10:30
Author:   Jason_Xue(薛伟-vx：xw809341512)
Version:  V 1.0
File:     level_manage.py
Describe: 这块主要写后台管理系统
          1、登陆。
          2、页面跳转
          3、session跟踪
          4、错误信息返回去等功能
"""
import hashlib
import random
import time

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect

# Create your views here.
from DBA import models
from DBA.GlobeUtils import send_email


# 登录
from DBA.submit_msg import submit_msg_data
from DBA.userCom_msg import userCom_msg_data


def login(request):
    if request.method == "GET":
        res = request.COOKIES
        userId = res.get('userId','')
        return render(request, 'DBA_tmp/login.html',context={'userId':userId})
    if request.method == "POST":
        userId = request.POST.get('userId')
        code = request.POST.get('code')
        password = request.POST.get('password')

        request.session['userId'] = userId
        # print(userId,'/',query_set[0].email)
        # if query_set[0].email:
        try:
            query_set = models.DBA_manager.objects.filter(email=userId)
            if password != '':
                # hash_pwd = hashlib.new('md5', hz.encode('utf-8')).hexdigest()
                 if query_set[0].password == password:
                     # 设置session
                     request.session['is_login'] = True
                     request.session['username'] = query_set[0].name
                     request.session['level'] = query_set[0].level
                     request.session['age'] = query_set[0].age
                     request.session['userId'] = query_set[0].email
                     request.session['password'] = query_set[0].password
                     request.session['original_id'] = query_set[0].original_id
                     request.session['password'] = query_set[0].password

                     # 设置session关闭浏览器时失效
                     request.session.set_expiry(0)

                     # 登陆成功后更换验证码
                     code = "".join(random.sample("0123efghjkl456mnopqstuvwsz789abcdef", 4))
                     models.DBA_manager.objects.filter(email=query_set[0].email).update(code=code)
                     re = redirect('../index/')
                     re.set_cookie('userId',query_set[0].email,path='../')
                     return re
                 else:
                     userId = request.session.get('userId', '')
                     return render(request, 'DBA_tmp/login.html', {'userId': userId, 'errormsg': '提示：密码错误！请重新输入'})

            else:
                if code == query_set[0].code:

                    # 设置session
                    request.session['is_login'] = True
                    request.session['username'] = query_set[0].name
                    request.session['level'] = query_set[0].level
                    request.session['age'] = query_set[0].age
                    request.session['userId'] = query_set[0].email
                    request.session['original_id'] = query_set[0].original_id
                    request.session['password'] = query_set[0].password


                    # 设置session关闭浏览器时失效
                    request.session.set_expiry(0)

                    # 登陆成功后更换验证码
                    code = "".join(random.sample("0123efghjkl456mnopqstuvwsz789abcdef", 4))
                    models.DBA_manager.objects.filter(email=query_set[0].email).update(code=code)
                    re = redirect('../index/')
                    re.set_cookie('userId', query_set[0].email, path='../')
                    return re
                else:
                    userId = request.session.get('userId', '')
                    return render(request, 'DBA_tmp/login.html', {'userId': userId, 'errormsg': '提示：验证码错误！请重新输入'})
        except Exception:
            userId = request.session.get('userId', '')
            return render(request, 'DBA_tmp/login.html', {'userId': userId, 'errormsg': '提示：用户名不存在！请获取验证码进行注册~'})


# 获取验证码
def acquire_code(request):
    if request.method == 'GET':
        userId = request.GET.get('userId')
        code = "".join(random.sample("0123efghjkl456mnopqstuvwsz789abcdef", 4))
        try:
            query_set = models.DBA_manager.objects.filter(email=userId).values('email')
            if list(query_set) == []:
                t = time.time()
                hz = 'ID_%s' % int(round(t * 1000))
                # print(code)
                hash_pwd = hashlib.new('md5',hz.encode('utf-8')).hexdigest()
                # print(hash_pwd)
                models.DBA_manager.objects.create(email=userId,code=code,add_by=userId,original_id=hz,password=hash_pwd)
            else:
                models.DBA_manager.objects.filter(email=userId).update(code=code)
                # print(code)

            msg = '验证码：<a style="color:#2684b6">' + code + '</a>,此验证码用于登录“ dbWorm后台管理系统。”'
            subject = '【生物信息实验室】'
            back_state = send_email(userId, msg, subject)

            if back_state == 0:
                models.DBA_manager.objects.filter(email=userId).update(is_delete=0)
            data = {
                'msg':back_state,
                'code':code,
            }
            return JsonResponse(data)
        except Exception as e:
            # print(e)
            pass

def logout(request):
    # 删除session
    request.session.flush()
    return redirect('../login')


# 主页
def index(request):
    is_login = request.session.get('is_login', False)
    if is_login:
        username = request.session.get('username', '')
        level = request.session.get('level', '')
        if level==0:
            level='超级管理员'
        else:
            level='普通管理员'
        age = request.session.get('age', '')
        userId = request.session.get('userId','')
        original_id = request.session.get('original_id','')
        password = request.session.get('password','')

        #############################Submit_msg#############################

        msg_data_s,new_count_s = submit_msg_data(request,userId)

        #############################User_com_msg#############################

        msg_data_u, new_count_u = userCom_msg_data(request, userId)

        ####################################################################

        data = {
            'userId': userId,
            'username': username,
            'level': level,
            'age': age,
            'original_id':original_id,
            'password':password,
            'msg_data_s': msg_data_s,#这是个对象（查询结果集）
            'msg_data_u': msg_data_u,  # 这是个对象（查询结果集）
            'new_count_s':new_count_s,
            'new_count_u':new_count_u,
        }
        return render(request, 'DBA_tmp/index.html', context=data)
    else:
        userId = request.session.get('userId', '')
        return render(request, 'DBA_tmp/login.html', {'userId':userId,'errormsg': '提示：请先登陆再进入系统！'})


def indexCol(request):
    pass
