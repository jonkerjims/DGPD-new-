# _*_ coding: utf-8 _*_
"""
Time:     2021/6/24 11:33
Author:   Jason_Xue(薛伟-vx：xw809341512)
Version:  V 1.0
File:     userCom_msg.py
Describe: 这块主要写后台管理系统
          1、登陆。

"""
from django.http import HttpResponse

from DB import models
from DBA import models as DBAmodels

def userCom_msg_data(request,userId):
    try:
        query_set = models.User_comment.objects.all().values('email', 'time', 'reference', 'text', 'read_sum',).order_by('-time')
        count = query_set.count()
        # 把userCom留言板数据存入read_count
        new_count = DBAmodels.DBA_manager.objects.filter(email=userId)[0].read_count_u
        new_count = count - new_count

        return query_set, new_count
    except Exception:
        pass


def userCom_msg_data_update(request):
    try:
        if request.method == "GET":
            userId = request.GET.get('userId')

            query_set = models.User_comment.objects.all()
            count = query_set.count()
            DBAmodels.DBA_manager.objects.filter(email=userId).update(read_count_u=count)

            # 查出所有的字段
            for item in query_set:
                read_sum = models.User_comment.objects.filter(id=item.id)[0].read_sum
                models.User_comment.objects.filter(id=item.id).update(read_sum=read_sum + 1)

            return HttpResponse(1)
    except Exception:
        pass