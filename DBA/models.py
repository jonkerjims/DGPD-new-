from django.db import models
from django.utils import timezone


class DBA_manager(models.Model):
    id = models.AutoField(primary_key=True)
    email = models.EmailField(max_length=50,default='-')
    code = models.CharField('验证码',max_length=4,default='-')
    name = models.CharField(max_length=30,default='-')
    level = models.IntegerField(max_length=10,default=1)
    age = models.IntegerField(max_length=3,default=22)
    is_delete = models.IntegerField(max_length=1,default=1)
    original_id = models.CharField(max_length=255,default=None)
    add_by = models.CharField(max_length=30,default='-')
    del_by = models.CharField(max_length=30, default='-')
    save_by = models.CharField(max_length=30, default='-')
    password = models.CharField(max_length=255,default='-')
    read_count = models.IntegerField(max_length=6, default=0)
    read_count_u = models.IntegerField(max_length=6, default=0)

class DBA_login_msg(models.Model):
    id = models.AutoField(primary_key=True)
    email = models.EmailField(max_length=50)
    time = models.DateTimeField('保存日期',default = timezone.now)
    code = models.CharField(max_length=50)

