# Generated by Django 2.2.2 on 2021-06-19 12:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DBA', '0002_dba_manager_code'),
    ]

    operations = [
        migrations.AddField(
            model_name='dba_manager',
            name='age',
            field=models.IntegerField(default=22, max_length=3),
        ),
        migrations.AddField(
            model_name='dba_manager',
            name='level',
            field=models.IntegerField(default=1, max_length=10),
        ),
        migrations.AddField(
            model_name='dba_manager',
            name='name',
            field=models.CharField(default='username', max_length=20),
        ),
        migrations.AlterField(
            model_name='dba_manager',
            name='code',
            field=models.CharField(default='abcd', max_length=4, verbose_name='验证码'),
        ),
    ]
