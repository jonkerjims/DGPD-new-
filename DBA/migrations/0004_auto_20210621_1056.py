# Generated by Django 2.2.2 on 2021-06-21 10:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DBA', '0003_auto_20210619_1210'),
    ]

    operations = [
        migrations.AddField(
            model_name='dba_manager',
            name='is_delete',
            field=models.IntegerField(default=1, max_length=1),
        ),
        migrations.AlterField(
            model_name='dba_manager',
            name='name',
            field=models.CharField(default='username', max_length=30),
        ),
    ]