# Generated by Django 2.2.2 on 2021-06-23 22:22

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('DB', '0006_auto_20210623_2142'),
    ]

    operations = [
        migrations.AlterField(
            model_name='new_entries',
            name='time',
            field=models.DateTimeField(default=django.utils.timezone.now, verbose_name='保存日期'),
        ),
    ]
