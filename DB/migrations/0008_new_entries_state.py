# Generated by Django 2.2.2 on 2021-06-24 10:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DB', '0007_auto_20210623_2222'),
    ]

    operations = [
        migrations.AddField(
            model_name='new_entries',
            name='state',
            field=models.CharField(default='未读', max_length=10),
        ),
    ]