# Generated by Django 2.2.2 on 2021-06-11 09:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DB', '0002_user_comment'),
    ]

    operations = [
        migrations.CreateModel(
            name='New_entries',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('email', models.EmailField(max_length=50)),
                ('gene_name', models.CharField(max_length=100)),
                ('organism', models.CharField(max_length=100)),
                ('more', models.CharField(max_length=500)),
            ],
        ),
    ]
