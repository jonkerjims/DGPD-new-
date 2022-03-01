from django.db import models
from django.utils import timezone


class Tbworm(models.Model):
    id = models.AutoField(primary_key=True)
    gene_name = models.CharField(db_column='Gene_name', max_length=255, blank=True, null=True)  # Field name made lowercase.
    organism = models.CharField(db_column='Organism', max_length=255, blank=True, null=True)  # Field name made lowercase.
    evidece_level = models.CharField(db_column='Evidece_level', max_length=255, blank=True, null=True)  # Field name made lowercase.
    intron = models.CharField(db_column='Intron', max_length=255, blank=True, null=True)  # Field name made lowercase.
    signal_peptide = models.CharField(db_column='Signal_Peptide', max_length=255, blank=True, null=True)  # Field name made lowercase.
    tmhmm = models.CharField(db_column='TMHMM', max_length=255, blank=True, null=True)  # Field name made lowercase.
    pmid = models.CharField(db_column='Pmid', max_length=255, blank=True, null=True)  # Field name made lowercase.
    molecular_weight = models.CharField(db_column='Molecular_Weight', max_length=255, blank=True, null=True)  # Field name made lowercase.
    domains = models.CharField(db_column='Domains', max_length=255, blank=True, null=True)  # Field name made lowercase.
    date_source = models.CharField(db_column='Date_Source', max_length=255, db_collation='utf8_bin', blank=True,
                                   null=True)  # Field name made lowercase.
    ncbi_index = models.CharField(db_column='NCBI_Index', max_length=255, blank=True,
                                  null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'tbworm'

class User_comment(models.Model):
    id = models.AutoField(primary_key=True)
    email = models.EmailField(max_length=50)
    reference = models.CharField(max_length=100)
    text = models.CharField(max_length=500)
    read_sum = models.IntegerField(max_length=6, default=1)
    time = models.DateTimeField('保存日期', default=timezone.now)

class New_entries(models.Model):
    id = models.AutoField(primary_key=True)
    time = models.DateTimeField('保存日期',default = timezone.now)
    email = models.EmailField(max_length=50,default='-')
    gene_name = models.CharField(max_length=100,default='-')
    organism = models.CharField(max_length=100,default='-')
    more = models.CharField(max_length=500,default='-')
    read_sum = models.IntegerField(max_length=6,default=1)


