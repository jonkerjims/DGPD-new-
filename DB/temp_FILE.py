List =[{'gene_name': 'TgGRA1', 'organism': 'Toxoplasma'}, {'gene_name': 'CSUI_005170', 'organism': 'Cystoisospora suis'}, {'gene_name': 'HHA_270250', 'organism': 'Hammondia hammondi'}, {'gene_name': 'NcGRA1', 'organism': 'Neospora'}, {'gene_name': 'TgGRA2', 'organism': 'Toxoplasma'}, {'gene_name': 'NcGRA2', 'organism': 'Neospora'}]

Dict = {}
gene= []
orga= []

for l in List:
    for key in l:
        if key=='gene_name':
            gene.append(l[key])
        if key=='organism':
            orga.append(l[key])
Dict['gene_name'] = gene
Dict['organism'] = orga
# print(Dict)
