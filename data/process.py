import random
import re

import xlrd

related = []
irrelated = []
train = []
val = []
test = []

data = xlrd.open_workbook('scrapiesrestb.xls', encoding_override="GB2312")
table = data.sheets()[0]
print(table.nrows)

for i in range(table.nrows - 1):
    if len(re.sub(r'{[^>]+}', "", table.cell(i + 1, 1).value, re.S)) > 100:
        if table.cell(i + 1, 12).value == '短缺药相关':
            related.append(i + 1)
        else:
            irrelated.append(i + 1)

random.shuffle(related)
random.shuffle(irrelated)

train = related[0:int(len(related)/13*10)] + irrelated[0:int(len(irrelated)/13*10)]
val = related[int(len(related)/13*10):int(len(related)/13*12)] + irrelated[int(len(irrelated)/13*10):int(len(irrelated)/13*12)]
test = related[int(len(related)/13*12):len(related)] + irrelated[int(len(irrelated)/13*12):len(irrelated)]
print(len(train)+len(val)+len(test))

f_train = open('drug/drug.train.txt', 'w', encoding='utf-8')
f_test = open('drug/drug.test.txt', 'w', encoding='utf-8')
f_val = open('drug/drug.val.txt', 'w', encoding='utf-8')

for i in train:
    f_train.write(table.cell(i, 12).value + '\t' + table.cell(i, 9).value + re.sub(r'{[^>]+}', "", table.cell(i, 1).value, re.S) + '\n')
for i in val:
    f_val.write(table.cell(i, 12).value + '\t' + table.cell(i, 9).value + re.sub(r'{[^>]+}', "", table.cell(i, 1).value, re.S) + '\n')
for i in test:
    f_test.write(table.cell(i, 12).value + '\t' + table.cell(i, 9).value + re.sub(r'{[^>]+}', "", table.cell(i, 1).value, re.S) + '\n')

f_train.close()
f_test.close()
f_val.close()