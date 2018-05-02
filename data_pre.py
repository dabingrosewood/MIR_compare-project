f=open('data/quora.csv')
f1=open('data/train.csv','w+')
f2=open('data/test.csv','w+')
count=0
th=350000
for line in open('data/quora.csv'):
    count+=1
    # print(line)
    if count<th:
        f1.write(line)
    else:
        f2.write(line)

# print("the th of record has bee")
f.close()
f1.close()
f2.close()