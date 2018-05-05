# f=open('data/quora.csv')
# f1=open('data/train.csv','w+')
# f2=open('data/test.csv','w+')
# count=0
# th=350000
# for line in open('data/quora.csv'):
#     count+=1
#     # print(line)
#     if count==0:
#         f1.write(line)
#         f2.write(line)
#     elif count<th:
#         f1.write(line)
#     else:
#         f2.write(line)
#
# # print("the th of record has bee")
# f.close()
# f1.close()
# f2.close()

with open('data/quora.csv') as reader, open('data/new1.csv', 'w') as writer:
    print("test_id","question1","question2",file=writer)
    count=0
    for line in reader:
        count+=1
        if count>350000:
            items = line.strip().split(',')
            # print(','.join(items[:1]+items[3:-1]), file=writer)
            print(''.join(items[-1]), file=writer)
# import pandas as pd
# data = pd.read_csv('data/test.csv',usecols=[3,4],quotechar ='“')
# print(data.iloc[0:5])
# data.to_csv('data/new.csv',header=["question1","question2"])