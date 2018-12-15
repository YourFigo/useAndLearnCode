# def text2vec1(text):
#     vec = []
#     for i in text:
#         for j in range(10):
#             x = ord(i)
#             print('i = {}'.format(i))
#             print('j = {}'.format(j))
#             print('ord(i) = {}'.format(x))
#             if ord(i) - 48 != j:
#                 vec.append(0)
#             else:
#                 vec.append(1)
#     return vec
#
# vec = text2vec1(['0','1','2','3','4','5','6','7','8','9'])
# l = []
# for i in range(100):
#     l.append(i)
# print(l)
# print(vec)

import  os

if not os.path.exists('/model_200'):
    os.mkdir('/model_200')

with open( '/model_200/test_acc.txt', 'w') as f:
    f.write('a')
    f.write('\n')
    f.write('b')