import itchat
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import jieba
import os
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud,ImageColorGenerator
import PIL.Image as Image

itchat.login()
friends = itchat.get_friends(update=True)
# 获取自己的昵称，自己的索引为0
myNickName = friends[0].NickName
# 建立以昵称为文件名的文件夹
isExist = os.path.exists(myNickName)
if not isExist:
    os.mkdir(myNickName)

# file = '\{}'.format(NickName)
file = '\%s' %myNickName
cp = os.getcwd()
path = os.path.join(cp + file)

os.chdir(path)

number_of_friends = len(friends)
print(number_of_friends)
df_friends = pd.DataFrame(friends)

# 获得朋友性别数
def get_sex_count(Sequence):
    counts = defaultdict(int)
    for x in Sequence:
        counts[x] += 1
    return counts

Sex = df_friends.Sex
print(Sex)
Sex_count1 = get_sex_count(Sex)
print(Sex_count1)
# pandas为Series提供了一个value_counts()方法，可以更方便统计各项出现的次数
Sex_count2 = Sex.value_counts()
print(Sex_count2)

Province = df_friends.Province
Provinde_count = Province.value_counts()
Provinde_count = Provinde_count[Provinde_count.index != '']
print(Provinde_count)

City = df_friends.City
City_count = City.value_counts()
City_count = City_count[City_count.index != '']
print(City_count)

# 画柱状图
def plot_bar_sex(x,y):
    # 设置中文字体和负号正常显示
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure()
    # plt.bar(sex_count1.keys(),sex_count1.values())
    plt.bar(x,y)
    plt.show()

# 将朋友地理信息写入txt
def write_name_all(myNickName, Sex_count2, Provinde_count, City_count):
    file_name_all = myNickName + '_basic_inf.txt'
    write_file = open(path + '\\' + file_name_all, 'w')
    write_file.write(
        '你共有%d个好友,其中有%d个男生，%d个女生，%d未显示性别。\n\n' % (number_of_friends, Sex_count2[1], Sex_count2[2], Sex_count2[0]) +
        '你的朋友主要来自省份：%s(%d人)、%s(%d人)、%s(%d人)、%s(%d人)和%s(%d人)。\n\n' % (
            Provinde_count.index[0], Provinde_count[0], Provinde_count.index[1], Provinde_count[1], Provinde_count.index[2],
            Provinde_count[2], Provinde_count.index[3], Provinde_count[3], Provinde_count.index[4], Provinde_count[4]) +
        '主要来自这些城市：%s(%d人)、%s(%d人)、%s(%d人)、%s(%d人)、%s(%d人)、%s(%d人)、%s(%d人)、%s(%d人)、%s(%d人)和%s(%d人)。' % (
            City_count.index[0], City_count[0], City_count.index[1], City_count[1], City_count.index[2], City_count[2],
            City_count.index[3], City_count[3], City_count.index[4], City_count[4], City_count.index[5], City_count[5],
            City_count.index[6], City_count[6], City_count.index[7], City_count[7], City_count.index[8], City_count[8],
            City_count.index[9], City_count[9]))
    write_file.close()

plot_bar_sex(Sex_count2.index, Sex_count2)
write_name_all(myNickName, City_count, Provinde_count, City_count)
