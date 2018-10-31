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

# 微信登陆，会跳出二维码，手机微信扫码登陆即可
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
# 获得当前路径
cp = os.getcwd()
# 路径拼接
path = os.path.join(cp + file)
# 切换路径
os.chdir(path)

number_of_friends = len(friends)
print(number_of_friends)
# 将friends转换为pandas DataFrame格式
df_friends = pd.DataFrame(friends)

# 获得朋友性别数，获得男、女性别数
def get_sex_count(Sequence):
    counts = defaultdict(int)
    for x in Sequence:
        counts[x] += 1
    return counts

Sex = df_friends.Sex
print(Sex)
# Sex_count1为一个字典
Sex_count1 = get_sex_count(Sex)
print(Sex_count1)
# pandas为Series提供了一个value_counts()方法，可以更方便统计各项出现的次数
Sex_count2 = Sex.value_counts()
print(Sex_count2)

# 获得省份信息
Province = df_friends.Province
Provinde_count = Province.value_counts()
Provinde_count = Provinde_count[Provinde_count.index != '']
print(Provinde_count)

# 获得城市信息
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
    plt.savefig(path + '\\sexFigure.jpg')
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

plot_bar_sex(['男','女','不详'], Sex_count2)
write_name_all(myNickName, Sex_count2, Provinde_count, City_count)

# 获得并处理 微信签名
Signatures = df_friends.Signature
# 匹配表情
regex1 = re.compile('<span.*?</span>')
# 匹配两个及以上任意非空字符
regex2 = re.compile('\s{2,}')
# 用一个空格替换表情和多个空格
# sub(pattern,repl,string)实现将一个字符串中所有符合正则表达式的子串进行替换
Signatures = [regex2.sub(' ',regex1.sub('',signature,re.S)) for signature in Signatures]
Signatures = [signature.replace('\n',' ') for signature in Signatures]
Signatures = [signature for signature in Signatures if len(signature)>0]

print(Signatures)
text_Signatures = ''.join(Signatures)
print(' ----------  ')
print(text_Signatures)
file_signatures = myNickName + '_wechat_signatures.txt'
file_signatures = path + '\\' + file_signatures
with open(file_signatures, 'w', encoding='utf-8') as f:
    f.write(text_Signatures)
    f.close()

sigt_word_list = jieba.cut(text_Signatures,cut_all=True)
word_space_split = '/'.join(sigt_word_list)
print(' ------  ')
print(word_space_split)

# coloring = np.array(Image.open(path + '\\' + 'background.jpg'))
#生成词云。font_path="C:\Windows\Fonts\msyhl.ttc"指定字体，有些字e不能解析中文，这种情况下会出现乱码。
my_wordcloud = WordCloud(background_color="white", max_words=2000, max_font_size=60, random_state=42, scale=2,
                         font_path="E:/GitCode/useAndLearnCode/spideruse/wechat/Figo/HYZhengYuan-75W-2.otf").generate(word_space_split)

file_name_p = myNickName + '_word.jpg'
file_name_p = path + '\\' + file_name_p
#保存图片
my_wordcloud.to_file(file_name_p)