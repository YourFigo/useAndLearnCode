# -- coding: utf-8 --
import requests
import re
import json
import os
import time
import sqlite3


prefix_path = 'http://xh.5156edu.com'
headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36'
    }
sqlite_path = R'D:/3_other_code/GitCode/useAndLearnCode/spiderlearn/xinhua.sqlite3'

letterList = ['y','j']
numMumList = [15,14]

total_num = 0

# 获取每个拼音对应的所有汉字
# 'http://xh.5156edu.com/html2' + '/j' + '01' + '.html'
def get_one_pinyin(letter,num):
    # requests.get(url,headers) 头这样设置无法返回结果，相当于没有加请求头
    path = prefix_path + '/html2/' + letter + ('0'+str(num) if num<10 else ''+str(num)) + '.html'
    # print(path)
    response = requests.get(path, headers=headers)
    if response.status_code == 200:
        response.encoding = 'gbk'
        return response.text
    return None

# 进行单拼音页的解析，返回单拼音页中我们需要的全部信息
def parse_one_page(content):
    # "href=.*？'(.*?)'>(.*?)<span>(.*?)</span></a>"
    pattern = re.compile("href='(.*?)'>(.*?)<span>(.*?)</span></a>",re.S)
    items = re.findall(pattern,content)
    for item in items:
        # href 笔画
        # character 汉字
        # numOfStrokes 笔画数
        yield {
            'href':item[0].strip(),
            'item':item[1].strip(),
            'numOfStrokes':item[2].strip(),
            'meaning':''
        }


# 获取每个汉字的含义
# 'http://xh.5156edu.com' + '/html3/2667.html'
def get_one_character(href):
    path = prefix_path + href
    # print(path)
    response = requests.get(path, headers=headers)
    if response.status_code == 200:
        response.encoding = 'gbk'
        return response.text
    return None

# 获取每个汉字的解释
def parse_one_character(content):
    pattern = re.compile('class=font_18>(.*?)相关词语', re.S)
    item = re.search(pattern, content).group(1)
    return item

def get_one_pinyin_all():
    html = get_one_pinyin('j', 1)
    for item in parse_one_page(html):
        oneCharacterHref = get_one_character(item['href'])
        meaning = parse_one_character(oneCharacterHref)
        item['meaning'] = meaning
        time.sleep(1)


def create_tb():
    sql = (
        "create table if not exists tb_xinhua("
        "id integer primary key autoincrement, "
        "letter varchar(2), "
        "href varchar(20), "
        "item varchar(5), "
        "num varchar(5), "
        "meaning text )"
        )
    with sqlite3.connect(sqlite_path) as conn: ##链接数据库
        cursor = conn.cursor() ##获得游标
        cursor.execute(sql) ##执行命令
        conn.commit() ##

def save_one(one):
    with sqlite3.connect(sqlite_path) as conn:
        sql = (
            "insert into tb_xinhua"
            "(id,letter,href,item,num,meaning)"
            "values (?,?,?,?,?,?)")
        cursor = conn.cursor()
        cursor.execute(sql, one)
        conn.commit()

def get_all_one(letter,num):
    print('---------------------------------------')
    one_pinyin_html = get_one_pinyin(letter, num)
    num_one_pinyin = 0
    for item in parse_one_page(one_pinyin_html):
        oneCharacterHref = get_one_character(item['href'])
        meaning = parse_one_character(oneCharacterHref)
        one = (None, letter, item['href'], item['item'], item['numOfStrokes'], meaning)
        save_one(one)
        num_one_pinyin += 1
        global total_num
        total_num = total_num + 1
        print('本音第: ' +str(num_one_pinyin) + ',  字: ' + item['item'] + ',  总: ' + str(total_num))

def get_all():
    for i in range(10,15):
        print(i+1)
        get_all_one('y', i+1)

    for i in range(14):
        get_all_one('j', i+1)


if __name__ == '__main__':

    # create_tb()
    get_all()