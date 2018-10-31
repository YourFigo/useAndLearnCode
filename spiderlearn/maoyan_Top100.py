import requests
import re
import json
import os
import time

# 获取单页的html
def get_one_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36'
    }
    # requests.get(url,headers) 头这样设置无法返回结果，相当于没有加请求头
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    return None

# 进行单页的解析，返回单页中我们需要的全部信息
def parse_one_page(html):
    # 提取排名数 <dd>.*?board-index.*?>(.*?)</i>
    # 提取图片  data-src="(.*?)"
    # 提取电影名  <p.*?name.*?a.*?>(.*?)</a></p>
    # 提取演员  <p.*?star">(.*?)</p>
    # 提取上映时间  <p.*?releasetime">(.*?)</p>
    # 提取评分  <p.*?score.*?integer">(.*?)</i><i.*?fraction">(.*?)</i></p>
    pattern = re.compile('<dd>.*?board-index.*?>(.*?)</i>.*?data-src="(.*?)".*?<p.*?name.*?a.*?>(.*?)</a></p>.*?<p.*?star">(.*?)</p>.*?<p.*?releasetime">(.*?)</p>.*?<p.*?score.*?integer">(.*?)</i><i.*?fraction">(.*?)</i></p>.*?</dd>',re.S)
    items = re.findall(pattern,html)
    for item in items:
        # 使用 yield 关键字，关键字所在的函数变为一个生成器（generator），
        # 调用该函数会返回一个 iterable 对象，然后就可以迭代进行输出。
        # 使用 yield 的好处是，对于需要返回很大空间的数据，可以节省内存空间。
        # 当你调用这个函数的时候，函数内部的代码并不立马执行 ，这个函数只是返回一个生成器对象。
        # 当你使用for进行迭代的时候，函数中的代码才会执行。相当于每次返回一次迭代的数据，节省了很大的内存空间。
        yield {
            'index':item[0].strip(),
            'image':item[1].strip(),
            'title':item[2].strip(),
            'actor':item[3].strip()[3:] if len(item[3])>3 else '',
            'time':item[4].strip()[5:] if len(item[4])>5 else '',
            'score':item[5].strip() + item[6].strip()
        }

# 写入一条电影信息，实现文本追加
def write_one_movie(content):
    with open('maoyanResult.txt','a',encoding='utf-8') as f:
        # 这里通过 JSON 库的 dumps()方法实现字典的序列化，
        # 指定 ensure_ascii 参数为 False，这样可以保证输出结果是中文形式而不是 Unicode 编码
        # print(type(json.dumps(content)))
        f.write(json.dumps(content,ensure_ascii=False) + '\n')
        # 但是试了一下，直接str()转换也可以实现相同的写入
        # f.write(str(content) + '\n')

# 获取一页数据的总控制方法
def main_one_page(page_num):
    maoyan_url = 'http://www.maoyan.com/board/4?offset=' + str(page_num)
    html = get_one_page(maoyan_url)
    for item in parse_one_page(html):
        write_one_movie(item)

if __name__ == '__main__':
    if os.path.exists('maoyanResult.txt'):
        os.remove('maoyanResult.txt')
    for i in range(0,99,10):
        print('正在爬取第 {} 页 ...'.format(int(i/10 + 1)))
        main_one_page(i)
        time.sleep(1)