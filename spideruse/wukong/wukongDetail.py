from bs4 import BeautifulSoup #解析网址模块
import time #时间模块
from selenium import webdriver #浏览器模块
from datetime import datetime
import my_mysql

selectNumOne = 'SELECT wk.title,wk.keyword,wk.url,MAX( wk.replyNum ),collectNum,likeNum,commentNum FROM tb_wukong wk GROUP BY wk.keyword;'
numOneResult = my_mysql.selectTable(selectNumOne)

for li in numOneResult:
    print(li[0],li[1],li[2])
    firstUrl = 'https://www.wukong.com'
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument('--headless')
    # chrome_options.add_argument('--disable-gpu')
    driv = webdriver.Chrome(chrome_options=chrome_options)
    driv.get(firstUrl)  # 在谷歌浏览器中打开网址
    driv.get(li[2])
    driv.close()
