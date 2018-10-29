import sqlite3 #数据模块
from bs4 import BeautifulSoup #解析网址模块
import time #时间模块
from selenium import webdriver #浏览器模块
from datetime import datetime
import traceback
import re
import my_mysql


def scroll(driv):
    driv.execute_script("""   
    (function () {   
        var y = document.body.scrollTop;   
        var step = 100;   
        window.scroll(0, y);   

        function f() {   
            if (y < document.body.scrollHeight) {   
                y += step;   
                window.scroll(0, y);   
                setTimeout(f, 50);   
            }  
            else {   
                window.scroll(0, y);   
                document.title += "scroll-done";   
            }   
        }   
        setTimeout(f, 1000);   
    })();   
    """)

def wukongOpearte(keyword, num):
    firstUrl = 'https://www.wukong.com'
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    driv = webdriver.Chrome(chrome_options=chrome_options)
    driv.get(firstUrl)  # 在谷歌浏览器中打开网址

    urlDict = {}
    soup = BeautifulSoup(driv.page_source, "lxml")  # 解析当前网页
    for li in soup.find_all('a', class_='tag-item'):
        urlName = li.span.text
        url = li['href']
        urlDict[urlName] = url
    # print("请输入以下内容中的一个，填完按Enter")
    # print("||",end=" ")
    # for key in urlDict:
    #     print(key,end=" ")
    # print("||",end="\n")
    # keyword = input().strip()
    # print("请输入滚动次数，整数：",end="\n")
    # maxTimes = int(input().strip())
    # print(keyword)
    keyword = keyword
    maxTimes = num
    keyUrl = urlDict[keyword]
    # print(keyUrl)

    wholeUrl = firstUrl + keyUrl
    if keyword == "热门":
        wholeUrl = firstUrl
    driv.get(wholeUrl)
    time.sleep(2)

    print("开始模拟鼠标拉到文章底部")
    b = 0
    c = 0
    while b < maxTimes:  # 设置循环，可替换这里值来选择你要滚动的次数
        scroll(driv)  # 滚动一次
        b = b + 1
        if (b % 10 == 0) or (b == 1):
            print('正在拉动第 {} 次...'.format(b))
        if c < 10:
            c = c + 1
        time.sleep(c)  # 休息c秒的时间

    title = ['序号', '问题网址', '关键词', '问题', '回答数', '收藏数', '获赞数', '评论数']

    # 只取字符串中的数字
    patt1 = r"\d+\.?\d*"
    patt1 = re.compile(patt1)
    print('-------------------------------------------------------------------')
    soup = BeautifulSoup(driv.page_source, "lxml")  # 解析当前网页
    a = 1
    ones = []
    questions = soup.find_all('div', class_="question-v3")
    print(questions.__len__())
    for li in questions:
        try:
            channels = keyword
            url = 'www.wukong.com' + li.a['href']  # 每个文章的地址
            NewsID = re.search(patt1, li.a['href']).group(0)
            question = li.find('a', target="_blank").text
            answer = li.find('span', class_="question-answer-num").text
            follow = li.find('span', class_="question-follow-num").text
            try:  # 捕获异常
                like = li.find('span', class_="like-num").text  # 检验语句
            except BaseException:  # 异常类型
                like = 0  # 如果检验语句有异常，那么执行这一句
            else:  # 如果没有异常，那么执行下一句
                like = li.find('span', class_="like-num").text
            try:  # 同上
                review = li.find('span', class_="comment-count").text
            except BaseException:
                review = 0
            else:
                review = li.find('span', class_="comment-count").text

            if follow == '暂无收藏':  # 如果问题没人收藏，那么跳过该问题
                continue
            elif question == '':  # 如果问题没有文字，跳过该问题
                continue
            elif int(like) == 0:  # 如果点赞人数为0，那么跳过该问题，在这里可以设置
                continue
            elif int(review) == 0:
                continue
            elif a < 1000000:  # 这里可以你需要爬取的问题的个数，如果已经爬取小于50个问题，那么爬下这个问题。
                print("正在爬取第{}篇文章".format(a))
                answer = re.search(patt1, answer).group(0)
                follow = re.search(patt1, follow).group(0)

                print("第 {} 个 :".format(a))
                print("NewsID ：", NewsID)
                print("详情URL ：", url)
                print("关键词 ：", channels)
                print("问题 ：", question)
                print("回答数 ：", answer)
                print("收藏数 ：", follow)
                print("点赞数 ：", like)
                print("评论数 ：", review)
                one = (NewsID, url, channels, question, answer, follow, like, review)
                ones.append(one)
                print('-------------------------------------------------------------------')
                a = a + 1
            else:  # 如果不满足以上条件，直接跳出循环，停止爬虫
                break
        except:
            print(traceback.format_exc())
            continue
    effectRow = my_mysql.insertBatchWukong('tb_wukong', ones)
    print("抓完咯")
    print("关闭浏览器")
    driv.quit()
    return effectRow

if __name__ == '__main__':
    startTime = datetime.now()
    keywords = ['科技', '美食', '军事', '财经', '动漫', '汽车', '热门', '国际', '育儿', '旅游',
                '三农', '数码', '家居', '时尚', '科学', '游戏', '历史', '收藏', '健康',
                '心理', '电影', '教育', '宠物', '职场', '娱乐', '社会', '体育','文化']

    keywords = ['科技', '美食', '军事', '财经', '动漫', '汽车']
    nums = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
            500, 500, 500, 500, 500, 500, 500]
    nums = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    nums = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100]

    num = 0
    insertNum = 0
    for x in keywords:
        print("=====================第 {0} 个关键字：{1}=====================".format(num + 1, x))
        startTime = datetime.now()
        print("开始时间：", startTime)
        keyword = x
        maxTimes = nums[num]
        num = num + 1
        effectRow = wukongOpearte(keyword, maxTimes)
        insertNum = insertNum + effectRow
        endTime = datetime.now()
        print("结束时间：", endTime)
        timeDuration = (endTime - startTime).seconds
        print('耗时 {} 秒'.format(timeDuration))
    print("-----总插入 {} 个记录------".format(insertNum))
    print("正在删除重复记录")
    my_mysql.deleteRepeatRows('tb_wukong')
    endTime = datetime.now()
    timeDuration = (endTime - startTime).seconds
    print('总耗时 {} 秒'.format(timeDuration))