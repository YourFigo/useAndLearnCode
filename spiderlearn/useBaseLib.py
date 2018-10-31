# urllib
# urlopen()
# 利用最基本的urlopen()方法，可以完成最基本的简单网页的GET请求抓取。
import urllib.request
import urllib.parse
import urllib.error
import socket
# x1 = 'http://www.python.org'
# x2 = 'https://www.9366df.com/move/3/index.html'
# response = urllib.request.urlopen(x1)
# 输出源代码
# print(response.read().decode('utf-8'))
# 输出响应类型
# print(type(response))
# 可以发现，它是一个HTTPResposne类型的对象，主要包含read()、read into()、getheader(name)、getheaders() 、fileno()等方法，
# 以及msg 、version 、status 、reason 、debuglevel 、closed 等属性。
# print(response.status)
# print(response.getheaders())
# print(response.getheader('Server'))

# data 参数
# data = bytes(urllib.parse.urlencode({'word':'hello'}),encoding='utf8')
# response = urllib.request.urlopen('http://httpbin.org/post',data= data)

# timeout 参数
# response = urllib.request.urlopen('http://httpbin.org/get',timeout=1)
# print(response.read())

# 捕获异常
# try:
#     response = urllib.request.urlopen('http://httpbin.org/get',timeout=0.1)
# except urllib.error.URLError as e:
#     if isinstance(e.reason,socket.timeout):
#         print('TIME OUT')

# 使用Request类型的对象参数 来进行请求
# request = urllib.request.Request('http://python.org')
# response = urllib.request.urlopen(request)
# print(response.read().decode('utf-8'))

# 传入多个参数构建请求
# from urllib import request,parse
# url = 'http://httpbin.org/post'
# headers = {
#     'User-Agent':'Mozilla/4.0(compatible;MAIE 5.5;Windows NT)',
#     'Host':'httpbin.org'
# }
# dict = {
#     'name':'Germey'
# }
# data = bytes(parse.urlencode(dict),encoding='utf8')
# req = request.Request(url=url,data=data,headers=headers,method='POST')
# response = request.urlopen(req)
# print(response.read().decode('utf-8'))

# Handler
# 简而言之，我们可以把它理解为各种处理器，有专门处理登录验证的，有处理Cookies 的，
# 有处理代理设置的。利用它们，我们几乎可以做到HTTP请求中所有的事情。
# HITPDefaultErrorHandler：用于处理HTTP响应错误，错误都会抛出HTTPError类型的异常。
# HTTPRedirectHandler：用于处理重定向。
# HTTPCookieProcessor： 用于处理Cookies 。
# ProxyHandler：用于设置代理，默认代理为空。
# HTTPPasswordMgr：用于管理密码，它维护了用户名和密码的表。
# HTTPBasicAuthHandler： 用于管理认证，如果一个链接打开时需要认证，那么可以用它来解决认证问题。

# 另一个比较重要的类就是OpenerDirector，我们可以称为Opener 。
# 我们之前用过urlopen()这个方法，实际上它就是urllib为我们提供的一个Opener。
# 之前使用的Request 和urlopen()相当于类库为你封装好了极其常用的请求方法，利用它们可以完成基本的请求.
# 利用Handler来构建Opener 可以实现更高级功能

# from urllib.request import HTTPPasswordMgrWithDefaultRealm,HTTPBasicAuthHandler,build_opener
# from urllib.error import URLError
# username = 'username'
# password = 'password'
# url = 'http://localhost:5000/'
# p = HTTPPasswordMgrWithDefaultRealm()
# p.add_password(None,url,username,password)
# auth_handler = HTTPBasicAuthHandler(p)
# opener = build_opener(auth_handler)
# try:
#     result = opener.open(url)
#     html = result.read().decode('utf-8')
#     print(html)
# except URLError as e:
#     print(e.reason)

# 代理
# from urllib.error import URLError
# from urllib.request import ProxyHandler,build_opener
# proxy_headler = ProxyHandler({
#     'http':'http://127.0.0.1:9743',
#     'https':'https://127.0.0.1:9743'
# })
# opener = build_opener(proxy_headler)
# try:
#     response = opener.open('http://www.baidu.com')
#     print(response.read().decode('utf-8'))
# except URLError as e:
#     print(e.reason)

# Cookies
# import http.cookiejar,urllib.request
# filename = 'cookies.txt'
# cookie = http.cookiejar.LWPCookieJar(filename)
# handler = urllib.request.HTTPCookieProcessor(cookie)
# opener = urllib.request.build_opener(handler)
# response = opener.open('http://www.baidu.com')
# for item in cookie:
#     print(item.name + "=" + item.value)
# cookie.save(ignore_discard=True,ignore_expires=True)

# 利用load()方法读取本地Cookies文件，获取Cookies的内容。读取该Cookies来请求网站
# import http.cookiejar,urllib.request
# filename = 'cookies.txt'
# cookie = http.cookiejar.LWPCookieJar(filename)
# cookie.load(filename,ignore_discard=True,ignore_expires=True)
# handler = urllib.request.HTTPCookieProcessor(cookie)
# opener = urllib.request.build_opener(handler)
# response = opener.open('http://www.baidu.com')
# print(response.read().decode('utf-8'))

# HTTPError 是 URLError的子类
# from urllib import request,error
# try:
#     response = request.urlopen('https://cuiqingcai.com/index.html')
# except error.HTTPError as e:
#     print(e.reason,e.code,e.headers,sep='\n')

# urlparse()
# 可以实现URL的识别和分段
# ://前面的就是scheme，代表协议；第一个/符号前面便是netloc，即域名，后面是path，即访问路径；
# 分号;后面是params ，代表参数；问号？后面是查询条件query，一般用作GET类型的URL;井号＃后面是锚点，用于直接定位页面内部的下拉位置。
# scheme://netloc/path ;params?query#fragment
# urlparse() 有三个参数urlstring、scheme、allow_fragments（如果它被设置为False ，干fagment部分就会被忽略，它会被解析为path 、parameters 或者query 的一部分，而fragment 部分为空）
# from urllib.parse import urlparse
# result = urlparse('http://www.baidu.com/index.html;user?id=5#comment')
# print(type(result),result,sep='\n')


# urlunparse() 实现了URL 的构造。
# 他接受的参数应该为可迭代对象，长度应该为6
# from urllib.parse import urlunparse
# data =['http','www.baidu.com','index.html','uset','a=6','comment']
# print(urlunparse(data))

# urlsplit()
# 这个方法和urlparse()方法非常相似，只不过它不再单独解析params 这一部分，只运回5个结果。上面例子中的params会合并到path中。
# from urllib.parse import urlsplit
# result = urlsplit('http://www.baidu.com/index.html;user?id=5#comment')
# print(result)
# print(result.scheme)
# print(result[0])

# urlunsplit()，它也是将链接各个部分组合成完整链接的方法
# 他接受的参数应该为可迭代对象，长度应该为5
# from urllib.parse import urlunsplit
# data =['http','www.baidu.com','index.html','a=6','comment']
# print(urlunsplit(data))

# urljoin(base_url,new_url) 生成链接
# 分析base_url的scheme 、netloc 和path这3 个内容并对新链接缺失的部分进行补充，最后返回结果。
# from urllib.parse import urljoin
# print(urljoin('http://www.baidu.com','FAQ.html'))
# print(urljoin('http://www.baidu.com','http://www.cuiqingcai.com/FAQ.html'))
# print(urljoin('http://www.baidu.com/index.html','http://www.cuiqingcai.com/FAQ.html'))
# print(urljoin('http://www.baidu.com/index.html','http://www.cuiqingcai.com/FAQ.html?question=2'))
# print(urljoin('www.baidu.com','?category=2#comment'))
# print(urljoin('www.baidu.com#comment','?category=2'))

# urlencode() 将参数由字典类型转化为GET请求参数了
# from urllib.parse import urlencode
# params = {
#     'name':'germey',
#     'age':22
# }
# base_url = 'http://www.baidu.com?'
# url = base_url + urlencode(params)
# print(url)

# parse_qs() 反序列化，如果我们有一串GET请求参数，利用parse_qs()方法，就可以将它转回字典
# from urllib.parse import parse_qs
# query = 'age=22&name=germey'
# print(parse_qs(query))
# print(type(parse_qs(query)))

# parse_qsl() 反序列化，如果我们有一串GET请求参数，利用parse_qsl()方法，就可以将它转回元组(tuple)组成的列表
# from urllib.parse import parse_qsl
# query =  'age=22&name=germey'
# print(parse_qsl(query))
# print(type(parse_qsl(query)))

# # quote() 该方法可以将内容转化为URL编码的格式。URL中带有中文参数时，有时可能会导致乱码的问题，此时用这个方法可以将中文字符转化为URL编码
# from urllib.parse import quote
# keyword = "壁纸"
# url = 'http://www.baidu.com/s?wd=' + quote(keyword)
# print(url)

# unquote() 将URL进行解码
# from urllib.parse import unquote
# url = 'http://www.baidu.com/s?wd=%E5%A3%81%E7%BA%B8'
# print(unquote(url))

# Robots协议，它的全名叫作网络爬虫排除标准（Robots Exclusion Protocol），用来告诉爬虫和搜索引擎哪些页面可以抓取，哪些不可以抓取
''' robots.txt
User-agent: *
Disallow: /
Allow: /public/
'''
# User-agent描述了搜索爬虫的名称，这里将其设置为*则代表该协议对任何爬取爬虫有效
# Disallow 指定了不允许抓取的目录，比如上例子中设置为/则代表不允许抓取所有页面。
# Allow一般和Disallow 一起使用，一般不会单独使用，用来排除某些限制。现在我们设置为/public/，则表示所有页面不允许抓取，但可以抓取public目录。

''' 禁止所有爬虫访问任何目录的代码如下：
User-agent: *
Disallow: /
'''

''' 允许所有爬虫访问任何目录的代码如下：
User-agent: *
Disallow:
'''

'''  只允许某一个爬虫访问任何目录的代码如下：
User-agent: WebCrawler
Disallow:
User-agent: *
Disallow: /
'''

# robotparser() 使用robotparser 模块来解析robots.txt 了。 urllib.robotparser.RobotFileParser(url='')
# 该模块提供了一个类RobotFileParser ，它可以根据某网站的robots.txt 文件来判断一个爬取爬虫是否有权限来爬取这个网页。
# set_url()： 用来设置robots.txt 文件的链接。如果在创建RobotFileParser对象时传入了链接，那么就不需要再使用这个方法设置了。
# read()： 读取robots.txt 文件并进行分析。这个方法不会返回任何内容，但是执行了读取操作。一定记得调用这个方法.
# parse()： 用来解析robots.txt文件，传人的参数是robots.txt某些行的内容，它会按照robots.txt的语法规则来分析这些内容。
# can_fetch()： 该方法传人两个参数， 第一个是User-agent，
# 第二个是要抓取的URL。返回的内容是该搜索引擎是否可以抓取这个URL，返回结果是True或False。
# mtime()： 返回的是上次抓取和分析robots.txt的时间
# modified()：将当前时间设置为上次抓取和分析robots.txt的时间。
# from urllib.robotparser import RobotFileParser
# from urllib.request import urlopen
# rp = RobotFileParser()
# rp.set_url('http://www.jianshu.com/robots.txt')
# rp.read()
# # rp.parse(urlopen('http://www.jianshu.com/robots.txt').read().decode().split('\n'))
# print(rp.can_fetch('*','http://www.jianshu.com/p/0826cf46928'))
# print(rp.can_fetch('*','http://www.jianshu.com/search?q=python&page=l&type=collections'))


# requests库 Cookies、登录验证、代理设置
# 调用get()方法 与 urllib.request.urlopen()实现相同的操作，得到一个Response对象。
# import requests
# r = requests.get('https://www.baidu.com')
# print(type(r))
# print(r.status_code)
# print(type(r.text))
# print(r.text)
# print(r.cookies)

###############   Get 请求
# import requests
# r = requests.get('http://httpbin.org/get')
# print(type(r.text))
# print(r.text)
# print(r.json())
# print(type(r.json()))
''' 返回结果中包含了 请求头、URL、IP等
{
  "args": {}, 
  "headers": {
    "Accept": "*/*", 
    "Accept-Encoding": "gzip, deflate", 
    "Connection": "close", 
    "Host": "httpbin.org", 
    "User-Agent": "python-requests/2.19.1"
  }, 
  "origin": "61.148.75.238", 
  "url": "http://httpbin.org/get"
}
'''

# 对于get请求，如果需要附加额外信息，使用params参数即可
# import requests
# data = {
#     'name':'germey',
#     'age':'22'
# }
# r = requests.get('http://httpbin.org/get',params=data)
# print(r.text)

# 抓取网页，知乎发现中的问题
# import requests
# import re
# headers = {
#     'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
# }
# r = requests.get("https://www.zhihu.com/explore",headers=headers)
# pattern = re.compile('explore-feed.*?question_link.*?>(.*?)</a>',re.S)
# titles = re.findall(pattern,r.text)
# print(titles)

# 抓取二进制 数据。
# open()的第一个参数是文件名称，第二个参数代表以二进制写的形式打开，可以向文件里写入二进制数据。
# import requests
# r = requests.get("https://github.com/favicon.ico")
# print(r.text)
# print(r.content)
# with open('favicon.ico','wb') as f:
#     f.write(r.content)

# 添加 headers
# import requests
# r = requests.get("https://www.zhihu.com/explore")
# print(r.text)
# # 提示 400 Bad Request  请求错误
# # headers 中加入 User-Agent 就可以成功访问，还可以加其他字段信息
# headers = {
#     'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
# }
# r = requests.get("https://www.zhihu.com/explore",headers = headers)
# print(r.text)

####################### post 请求
# import requests
# data = {'name':'germey','age':'22'}
# r = requests.post("http://httpbin.org/post",data = data)
# print(r.text)
''' 我们成功获得了返回结果，其中form 部分就是提交的数据，这就证明POST请求成功发送了。
{
  "args": {}, 
  "data": "", 
  "files": {}, 
  "form": {}, 
  "headers": {
    "Accept": "*/*", 
    "Accept-Encoding": "gzip, deflate", 
    "Connection": "close", 
    "Content-Length": "0", 
    "Host": "httpbin.org", 
    "User-Agent": "python-requests/2.19.1"
  }, 
  "json": null, 
  "origin": "61.148.75.238", 
  "url": "http://httpbin.org/post"
}
'''

# 响应
# import requests
# r = requests.get("http://www.jianshu.com")
# print(type(r.status_code),r.status_code)
# print(type(r.headers),r.headers)
# print(type(r.cookies),r.cookies)
# print(type(r.url),r.url)
# print(type(r.history),r.history)

# 状态码查询对象requests.codes
# 1xx 信息性状态码
# 2xx 成功状态码
# 3xx 重定向状态码
# 4xx 客户端错误状态码
# 5xx 服务端错误状态码
# import requests
# # 没有headers，不会输出 Request Successfully
# headers = {
#     'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
# }
# r = requests.get("http://www.jianshu.com",headers = headers)
# exit() if not r.status_code == requests.codes.ok else print('Request Successfully')

# 文件上传
# 这个网站会返回响应，里面包含files这个字段，
# 而form 字段是空的，这证明文件上传部分会单独有一个files字段来标识。
# import requests
# files = {'file':open('favicon.ico','rb')}
# r = requests.post("http://httpbin.org/post",files=files)
# print(r.text)

# Cookies
import requests
# r = requests.get("http://www.baidu.com")
# print(r.cookies)
# # RequestCookieJar 类型
# print('----')
# for key,value in r.cookies.items():
#     print(key + '=' + value)

# headers = {
#     'cookie':'_xsrf=4QA2jsWKnuefs3dnDCu3hPyZmEKjVpiI; _zap=aebaeebf-f9b9-45fa-b1c2-4b6b2f0c3bf0; d_c0="AECml1Ca9Q2PTpLvgVJv6UIHzNGTWWCtjuk=|1532600423"; __utmz=51854390.1537174049.1.1.utmcsr=zhihu.com|utmccn=(referral)|utmcmd=referral|utmcct=/people/zhaochenfei/activities; __utmv=51854390.100-1|2=registration_date=20140508=1^3=entry_date=20140508=1; q_c1=0234364453514e2eaa8a80989057c737|1538271899000|1532600423000; tst=r; __gads=ID=5619dd3aef591d0c:T=1539766577:S=ALNI_MarOTz6H8w6GaAplCRopXKA3HhKMg; __utma=51854390.1114622663.1537174049.1537174049.1539766598.2; __utmc=51854390; tgw_l7_route=5bcc9ffea0388b69e77c21c0b42555fe; capsion_ticket="2|1:0|10:1539773394|14:capsion_ticket|44:NTlhNWIwYWRiZjM2NGZlYWFkMTY5Zjk4N2ZkMGUwZjI=|fb6066dbb72aea47a2fd088d3ae5fc84a3e0577ddad168e94f2a8d629e1c2b40"; z_c0="2|1:0|10:1539773395|4:z_c0|92:Mi4xRkxkWEFBQUFBQUFBUUthWFVKcjFEU1lBQUFCZ0FsVk4wMkcwWEFDYTdDdlNyRzMtSlZBS3prNk1hU19DQm0tcWZ3|a0bff5a40e42ed394f749b41f961bc2db22ac94065e79ddaac470141de0743e2"',
#     'Host':'www.zhihu.com',
#     'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
# }
# r = requests.get('http://www.zhihu.com',headers=headers)
# print(r.text)

########## 会话维持
# 一次请求成功后，再次请求已经没有cookies了
# import requests
# requests.get('http://httpbin.org/cookies/set/number/123456789')
# r = requests.get('http://httpbin.org/cookies')
# print(r.text)
# 用Session
# import requests
# S = requests.Session()
# S.get('http://httpbin.org/cookies/set/number/123456789')
# r = S.get('http://httpbin.org/cookies')
# print(r.text)

########### 正则表达式
# p = '[a-zA-z]+://[^\s]*'  匹配URL
# \w 匹配字母、数字及下划线
# \W 匹配不是字母、数字及下划线的字符
# \s 匹配任意空白字符，等价于 [\t\n\r\f]
# \S 匹配任意非空字符
# \d 匹配任意数字，等价于 [0-9]
# \D 匹配任意非数字的字符
# \A 匹配字符串开头
# \Z 匹配字符串结尾，如果存在换行，只匹配到换行前的结束字符串
# \z 匹配字符串结尾，如果存在换行，同时还会匹配换行符
# \G 匹配最后匹配完成的位置
# \n 匹配一个换行符
# \t 匹配一个制表符
# ^ 匹配一行字符串的开头
# $ 匹配一行字符串的结尾
# . 匹配任意字符，处理换行符，当re.DOTALL 标记被指定时，则可以匹配包括换行符的任意字符
# [...] 用来表示一组字符，单独列出，比如 [amk] 匹配 a、m、k
# [^...] 不在[]中的字符，比如[^abc]匹配除了a、b、c之外的字符
# * 匹配0个或多个表达式
# + 匹配一个或多个表达式
# ? 匹配0个或者1个前面的正则表达式定义的片段，非贪婪方式
# {n} 精确匹配n个前面的表达式
# {n,m} 匹配n到m次由前面正则表达式定义的片段，贪婪方式
# a|b 匹配a或b
# () 匹配括号内的表达式，也表示一个组

# match()
# match()方法会尝试从字符串的起始位置匹配正则表达式，
# 如果匹配，就返回匹配成功的结果；如果不匹配，就返回None。
# import re
# content = 'Hello 123 4567 World_This is a Regex Demo'
# print(len(content))
# result = re.match('^Hello\s\d\d\d\s\d{4}\s\w{10}',content)
# print(result)
# # group()方法可以输出匹配到的内容
# # span()方法可以输出匹配的范围
# print(result.group())
# print(result.span())

# 匹配目标
# group()输出完整的匹配结果
# group(1)输出第1个被()包围的匹配结果
# import re
# content = 'Hello 1234567 World_This is a Regex Demo'
# result = re.match('^Hello\s(\d+)\sWorld',content)
# print(result)
# print(result.group())
# print(result.group(1))
# print(result.span())

# 通用匹配
# import re
# content = 'Hello 123 4567 World_This is a Regex Demo'
# result = re.match('^Hello.*Demo$',content)
# print(result)
# print(result.group())
# print(result.span())

# 贪婪与非贪婪
# group(1) 为 7 ，(\d+)只匹配了7，(\d+)前面的 .* 是贪婪匹配，把123456给匹配了
# import re
# content = 'Hello 1234567 World_This is a Regex Demo'
# result = re.match('^He.*(\d+).*Demo$',content)
# print(result)
# print(result.group(1))
# # (\d+)前面的 .* 改为 .*? 变为
# # 非贪婪方式 当 .*? 匹配到Hello后面的空白字符时，再往后的字符就是数字了，
# # 而 \d+ 恰好可以匹配，那么这里 .*? 的就不再进行匹配，交给 \d+ 去匹配后面的数字。
# # 在做匹配的时候，字符串中间尽量使用非贪婪匹配，也就是用 .*? 叫来代替 .*
# result = re.match('^He.*?(\d+).*Demo$',content)
# print(result)
# print(result.group(1))
# # 但这里需要注意，如果匹配的结果在字符串结尾，
# # .*? 就有可能匹配不到任何内容了，因为它会匹配尽可能少的字符.
# content = 'http://weibo.com/comment/kEeaCN'
# result1 = re.match('http.*?comment/(.*?)',content)
# result2 = re.match('http.*?comment/(.*)',content)
# print('result1 ',result1.group(1))
# print('result2 ',result2.group(1))

# 修饰符
# re.S 这个修饰符的作用是使 ．匹配包括换行符在内的所有字符。
# re.I 使匹配对大小写不敏感
# re.L 做本地化识别（locale-aware）匹配
# re.M 多行匹配，影响^和$
# re.U 根据Unicode 字符集解析字符。这个标志影响\w、\W 、\b 和\B
# re.X 该标志通过给予你更灵活的格式以便你将正则表达式写得更易于理解
# import re
# content = '''Hello 1234567 World_This
# is a Regex Demo
#  '''
# result = re.match('^He.*?(\d+).*Demo$',content,re.S)
# print(result.group(1))


# search() 它在匹配时会扫描整个字符串，然后返回第一个成功匹配的结果。
# 前面提到过 match() 方法是从字符串的开头开始匹配的，一旦开头不匹配，那么整个匹配就失败了。
# 比如下面这个例子，这里的字符串以 Extra 开头，但是正则表达式以 Hello 开头，
# 整个正则表达式是字符串的一部分，但匹配结果时None，匹配失败了，这样很不方便。
# import  re
# content = 'Extra stings Hello 1234567 World This is a Regex Demo Extra stings'
# result = re.match('Hello.*?(\d+).*?Demo',content)
# print(result)
# # 下面用 search() 就可以正确返回我们想要的结果了。
# result = re.search('Hello.*?(\d+).*?Demo',content)
# print(result.group())

# findall() 搜索整个字符串，获取匹配正则表达式的所有内容。
# import re
# str = '<a>我是小赵</a>' \
#       '<p>我是小钱</p>' \
#       '<a>我是小孙</a>' \
#       '<a>我是小李</a>' \
#       '<h1>我是小周</h1>' \
#       '<a>我是小吴</a>'
# print(str)
# result = re.findall('<a>(.*?)</a>',str,re.S)
# print(type(result))
# for x in result:
#     print(x)

# sub() 除了使用正则表达式提取信息外，有时候还需要借助它来修改文本。
# 比如，想要把一串文本中的所有数字都去掉，如果只用字符串的 replace()方法，那就太烦琐了，这时可以借助 sub()方法。
# import re
# content = 'ab2cd3Ef8RG85yujh798'
# content = re.sub('\d+','',content)
# print(content)
# content = 'ab2cd3Ef8RG85yujh798'
# content = re.sub('[a-zA-Z]','',content)
# print(content)

# compile() 将正则字符串编译成正则表达式对象，以便在后面的匹配中复用
import  re
contentl = '2016 12 15 12:00'
content2 = '2016-12-17 12:55'
content3 = '2016-12-22 13:21'
pattern = re.compile('\d{2}:\d{2}')
result1 = re.sub(pattern,'',contentl)
result2 = re.sub(pattern,'',content2)
result3 = re.sub(pattern,'',content3)
print(result1,result2,result3)

pattern = re.compile('\d{2}:\d{2}')
result1 = re.search(pattern,contentl).group()
result2 = re.search(pattern,content2).group()
result3 = re.search(pattern,content3).group()
print(result1,result2,result3)