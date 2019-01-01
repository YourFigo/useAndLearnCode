# wechat autoreply
import itchat
import requests
import re



# 抓取网页
def getHtmlText(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""


# 自动回复
# 封装好的装饰器，当接收到的消息是Text，即文字消息
@itchat.msg_register(itchat.content.TEXT)
def text_reply(msg):
    # 当消息不是由自己发出的时候
    if not msg['FromUserName'] == Name["Figo"]:
        if (msg['FromUserName'] == Name["葡萄挡油儿风"]) or (msg['FromUserName'] == Name["liankun1994"]) or (msg['FromUserName'] == Name["liankun1994"]):
            # 回复给好友
            url = "http://www.tuling123.com/openapi/api?key=462c1aa417ed45ebbc2d3d8b6c7ad80a&info="
            url = url + msg['Text']
            html = getHtmlText(url)
            message = re.findall(r'\"text\"\:\".*?\"', html)
            message = message[0].split(':')[1]
            reply = eval(message)
            return reply


if __name__ == '__main__':
    itchat.auto_login()

    # 获取自己的UserName
    friends = itchat.get_friends(update=True)[0:]
    Name = {}
    Nic = []
    User = []
    for i in range(len(friends)):
        Nic.append(friends[i]["NickName"])
        User.append(friends[i]["UserName"])
    for i in range(len(friends)):
        Name[Nic[i]] = User[i]
    itchat.run()

