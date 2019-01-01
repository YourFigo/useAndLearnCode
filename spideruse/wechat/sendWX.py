import requests
import itchat
import json
from threading import Timer

key = '07d70399294649a8ae8917aadbeaac1c'

# 获取金山词霸每日一句，英文和翻译
def get_news():
    url = "http://open.iciba.com/dsapi"
    r = requests.get(url)
    contents = r.json()['content']
    return contents

def get_weather(loc,name="老哥"):
    forecast_urls = 'https://free-api.heweather.com/s6/weather/forecast?' + 'key=' + key + '&location=' + loc

    forecast = json.loads(requests.get(forecast_urls).text)

    forecast_daily = forecast["HeWeather6"][0]["daily_forecast"]
    date = forecast_daily[0]["date"]
    cond_txt_n = forecast_daily[0]["cond_txt_n"]
    tmp_max = forecast_daily[0]["tmp_max"]
    tmp_min = forecast_daily[0]["tmp_min"]
    out = "{},今天是:{},{} 今天 {},最高温度:{},最低温度:{}".format(name,date,loc,cond_txt_n,tmp_max,tmp_min)

    date2 = forecast_daily[1]["date"]
    cond_txt_n2 = forecast_daily[1]["cond_txt_n"]
    tmp_max2 = forecast_daily[1]["tmp_max"]
    tmp_min2 = forecast_daily[1]["tmp_min"]
    out2 = "{},明天是:{},{} 明天 {},最高温度:{},最低温度:{}".format(name, date2, loc, cond_txt_n2, tmp_max2, tmp_min2)

    return out,out2


# 发送消息
def send_news():
    try:
        # itchat.auto_login()  # 会弹出网页二维码，扫描即可，登入你的微信账号，True保持登入状态

        names = ['尚矫健','Dear Snow']
        names = ['尚矫健']

        for name in names:
            friend = itchat.search_friends(name=name)
            myfriend = friend[0]["UserName"]

            message1 = get_weather('天津')
            message2 = str(get_news())
            message3 = "来自 Figo"

            if name == 'Dear Snow':
                message1 = get_weather('天津','亲爱的')
                message3 = "来自 你亲爱的Figo"
            elif name == '尚矫健':
                message1 = get_weather('晋中')

            print(message1,message2,message3)
            itchat.send(message1[0], toUserName=myfriend)
            itchat.send(message1[1], toUserName=myfriend)
            itchat.send(message2, toUserName=myfriend)
            itchat.send(message3, toUserName=myfriend)

        Timer(86400, send_news).start()  # 每隔86400秒发送一次，也就是每天发一次
    except:
        message4 = "出错~~"
        me = itchat.search_friends(name='Figo')
        figo = me[0]["UserName"]
        itchat.send(message4, toUserName=figo)


if __name__ == "__main__":
    send_news()