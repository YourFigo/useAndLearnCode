import pymongo
import requests
import json

mongoClient = pymongo.MongoClient('127.0.0.1',27017)
book_weather = mongoClient['weather']
sheet_weather = book_weather['sheet_weather_top']

# sheet_weather.remove()
# for i in sheet_weather.find():
#     print(i)
# print('end')

# 查询北京天气
# for item in sheet_weather.find({'HeWeather6.basic.location':'北京'}):
#     print(item)

# 显示所有城市的最高和最低温度
# 查找所有元素，并且_id不显示，只显示HeWeather6。
# for item in sheet_weather.find({},{'_id':0,'HeWeather6':1}):
#     item = item['HeWeather6'][0]
#     city = item['basic']['location']
#     daily_forecast1 = item['daily_forecast'][0]
#     date = daily_forecast1['date']
#     tmp_max = daily_forecast1['tmp_max']
#     tmp_min = daily_forecast1['tmp_min']
#     print('城市:{},日期:{},最高温度:{},最低温度:{}\n'.format(city,date,tmp_max,tmp_max))

# 查找最低温度大于5的城市名
for item in sheet_weather.find():
    # 一共只有三天数据
    for i in range(3):
        tmp = item['HeWeather6'][0]['daily_forecast'][i]['tmp_min']
        # update_one() 更新指定的一条数据，第一个参数为要更新的条件，第二个参数表示要更新的信息，
        # $set是MongoDB的一个修改器，用于指定一个键并更新键值。除此之外还有：$inc、$unset、$push等。
        # $inc 用于为某个值为数字型的键进行增减操作
        # $unset 用于删除键
        # $push 向文档的某个数组类型的键添加一个数组元素，不过滤重复的数据，添加时，若键存在，要求键值类型必须是数组，若键不存在则创建数组类型的键。
        sheet_weather.update_one({'_id':item['_id']},{'$set':{'HeWeather6.0.daily_forecast.{}.tmp_min'.format(i):int(tmp)}})
num = 0
# $lt、$lte、$gt、$gte 分别表示： <、<=、>、>=
for item in sheet_weather.find({'HeWeather6.daily_forecast.tmp_min':{'$gt':5}}):
    num = num + 1
    print(item['HeWeather6'][0]['basic']['location'])
print(num)