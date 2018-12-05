import requests
import json
import pymongo

city_top_url = 'https://search.heweather.com/top?'
key = '07d70399294649a8ae8917aadbeaac1c'
group = 'cn'
number = 50

city_top_url = city_top_url + 'key=' + key + '&group=' + group + '&number=' + str(number)
# print(city_top_url)
city_top_response = requests.get(city_top_url)
city_top_response = city_top_response.text
city_top_response = json.loads(city_top_response)
# print(city_top_response)
city_top = city_top_response["HeWeather6"][0]["basic"]
city_top_num = len(city_top)
# print(city_top_num)
city_top_cid = []
for i in city_top:
    city_top_cid.append(i["cid"])
# print(city_top_cid)
city_status = city_top_response["HeWeather6"][0]["status"]
# print(city_status)

mongoClient = pymongo.MongoClient('127.0.0.1',27017)
book_weather = mongoClient['weather']
sheet_weather = book_weather['sheet_weather_top']

forecast_urls = ['https://free-api.heweather.com/s6/weather/forecast?'+'key='+key+'&location='+x for x in city_top_cid]
# print(forecast_url)
for url in forecast_urls:
    forecast = json.loads(requests.get(url).text)
    forecast_city = forecast["HeWeather6"][0]["basic"]["location"]
    print(forecast_city)
    sheet_weather.insert_one(forecast)
    # forecast_city = str(forecast_city)
    # forecast_daily = forecast["HeWeather6"][0]["daily_forecast"]
    # forecast_daily = str(forecast_daily)
    # print(forecast_daily)
for i in sheet_weather.find():
    print(i)