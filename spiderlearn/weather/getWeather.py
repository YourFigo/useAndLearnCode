import requests
import json
import os
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


# 写之前，先检验文件是否存在，存在就删掉
if os.path.exists("dest.txt"):
  os.remove("dest.txt")
file_write_obj = open("dest.txt", 'w')

forecast_urls = ['https://free-api.heweather.com/s6/weather/forecast?'+'key='+key+'&location='+x for x in city_top_cid]
# print(forecast_url)
for url in forecast_urls:
    forecast = json.loads(requests.get(url).text)
    forecast_city = forecast["HeWeather6"][0]["basic"]
    forecast_city = str(forecast_city)
    forecast_daily = forecast["HeWeather6"][0]["daily_forecast"]
    forecast_daily = str(forecast_daily)
    print(forecast_daily)
    file_write_obj.writelines(forecast_city)
    file_write_obj.writelines(forecast_daily)
    file_write_obj.write('\n')
file_write_obj.close()

