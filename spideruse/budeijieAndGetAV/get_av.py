# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 09:16:21 2018

@author: Figo
"""
import requests
import re
import os
import time
import urllib.request as urlreq

#获取网页源代码
def get_response(url):
    response = requests.get(url)
    response.encoding = 'utf-8' #解决中文乱码
    return response.text

#把包含视频的内容返回，参数为网页源代码
def get_vedio_content(html,tpye_num,total_class='move'):
    # . 匹配任意字符
    # * 匹配匹配前面的元字符的0到n次
    # ? 匹配前面的子表达式0次或1次
    # .* 贪婪模式 
    # .*? 非贪婪模式（最小匹配）
    # re.S 正则表达式可以匹配到换行
    reg = re.compile(r'(<li><a href="/' + total_class + '/' + str(tpye_num) + '/.*?</h3>)',re.S)
    x = re.findall(reg,html)
    return re.findall(reg,html)

# 由vedio_content 获取视频名字
def get_vedio_name(response):
    reg = r'<h3>(.*?)</h3>'
    return re.findall(reg,response)

# 由vedio_content 获取单个视频的网页的 url尾部
def get_end_path(response,type_num,total_class='move'):
    reg = r'<li><a href="/' + total_class + '/' + str(type_num) + '/(.*?)"'
    return re.findall(reg,response)

#获取视频下载url
def get_vedio_path(response):
    reg = r'<td bgcolor="#F4F9FD"><a target="_blank" href="(.*?)" ><b><font color="#0000ff">'
    return re.findall(reg,response)
    
#下载视频并保存    
def download_vedio(vedio_url,vedio_name,vedio_type_name):
    #path先拆分后拼接
    vedio_name = ''.join(vedio_name.split())
    print(vedio_name)
    nowtime = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
    save_path = path_base + vedio_type_name + '/{}_{}.mp4'.format(vedio_name,nowtime)
    #    content = get_reponse(vedio_url)  #另一种下载方法
    #    with open(path,'wb') as f:
    #        f.write(content)
    if not os.path.exists(save_path):
        try:
            urlreq.urlretrieve(vedio_url,save_path) #下载urlretrieve
            print('下载成功 !!')
        except:
            print('下载失败 !!!')
    else:
        print('该文件已存在 !!!')

# 运行一个网页
def run_curr_web(vedio_content,vedio_type_name,type_num,total_class):
    # 当前页计数器
    count_curr_page_vedio = 0
    #在每页18个视频中，逐个处理每个视频内容
    for one_vedio_content in vedio_content:
        try:
            count_curr_page_vedio = count_curr_page_vedio + 1
            print('  ***** 正在处理第 ',count_curr_page_vedio,' 个视频 ****')
            # 获得视频名 vedio_name 以及 单个视频页的尾部url end_path
            #正则表达式匹配到的是一个list，因此要加[0]，取 list[0] 中的内容
            vedio_name = get_vedio_name(one_vedio_content)[0]
            end_path = get_end_path(one_vedio_content,type_num,total_class)[0]
            if end_path != '':
                #得到单个视频网页的地址
                one_path = web_base_url + '/' + total_class + '/' + str(type_num) + '/' + end_path
                #获得视频的下载地址 vedio_path
                vedio_path = get_vedio_path(get_response(one_path))[0]
                download_vedio(vedio_path,vedio_name,vedio_type_name)
        except:
            continue

# 对给定的列表 start_urls，进入 start_urls 的每一页进行处理
def main(start_urls,vedio_type_name,type_num,total_class='move'):
    # 页数 计数器
    count_page = 0
    for start_url in start_urls: #进入每一页
        try:
            count_page = count_page + 1
            print('--------------- 当前进行第 ',count_page,' 页 -----------------')
            #获得 start_url 的整个html
            html_text = get_response(start_url)
            # 获得 html_text 中 包含视频的那部分 html
            # vedio_content 是一个列表，正则表达式匹配到的内容列表，每页18个视频内容
            vedio_content = get_vedio_content(html_text,vedio_type,total_class)
            # 进入run_curr_web() 处理每页的18个视频内容
            run_curr_web(vedio_content,vedio_type_name,type_num,total_class)
        except:
            continue


if __name__ == '__main__':
    web_base_url = 'https://www.9229df.com'
    path_base = 'D:/internet spider/game5/game9/game7/game0/game0/game9/game6/'

    # vedio_type 范围是 1-7
    vedio_type = 7
    '''
    vedio_type = 1 代表 yzwm
    vedio_type = 3 代表 zptp
    vedio_type = 5 代表 sjxz
    vedio_type = 7 代表 sjsp
    '''
    type_dict = {1:'yzwm',3:'zptp',5:'sjxz',7:'sjsp'}
    # max_page 最大网页数目，这个数目每天都会变
    start_page = 2 # >= 2
    max_page = 10

    
    #列表生成器，生成 1到max_page个网页url
    start_urls = [web_base_url + '/move/' + str(vedio_type) +
                  '/index_{}.html'.format(i) for i in range(start_page,max_page)]
    #第一页比较特殊，最后插入
    start_urls.insert(0,web_base_url + '/move/' + str(vedio_type) +  '/index.html')

    path = path_base + type_dict[vedio_type]
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path) 
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")
    
    main(start_urls,type_dict[vedio_type],vedio_type)
    
    

