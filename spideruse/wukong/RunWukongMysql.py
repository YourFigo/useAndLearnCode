import wukongByTypeMysql
import traceback

keywords = ['科技', '美食', '军事', '财经','动漫' ,'汽车' ,'热门' ,'国际', '育儿', '旅游',
     '三农', '文化', '数码', '家居', '时尚', '科学', '游戏', '历史','收藏', '健康',
     '心理', '电影', '教育', '宠物', '职场', '娱乐', '社会', '体育']

keywords = ['心理', '电影', '教育', '宠物', '职场', '娱乐', '社会', '体育']
nums = [500,500,500,500,500,500,500,500]

num = 0
for x in keywords:
    print("=====================第 {0} 个关键字：{1}=====================".format(num + 1, x))
    try:
        keyword = x
        maxTimes = nums[num]
        num = num + 1
        wukongByTypeMysql.wukongOpearte(keyword, maxTimes)
    except:
        error = traceback.format_exc()
        with open('error.txt','w') as f:
            f.write(error)
        print(error)