from bs4 import BeautifulSoup
import requests
 
def download(img_url,headers,n):
    req = requests.get(img_url, headers=headers)
    name = '%s'%n+'='+img_url[-15:]
    path = r'I:\\python pro\\DeepLearningByPython\\clothing_retrieve\\image\\'
    file_name = path + '\\' + name
    f = open(file_name, 'wb')
    f.write(req.content)
    f.close
 
def parses_picture(url,headers,n):
    url = r'http://desk.zol.com.cn/' + url
    img_req = requests.get(url, headers=headers)
    img_req.encoding = 'gb2312'
    html = img_req.text
    bf = BeautifulSoup(html, 'lxml')
    try:
        img_url = bf.find('div', class_='photo').find('img').get('src')
        download(img_url,headers,n)
        url1 = bf.find('div',id='photo-next').a.get('href')
        parses_picture(url1,headers,n)
    except:
        print(u'第%s图片集到头了'%n)
 
if __name__=='__main__':
    url='http://desk.zol.com.cn/dongman/huoyingrenzhe/'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36"}
    req = requests.get(url=url, headers=headers)
    req=requests.get(url=url,headers=headers)
    req.encoding = 'gb2312'
    html=req.text
    bf=BeautifulSoup(html,'lxml')
    targets_url=bf.find_all('li',class_='photo-list-padding')
    n=1
    for each in targets_url:
        url = each.a.get('href')
        parses_picture(url,headers,n)
        n=n+1