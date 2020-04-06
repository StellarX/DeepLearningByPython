#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import requests
from pyquery import PyQuery as pq
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'
}


# 请求网页 获取源码
def start_request(url):
    r = requests.get(url, headers=headers)
    # 这个网站页面使用的是GBK编码 这里进行编码转换
    r.encoding = 'GBK'
    html = r.text
    return html


# 解析网页 获取图片
def parse(text):
    doc = pq(text)
    # 锁定页面中的img标签
    images = doc('div.list ul li img').items()
    x = 0
    for image in images:
        # 获取每一张图片的链接
        img_url = image.attr('src')
        # 获得每张图片的二进制内容
        img = requests.get(img_url, headers=headers).content
        # 定义要存储图片的path
        path = "I:\\python pro\\DeepLearningByPython\\clothing_retrieve\\image\\" + str(x) + ".jpg"
        # 将图片写入指定的目录 写入文件用"wb"
        with open(path, 'wb') as f:
            f.write(img)
            time.sleep(1)
            print("正在下载第{}张图片".format(x))
            x += 1
    print("写入完成")


def main():
    # url = "http://www.netbian.com"
    url = "https://cn.bing.com/images/search?q=%e6%b7%98%e5%ae%9d%e7%bd%91+%e8%a1%a3%e6%9c%8d%e5%ba%97&FORM=RESTAB"
    text = start_request(url)
    parse(text)


if __name__ == "__main__":
    main()
