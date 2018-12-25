# coding=utf-8
import random
import requests
import hashlib
import time


# 获取验证码图片URL
def get_verify_image_url():
    try:
        headers = {
            'Host': 'cache.neea.edu.cn',
            'Referer': 'http://cet.neea.edu.cn/cet/'
        }
        url = "http://cache.neea.edu.cn/Imgs.do?c=CET&ik=123456789123456&t=" + str(random.random())
        response = requests.get(url=url, headers=headers)
        text = response.text
        return text[text.index('http'):text.index('png') + 3]
    except Exception:
        return get_verify_image_url()


# 保存图片并返回相应MD5值
def load_image(path):
    try:
        url = get_verify_image_url()
        response = requests.get(url)
        img = response.content
        with open(path, 'wb') as f:
            f.write(img)
        # MD5
        my_hash = hashlib.md5()
        my_hash.update(img)
        return my_hash.hexdigest(), url[url.index('imgs') + 5:url.index('.png')]
    except Exception:
        return load_image(path)


# 保存图片
def load_image2(path, url):
    try:
        response = requests.get(url)
        img = response.content
        with open(path, 'wb') as f:
            f.write(img)
    except Exception:
        load_image(path)


# main
if __name__ == '__main__':
    with open('captcha.csv', 'r') as csv_file:
        path = 'dataset/train/'
        line = csv_file.readline()
        i = 0
        while line:
            names = line.split(",")
            image_path = path + str(i) + '_' + names[1].strip() + '.png'
            url = 'http://cet.neea.edu.cn/imgs/' + names[0].strip() + '.png'
            print(image_path, url)
            load_image2(image_path, url)
            line = csv_file.readline()
            i += 1
