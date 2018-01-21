import requests
import json
import os.path
from bs4 import BeautifulSoup
from time import sleep

import io
import urllib.request
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))

def get_thumbnail_urls(start=1, end=10):
  
  urls = []
  
  for n in range(start, end):
    url = "https://www.google.co.jp/search"
    
    querystring = {"yv":"2","tbm":"isch","q":"web+design+layout","ijn":str(n),"start":str(n*100),"asearch":"ichunk","async":"_id:rg_s,_pms:s"}
    
    headers = {
      'accept-encoding': "gzip, deflate",
      'accept-language': "ja,en-US;q=0.9,en;q=0.8",
      'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36",
      'x-chrome-uma-enabled': "1",
      'accept': "*/*",
      'referer': "https://www.google.co.jp/",
      'authority': "www.google.co.jp",
      'Cache-Control': "no-cache"
    }
    
    response = requests.request("GET", url, headers=headers, params=querystring)
    
    dom = response.json()[1][1]
    
    soup = BeautifulSoup(dom, "lxml")
    
    metadata = soup.findAll("div", {"class": "rg_meta"})
    
    for i in metadata:
      urls.append(json.loads(i.text)["ou"])
    
    sleep(0.1)
  
  return urls

if __name__ == '__main__':
  urls = get_thumbnail_urls()
  i = 0
  for url in urls:
    try:
      image_data = io.BytesIO(urllib.request.urlopen(url).read())
      img = Image.open(image_data)
      width, height = img.size
      img.crop((0, 0, width, width)) \
         .resize((128, 128)) \
         .convert("RGB") \
         .save(os.path.join(current_dir, "train_imgs", str(i) + ".png"))
      i = i + 1
    except Exception as e:
      print("An Exception occurred: " + str(e))