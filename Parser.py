import requests
from bs4 import BeautifulSoup
import urllib.request

URL = 'https://ru.ucoin.net/catalog/?country=ussr'
HEADERS = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36', 'accept':'*/*'}

def get_html(url, params = None):
    r = requests.get(url, headers = HEADERS, params = params)
    return r

def get_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('table')



    coins = []
    for item in items:
        img2 = item.find('img').get('src')
        img2 = img2.replace('-1c', '-2c')
        dis = item.find('td', class_='mgray-11').get_text(strip=True)
        dis = dis.replace('\xa0ø\xa0', ' ')
        dis = dis.replace('Y#', ', ')
        coins.append(dict(title = item.find('tr', class_='marked-0').get_text(strip=True),
                          image1 = item.find('img').get('src'),
                          image2 = img2,
                          discr = dis
                          ))
    print(coins)

def parse():
    html = get_html(URL)
    if html.status_code == 200:
        get_content(html.text)
        pass
    else:
        print('ERROR')
parse()
