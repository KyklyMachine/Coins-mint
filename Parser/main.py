import requests
from bs4 import BeautifulSoup
import urllib.request
import sqlite3


URL = 'https://us.ucoin.net/catalog/?country=ussr'
HEADERS = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36', 'accept':'*/*'}

def get_html(url, params = None):
    r = requests.get(url, headers = HEADERS, params = params)
    return r

def get_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('table')
    with sqlite3.connect('coins.db') as db:
        cursor = db.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS query (title TEXT, discriptions TEXT, image1 TEXT, image2 TEXT"
                       "image3 TEXT, image4 TEXT, image5 TEXT, image6 TEXT, image7 TEXT, image8 TEXT, image9 TEXT"
                       "image10 TEXT, image11 TEXT, image12 TEXT, image13 TEXT, image14 TEXT, image15 TEXT, image16 TEXT)")
        for item in items:
            img2 = item.find('img').get('src')
            img2 = img2.replace('-1c', '-2c')
            disC = item.find('td', class_='mgray-11').get_text(strip=True)
            disC = disC.replace('\xa0ø\xa0', ' ')
            disC = disC.replace('Y#', ', ')
            titleC = item.find('tr', class_='marked-0').get_text(strip=True)
            image1C = item.find('img').get('src')
            image2C = img2

            nou = '-'

            cursor.execute("INSERT INTO query VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (titleC, disC, image1C, image2C, nou, nou, nou, nou, nou, nou, nou, nou, nou, nou, nou, nou) )
        db.commit()


def parse():
    html = get_html(URL)
    if html.status_code == 200:
        get_content(html.text)
        pass
    else:
        print('ERROR')
parse()
