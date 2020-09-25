import sqlite3 as lite
import sys
import os
import urllib
import urllib.request


coin = lite.connect('coins.db')

#os.mkdir("Images")



with coin:
    cur = coin.cursor()
    cur.execute("SELECT * FROM query")
    rows = cur.fetchall()
    for row in rows:
        #os.mkdir(row[0])
        os.chdir("C:\\Users\\Nikolay\\Desktop\\PrPr\\Parser\\images")
        url = row[10]
        if url != '-':
            print(url)
            print(row[0])
            img = urllib.request.urlopen(url).read()
            out = open('class ' + row[0] + '_8' + ".jpg", "wb")
            out.write(img)
            out.close
        else:
            continue

