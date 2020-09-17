import sqlite3

with sqlite3.connect('coins.db') as db:
    cursor = db.cursor()
    query = """CREATE TABLE IF NOT EXISTS expenses(name TEXT, discriptions TEXT) """
    cursor.execute(query)
