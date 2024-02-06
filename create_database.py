import sqlite3

def create_students_table(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB,
            input_image_path TEXT
        )
    ''')

def create_table():
    database_path = 'database.db'
    # Connect to the database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Create the students table if it doesn't exist
    create_students_table(cursor)

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    create_table()
