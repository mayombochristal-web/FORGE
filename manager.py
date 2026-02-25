import sqlite3

class DBManager:
    def __init__(self):
        # Utilisation de SQLite pour le stockage persistant
        self.conn = sqlite3.connect("ttu_memory.db", check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS crystals 
            (sig TEXT PRIMARY KEY, content TEXT, stability FLOAT)''')
        self.conn.commit()

    def check_memory(self, signature):
        self.cursor.execute("SELECT content FROM crystals WHERE sig=?", (signature,))
        return self.cursor.fetchone()

    def save_crystal(self, sig, content):
        try:
            self.cursor.execute("INSERT INTO crystals VALUES (?,?,?)", (sig, content, 0.98))
            self.conn.commit()
        except: 
            pass # Évite les erreurs si la signature existe déjà
