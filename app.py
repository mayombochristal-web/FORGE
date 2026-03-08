import os
import tempfile
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

# Importer la classe OracleEngine depuis le module où elle est définie
# (supposé être dans oracle_engine.py dans le même dossier)
from oracle_engine import OracleEngine

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Initialisation du moteur Oracle (unique pour toute l'application)
engine = OracleEngine()

# Page d'accueil simple (facultative)
INDEX_HTML = '''
<!doctype html>
<title>Oracle Mémoire</title>
<h1>Oracle Memory Engine</h1>
<ul>
    <li><a href="/stats">Statistiques</a></li>
    <li><form action="/learn" method="post">
        <h3>Apprendre un texte</h3>
        <textarea name="text" rows="5" cols="40"></textarea><br>
        <input type="submit" value="Apprendre">
    </form></li>
    <li><form action="/upload" method="post" enctype="multipart/form-data">
        <h3>Uploader un document</h3>
        <input type="file" name="file"><br>
        <input type="submit" value="Uploader">
    </form></li>
    <li><form action="/query" method="post">
        <h3>Posez une question</h3>
        <input type="text" name="question" size="40"><br>
        <input type="submit" value="Questionner">
    </form></li>
</ul>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/stats', methods=['GET'])
def stats():
    """Retourne les statistiques du moteur."""
    stats_data = engine.stats()
    return jsonify(stats_data)

@app.route('/learn', methods=['POST'])
def learn():
    """Apprend un texte envoyé dans le formulaire."""
    text = request.form.get('text', '')
    if not text:
        return jsonify({'error': 'Aucun texte fourni'}), 400
    nb_blocks = engine.learn(text, source='web_form')
    return jsonify({'message': f'{nb_blocks} bloc(s) appris avec succès'})

@app.route('/upload', methods=['POST'])
def upload():
    """Apprend un fichier uploadé."""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400

    # Sauvegarder temporairement pour que l'engine puisse le lire (simuler un objet fichier)
    # Mais notre engine utilise déjà file.read() directement, donc on peut passer l'objet file
    # Cependant, pour les types comme PDF, l'objet file doit être seekable ?
    # On va créer un fichier temporaire pour garantir la compatibilité avec PyPDF2, docx, etc.
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # On rouvre le fichier en mode binaire pour le passer à la méthode
        with open(tmp_path, 'rb') as f:
            # On a besoin de simuler un objet avec .type et .name
            class FakeFile:
                def __init__(self, fileobj, filename, content_type):
                    self.fileobj = fileobj
                    self.name = filename
                    self.type = content_type
                def read(self, *args, **kwargs):
                    return self.fileobj.read(*args, **kwargs)
            content_type = file.content_type or 'application/octet-stream'
            fake_file = FakeFile(f, file.filename, content_type)
            nb_blocks = engine.learn_document(fake_file)
    finally:
        os.unlink(tmp_path)

    return jsonify({'message': f'{nb_blocks} bloc(s) appris depuis le fichier {file.filename}'})

@app.route('/query', methods=['POST'])
def query():
    """Pose une question et retourne la réponse du moteur."""
    question = request.form.get('question', '')
    if not question:
        return jsonify({'error': 'Aucune question fournie'}), 400
    answer = engine.reason(question)
    return jsonify({'question': question, 'answer': answer})

if __name__ == '__main__':
    # Lancer l'application sur le port 5000 par défaut
    app.run(host='0.0.0.0', port=5000, debug=True)