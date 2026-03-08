import os
import tempfile
import sys
import pathlib

# Ajoute le dossier contenant app.py au PYTHONPATH
ROOT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))
from oracle_engine import OracleEngine
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Moteur Oracle global
engine = OracleEngine()

INDEX_HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Oracle Mémoire</title>
</head>
<body>
<h1>Oracle Memory Engine</h1>
<ul>
  <li><a href="/stats">Statistiques</a></li>
  <li>
    <form action="/learn" method="post">
      <h3>Apprendre un texte</h3>
      <textarea name="text" rows="5" cols="60"></textarea><br>
      <input type="submit" value="Apprendre">
    </form>
  </li>
  <li>
    <form action="/upload" method="post" enctype="multipart/form-data">
      <h3>Uploader un document</h3>
      <input type="file" name="file"><br>
      <input type="submit" value="Uploader">
    </form>
  </li>
  <li>
    <form action="/query" method="post">
      <h3>Posez une question</h3>
      <input type="text" name="question" size="60"><br>
      <input type="submit" value="Questionner">
    </form>
  </li>
</ul>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(engine.stats())


@app.route("/learn", methods=["POST"])
def learn():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Aucun texte fourni"}), 400

    nb_blocks = engine.learn(text, source="web_form")
    return jsonify({"message": f"{nb_blocks} bloc(s) appris avec succès"})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Nom de fichier vide"}), 400

    filename = secure_filename(file.filename)

    # Fichier temporaire pour compatibilité avec PyPDF2, docx, etc.
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:

            class FakeFile:
                def __init__(self, fileobj, filename, content_type):
                    self.fileobj = fileobj
                    self.name = filename
                    self.type = content_type

                def read(self, *args, **kwargs):
                    return self.fileobj.read(*args, **kwargs)

            content_type = file.content_type or "application/octet-stream"
            fake_file = FakeFile(f, filename, content_type)
            nb_blocks = engine.learn_document(fake_file)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return jsonify(
        {"message": f"{nb_blocks} bloc(s) appris depuis le fichier {filename}"}
    )


@app.route("/query", methods=["POST"])
def query():
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "Aucune question fournie"}), 400

    answer = engine.reason(question)
    return jsonify({"question": question, "answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)