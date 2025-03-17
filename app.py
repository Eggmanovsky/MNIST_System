from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/recognize")
def test():
    path = request.args.get("img_dir")
    if not path:
        return jsonify({"error": "img_dir is missing!"}), 400
    