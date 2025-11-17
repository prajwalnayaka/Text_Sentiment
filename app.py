from flask import Flask, jsonify, request, render_template
from inference import infer
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/about", methods=["GET"])
def about():
    return jsonify({"message": "This API is for our emotion model."})


@app.route("/accept", methods=["POST"])
def accept_text():
    data = request.get_json()
    user_text = data.get('text')
    if not user_text:
        return jsonify({"error": "No text was provided"}), 400
    return infer(user_text)

if __name__ == "__main__":
    app.run(debug=True)