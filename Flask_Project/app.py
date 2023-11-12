from flask import Flask , render_template , request , jsonify
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load your saved Keras model
model = load_model("ANN-Regression_model.h5")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        feature_names = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ]
        features = [float(request.form[feature]) for feature in feature_names]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        result = round(prediction[0][0], 2)
        return render_template('index.html', result=result)
    return render_template('index.html', result=None)




@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)