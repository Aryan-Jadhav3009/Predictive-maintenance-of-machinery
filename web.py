import sklearn
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('c:\\Users\\Aryan\\Desktop\\College\\ml ds\\Predictive Maintenance\\model.pkl', 'rb'))
from flask import Flask, render_template, request
import numpy as np
import joblib  # Assuming you used joblib to save your model

app = Flask(__name__)

# Load the trained model


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the HTML form
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        # Make predictions using the model
        prediction = model.predict(final_features)

        # You can do more processing with the prediction if needed

        return render_template('index.html', prediction_text=f'State of equipment is: {prediction[0]}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)