import pandas as pd
import xgboost as xgb
import sys, os
from flask import Flask, request, render_template
from scipy.stats import zscore
import pickle

app = Flask(__name__)

# Load the model from the notebook
xgb_model = xgb.Booster(model_file='best_model.model')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction_result', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        features = request.form.to_dict()
        print(features)

        try:
            prediction = get_prediction(features)
            print(prediction)
            return render_template('prediction_result.html', prediction=prediction)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            return f'Error: {exc_type}\n{fname}\n{exc_tb.tb_lineno}\n{str(e)}'

def preprocess(features):
    typed_features = {k: float(v) for k, v in features.items()}
    df = pd.DataFrame(typed_features, index=[0])
    df['modular ratio / interlinear spacing'] = df['modular ratio'] / df['interlinear spacing']

    # Need to apply z-normalization on data as it's been applied on the training data
    z_df = zscore(df, axis=1)

    return z_df

def get_prediction(features):
    clean_df = preprocess(features)
    return xgb_model.predict(clean_df)

if __name__ == '__main__':
    app.run(debug=True)