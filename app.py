from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

numerical_cols = ["tv", "radio", "newspaper"]
columns = numerical_cols


@app.route("/")
def index():
    return render_template("index.html", col=columns, enumerate=enumerate)


@app.route("/predict", methods=["GET", "POST"])
def new_prediction():
    tv = request.form["tv"]
    radio = request.form["radio"]
    newspaper = request.form["newspaper"]
    data = CustomData(tv, radio, newspaper)
    df = data.get_data_as_dataframe()
    model = PredictPipeline()
    prediction = model.predict(df)
    return render_template("result.html", predict=round(prediction[0], 1))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
