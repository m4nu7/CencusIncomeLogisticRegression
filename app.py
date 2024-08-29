from flask import Flask,render_template,request,jsonify
from src.pipeline.prediction_pipeline import CustomData,PreditPipeline

application = Flask(__name__)
app = application

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data = CustomData(
            age=int(request.form.get("age")),
            education_num=int(request.form.get("education_num")),
            capital_gain=int(request.form.get("capital_gain")),
            capital_loss=int(request.form.get("capital_loss")),
            hours_per_week=int(request.form.get("hours_per_week")),
            workclass=request.form.get("workclass"),
            marital_status=request.form.get("marital_status"),
            occupation=request.form.get("occupation"),
            relationship=request.form.get("relationship"),
            race=request.form.get("race"),
            sex=request.form.get("sex"),
            native_country=request.form.get("native_country")
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PreditPipeline()
        pred = predict_pipeline.predict(final_new_data)
        
        print(pred)

        results = pred[0]

        return render_template("results.html", final_result = results)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
