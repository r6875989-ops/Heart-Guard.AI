from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
from sqlalchemy import func
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import io
from flask import send_file
from flask import jsonify
from openai import OpenAI
import shap
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from datetime import datetime
import os
from dotenv import load_dotenv


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Load ML files
model = joblib.load("best_scaled_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------- DATABASE MODELS ---------------- #

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    age = db.Column(db.Integer)
    result = db.Column(db.String(50))
    probability = db.Column(db.Float)
    diet_plan = db.Column(db.Text)
    exercise_plan = db.Column(db.Text)
    emergency = db.Column(db.Boolean)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    full_name = db.Column(db.String(150))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    phone = db.Column(db.String(20))
    email = db.Column(db.String(120))

    blood_group = db.Column(db.String(10))
    chronic_disease = db.Column(db.String(200))
    allergies = db.Column(db.String(200))
    medications = db.Column(db.String(200))

    aadhaar_masked = db.Column(db.String(20))
    pan_masked = db.Column(db.String(20))
    ayushman_masked = db.Column(db.String(30))

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- ROUTES ---------------- #

@app.route("/")
def landing():

    if current_user.is_authenticated:
        return redirect(url_for("home"))

    return render_template("landing.html")


@app.route("/dashboard")
@login_required
def home():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    return render_template("index.html", predictions=user_predictions)

def generate_diet_plan(age, cholesterol, bp, risk):

    if risk == "High Risk":

        if cholesterol > 240:
            return """
🥗 AI Heart Diet Plan
• Oatmeal breakfast
• Avoid fried foods
• Eat walnuts & almonds
• Olive oil instead of butter
• Fish twice per week
• Green vegetables daily
"""

        else:
            return """
🥗 AI Balanced Heart Diet
• Whole grains
• Fruits twice daily
• Low fat dairy
• Reduce salt
"""

    else:
        return """
🥗 Maintenance Diet
• Balanced carbs & protein
• Fruits & vegetables
• Moderate sugar & salt
• Stay hydrated
"""


@app.route("/predict", methods=["POST"])
@login_required
def predict():

    age = int(request.form['age'])

    raw_input = {
        'Age': age,
        'RestingBP': int(request.form['resting_bp']),
        'Cholesterol': int(request.form['cholesterol']),
        'FastingBS': int(request.form['fasting_bs']),
        'MaxHR': int(request.form['max_hr']),
        'Oldpeak': float(request.form['oldpeak']),
        'Sex_' + request.form['sex']: 1,
        'ChestPainType_' + request.form['chest_pain']: 1,
        'RestingECG_' + request.form['resting_ecg']: 1,
        'ExerciseAngina_' + request.form['exercise_angina']: 1,
        'ST_Slope_' + request.form['st_slope']: 1
    }

    input_df = pd.DataFrame([raw_input])
    

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1] * 100
    explainer = shap.LinearExplainer(model, input_df)
    shap_values = explainer.shap_values(input_df)

    coefficients = model.coef_[0]

    feature_importance = dict(
        zip(expected_columns, coefficients)
    )

    # Top important features
    top_features = dict(
        sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    )

    print("Top Features:", top_features)

    result = "High Risk" if prediction == 1 else "Low Risk"
    emergency = False

    if probability > 80:
        emergency = True
    # ---------------- DIET & EXERCISE SUGGESTION ---------------- #

    if result == "High Risk":

     diet_plan = generate_diet_plan(
    age,
    raw_input['Cholesterol'],
    raw_input['RestingBP'],
    result
)

     exercise_plan = """🏃 Controlled Exercise Plan:
        • 30 min brisk walking daily
        • Light yoga (Anulom Vilom, Pranayama)
        • Avoid heavy gym workouts
        • Stretching exercises
        • Consult cardiologist before intense activity"""

    else:

     diet_plan = generate_diet_plan(
    age,
    raw_input['Cholesterol'],
    raw_input['RestingBP'],
    result
)

     exercise_plan = """🏋 Active Lifestyle Plan:
        • 30-45 min cardio 5 days/week
        • Light strength training
        • Cycling / swimming
        • Yoga for flexibility
        • Maintain healthy BMI"""
    print("Saving Diet:", diet_plan)
    print("Saving Exercise:", exercise_plan)

    # Save to database
    new_prediction = Prediction(
    age=age,
    probability=probability,
    result=result,
    diet_plan=diet_plan,
    exercise_plan=exercise_plan,
    emergency=emergency,
    user_id=current_user.id
    
    )
    db.session.add(new_prediction)
    db.session.commit()

    return redirect(url_for("home"))

@app.route("/view_suggestion/<int:id>")
@login_required
def view_suggestion(id):

    prediction = Prediction.query.get_or_404(id)

    if prediction.user_id != current_user.id:
        return "Unauthorized"

    # AI feature importance
    feature_data = {
        "Age": 0.30,
        "RestingBP": 0.22,
        "Cholesterol": 0.25,
        "MaxHR": 0.15,
        "Oldpeak": 0.08
    }

    return render_template(
        "suggestion.html",
        prediction=prediction,
        feature_data=feature_data
    )

@app.route("/delete_suggestion/<int:id>")
@login_required
def delete_suggestion(id):
    prediction = Prediction.query.get_or_404(id)

    if prediction.user_id != current_user.id:
        return "Unauthorized"

    prediction.diet_plan = None
    prediction.exercise_plan = None
    db.session.commit()

    return redirect(url_for("home"))


@app.route("/admin")
@login_required
def admin():

    if current_user.username != "Rajesh45":
        return "Access Denied"

    total_users = User.query.count()
    total_predictions = Prediction.query.count()

    high_risk = Prediction.query.filter_by(result="High Risk").count()
    low_risk = Prediction.query.filter_by(result="Low Risk").count()

    risk_percentage = (high_risk / total_predictions * 100) if total_predictions else 0

    recent_predictions = Prediction.query.order_by(Prediction.id.desc()).limit(10).all()

    # Monthly predictions
    monthly_data = db.session.query(
        func.strftime("%m", Prediction.id),
        func.count(Prediction.id)
    ).group_by(func.strftime("%m", Prediction.id)).all()

    monthly_labels = [m[0] for m in monthly_data]
    monthly_values = [m[1] for m in monthly_data]

    # Age group analytics
    age_data = db.session.query(
        Prediction.age,
        func.count(Prediction.id)
    ).filter_by(result="High Risk").group_by(Prediction.age).all()

    age_labels = [str(a[0]) for a in age_data]
    age_values = [a[1] for a in age_data]

    # Cholesterol insights
    chol_data = db.session.query(
        Prediction.age,
        func.avg(Prediction.probability)
    ).group_by(Prediction.age).all()

    chol_labels = [str(c[0]) for c in chol_data]
    chol_values = [round(c[1],2) for c in chol_data]

    return render_template(
        "admin.html",
        total_users=total_users,
        total_predictions=total_predictions,
        high_risk=high_risk,
        low_risk=low_risk,
        risk_percentage=risk_percentage,
        recent_predictions=recent_predictions,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values,
        age_labels=age_labels,
        age_values=age_values,
        chol_labels=chol_labels,
        chol_values=chol_values
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        hashed_password = generate_password_hash(request.form['password'])
        new_user = User(
            username=request.form['username'],
            password=hashed_password
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Registration Successful! Please login.")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for("home"))
        flash("Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("landing"))

@app.route("/delete_prediction/<int:id>")
@login_required
def delete_prediction(id):
    prediction = Prediction.query.get_or_404(id)

    if prediction.user_id != current_user.id:
        return "Unauthorized"

    db.session.delete(prediction)
    db.session.commit()

    return redirect(url_for("home"))

@app.route("/delete_all")
@login_required
def delete_all():
    Prediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return redirect(url_for("home"))

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    profile = UserProfile.query.filter_by(user_id=current_user.id).first()

    if request.method == "POST":

        if not profile:
            profile = UserProfile(user_id=current_user.id)

        profile.full_name = request.form["full_name"]
        profile.age = request.form["age"]
        profile.gender = request.form["gender"]
        profile.phone = request.form["phone"]
        profile.email = request.form["email"]

        profile.blood_group = request.form["blood_group"]
        profile.chronic_disease = request.form["chronic_disease"]
        profile.allergies = request.form["allergies"]
        profile.medications = request.form["medications"]

        profile.aadhaar_masked = request.form["aadhaar"]
        profile.pan_masked = request.form["pan"]
        profile.ayushman_masked = request.form["ayushman"]

        db.session.add(profile)
        db.session.commit()

        return redirect(url_for("profile"))

    return render_template("profile.html", profile=profile)




@app.route("/download_report/<int:id>")
@login_required
def download_report(id):

    prediction = Prediction.query.get_or_404(id)

    buffer = io.BytesIO()
    styles = getSampleStyleSheet()

    elements = []

    # ================= HEADER ================= #

    header_style = styles['Title']
    header_style.textColor = colors.darkblue

    elements.append(Paragraph("HeartGuard AI Medical Report", header_style))
    elements.append(Paragraph("AI Powered Cardiovascular Risk Assessment System", styles['Normal']))
    elements.append(Spacer(1,20))

    # ================= PATIENT DETAILS ================= #

    patient_data = [
        ["Patient Age", prediction.age],
        ["Prediction Result", prediction.result],
        ["Risk Probability", f"{round(prediction.probability,2)} %"],
        ["Report Generated On", datetime.now().strftime("%d %B %Y")]
    ]

    patient_table = Table(patient_data, colWidths=[220,250])

    patient_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1),colors.whitesmoke),
        ('TEXTCOLOR',(0,0),(-1,-1),colors.black),
        ('GRID',(0,0),(-1,-1),1,colors.grey),
    ]))

    elements.append(Paragraph("Patient Information", styles['Heading2']))
    elements.append(Spacer(1,10))
    elements.append(patient_table)

    elements.append(Spacer(1,25))

    # ================= AI RISK ANALYSIS ================= #

    elements.append(Paragraph("AI Risk Analysis", styles['Heading2']))
    elements.append(Spacer(1,10))

    if prediction.result == "High Risk":

        analysis = """
The AI model detected patterns associated with higher cardiovascular risk.
Possible contributing factors may include elevated cholesterol levels,
abnormal blood pressure, reduced heart rate capacity, and age related
cardiovascular stress.
"""

        risk_color = colors.red
        risk_summary = "High cardiovascular risk detected. Immediate lifestyle and medical consultation recommended."

    else:

        analysis = """
The AI model indicates a lower probability of heart disease based on
the provided health metrics. Cardiovascular indicators appear within
safer ranges.
"""

        risk_color = colors.green
        risk_summary = "Low probability of heart disease. Maintain healthy lifestyle and regular checkups."

    elements.append(Paragraph(analysis.replace("\n","<br/>"), styles['Normal']))

    elements.append(Spacer(1,20))

    # ================= RISK SUMMARY BOX ================= #

    risk_box = Table([[Paragraph(risk_summary, styles['Normal'])]], colWidths=[470])

    risk_box.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),risk_color),
        ('TEXTCOLOR',(0,0),(-1,-1),colors.white),
        ('PADDING',(0,0),(-1,-1),12)
    ]))

    elements.append(Paragraph("Risk Summary", styles['Heading2']))
    elements.append(Spacer(1,10))
    elements.append(risk_box)

    elements.append(Spacer(1,25))

    # ================= DIET PLAN ================= #

    elements.append(Paragraph("Recommended Diet Plan", styles['Heading2']))
    elements.append(Spacer(1,10))

    diet_box = Table([[Paragraph(prediction.diet_plan.replace("\n","<br/>"), styles['Normal'])]])

    diet_box.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),colors.lightgreen),
        ('PADDING',(0,0),(-1,-1),10)
    ]))

    elements.append(diet_box)

    elements.append(Spacer(1,25))

    # ================= EXERCISE PLAN ================= #

    elements.append(Paragraph("Recommended Exercise Plan", styles['Heading2']))
    elements.append(Spacer(1,10))

    exercise_box = Table([[Paragraph(prediction.exercise_plan.replace("\n","<br/>"), styles['Normal'])]])

    exercise_box.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),colors.lightyellow),
        ('PADDING',(0,0),(-1,-1),10)
    ]))

    elements.append(exercise_box)

    elements.append(Spacer(1,25))

    # ================= EMERGENCY ALERT ================= #

    if prediction.emergency:

        elements.append(Paragraph("Emergency Alert", styles['Heading2']))
        elements.append(Spacer(1,10))

        emergency_box = Table([[
            Paragraph("⚠ Immediate medical consultation recommended due to high predicted heart risk.", styles['Normal'])
        ]])

        emergency_box.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,-1),colors.red),
            ('TEXTCOLOR',(0,0),(-1,-1),colors.white),
            ('PADDING',(0,0),(-1,-1),12)
        ]))

        elements.append(emergency_box)

        elements.append(Spacer(1,20))

    # ================= FOOTER ================= #

    elements.append(Spacer(1,20))
    elements.append(Paragraph("Generated by HeartGuard AI System", styles['Normal']))
    elements.append(Paragraph("This report is AI generated and should not replace professional medical advice.", styles['Italic']))

    # ================= BUILD PDF ================= #

    pdf = SimpleDocTemplate(buffer, pagesize=letter)
    pdf.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="HeartGuard_AI_Report.pdf",
        mimetype="application/pdf"
    )



@app.route("/chatbot", methods=["GET", "POST"])
@login_required
def chatbot():

    if request.method == "POST":

        try:

            data = request.get_json()

            if not data:
                return jsonify({"reply": "No message received"})

            user_message = data.get("message", "")

            if user_message.strip() == "":
                return jsonify({"reply": "Please type a message."})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            reply = response.choices[0].message.content

            return jsonify({"reply": reply})

        except Exception as e:
            print("Chatbot error:", e)
            return jsonify({"reply": "AI assistant temporarily unavailable."})

    return jsonify({"reply": "Chatbot is running"})


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        with app.app_context():
            db.create_all()
            print(Prediction.__table__.columns.keys())
    app.run(debug=True)




