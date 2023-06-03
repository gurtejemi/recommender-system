#ssh -i jobportal.pem -L 3336:localhost:3306 ec2-user@3.97.197.26
import pandas as pd
import numpy as np
import mysql.connector
from functools import wraps
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
#run_with_ngrok(app)
AUTH_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'

def authenticate(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")

        if token != AUTH_TOKEN:
            return jsonify({"message": "Unauthorized"}), 401

        return func(*args, **kwargs)

    return decorated

def fetch_job_postings():
    conn = mysql.connector.connect(
        host="localhost",
        user="APP_user",
        password="App6510P8",
        database="app",
        port=3336
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Job")

    job_postings = []
    rows = cursor.fetchall()
    for row in rows:
        job_postings.append({
            'id': row[0],
            'skills': row[17].split(','),
            'experience': row[3],
            'education': row[23]
        })
    cursor.close()
    conn.close()
    return job_postings


def fetch_student_profiles():
    conn = mysql.connector.connect(
        host="localhost",
        user="APP_user",
        password="App6510P8",
        database="app",
        port=3336
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM StudentDetails")

    student_profiles = []

    rows = cursor.fetchall()
    for row in rows:
        student_profiles.append({
            'id': row[0],
            'skills': row[13].split(','),
            'experience': row[15],
            'education': row[11]
        })
    cursor.close()
    conn.close()
    return student_profiles


def preprocess_data(job_postings, student_profiles):
    # Combine skills, location,experience and education for each job posting and student profile
    job_texts = [
        f"{job['skills']} {job['experience']} {job['education']}" for job in job_postings]
    student_texts = [
        f"{profile['skills']} {profile['experience']} {profile['education']}" for profile in student_profiles]

    # Fit TF-IDF vectorizer to combined job texts and student texts
    vectorizer = TfidfVectorizer()
    vectorizer.fit(job_texts + student_texts)

    return vectorizer

# Example route

''''
@app.route('/')
def hello_world():
    app.logger.info('Hello World!')
    return 'Hello World!'
'''

@app.route('/', methods=['POST'])
@authenticate
def recommend_jobs():
    student_profile = request.json

    # Fetch job postings and student profiles from the MySQL database
    job_postings = fetch_job_postings()
    student_profiles = fetch_student_profiles()

    # Preprocess the data and fit TF-IDF vectorizer
    with app.app_context():
        vectorizer = preprocess_data(job_postings, student_profiles)

    similarities = []
    for job in job_postings:
        job_text = f"{job['skills']} {job['experience']} {job['education']}"
        job_vector = vectorizer.transform([job_text])

        student_text = f"{student_profile['skills']} {student_profile['experience']} {student_profile['education']}"
        student_vector = vectorizer.transform([student_text])

        similarity = cosine_similarity(job_vector, student_vector)[0][0]
        similarities.append({
            'job_id': job['id'],
            'similarity': similarity
        })

    # Sort the similarities based on the highest similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    print(vectorizer)

    with app.app_context():
        return jsonify(similarities)

@app.route("/recommend_profiles", methods=["POST"])
def recommend_profiles():
    job_id = request.json

    job_posting = fetch_job_postings()
    student_profiles = fetch_student_profiles()

    vectorizer = preprocess_data(job_posting, student_profiles)

    similarities = []
    for profile in student_profiles:
        student_text = (
            f"{profile['skills']} {profile['experience']} {profile['education']}"
        )
        student_vector = vectorizer.transform([student_text])
        similarity = cosine_similarity(student_vector, vectorizer.transform([job_posting["skills"]]))[0][0]
        similarities.append({"student_id": profile["id"], "similarity": similarity})

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return jsonify(similarities)

if __name__ == '__main__':
    app.run()
