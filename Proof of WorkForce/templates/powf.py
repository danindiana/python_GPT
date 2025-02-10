from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a strong secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jobs.db'  # Use SQLite for simplicity
db = SQLAlchemy(app)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    requirements = db.Column(db.Text, nullable=False)
    posted_date = db.Column(db.DateTime, default=datetime.utcnow)
    # Add blockchain transaction ID here later for verification

    def __repr__(self):
        return f'<Job {self.title}>'

# Create database if it doesn't exist
with app.app_context():
    db.create_all()


@app.route('/')
def index():
    jobs = Job.query.order_by(Job.posted_date.desc()).all()
    return render_template('index.html', jobs=jobs)

@app.route('/job/<int:id>')
def job_details(id):
    job = Job.query.get_or_404(id)
    return render_template('job_details.html', job=job)


@app.route('/post_job', methods=['GET', 'POST'])
def post_job():
    if request.method == 'POST':
        title = request.form['title']
        company = request.form['company']
        description = request.form['description']
        requirements = request.form['requirements']

        new_job = Job(title=title, company=company, description=description, requirements=requirements)
        try:
            db.session.add(new_job)
            db.session.commit()
            flash('Job posted successfully!', 'success')
            return redirect(url_for('index'))  # Redirect to job listings page
        except Exception as e:
            db.session.rollback()  # Important: Rollback on error
            flash(f'Error posting job: {str(e)}', 'danger')

    return render_template('post_job.html')

# Placeholder for blockchain integration (simplified)
def register_job_on_blockchain(job_id, job_data):
    # In a real application, you would interact with a blockchain here.
    # This is a simplified example.
    print(f"Registering job {job_id} on blockchain: {job_data}")
    # Here, you would interact with a blockchain (e.g., using a library like Web3.py)
    # to create a transaction that records the job data.
    # The transaction ID would then be stored in the Job model.
    transaction_id = "dummy_transaction_id"  # Replace with actual transaction ID
    return transaction_id

if __name__ == '__main__':
    app.run(debug=True)
