# Proof-of-Workforce: Job Listing Web Application

This is a simple web application built using Flask and SQLAlchemy for posting and viewing job listings. The application allows users to post new job listings, view existing job listings, and see detailed information about each job.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Templates](#templates)
- [Future Enhancements](#future-enhancements)

## Features

- **Post a Job**: Users can post new job listings by filling out a form with job details.
- **View Job Listings**: Users can view a list of all posted jobs, sorted by the date they were posted.
- **Job Details**: Users can click on a job listing to view more detailed information about the job.
- **Flash Messages**: The application uses Flask's flash messaging system to provide feedback to the user (e.g., success or error messages).

## Technologies Used

- **Flask**: A lightweight web framework for Python.
- **SQLAlchemy**: An ORM (Object-Relational Mapping) tool for database interactions.
- **SQLite**: A lightweight, file-based database used for simplicity.
- **Bootstrap**: A front-end framework for responsive and modern web design.

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/job-listing-app.git
   cd job-listing-app
   ```

2. **Install Dependencies**:
   ```bash
   pip install Flask Flask-SQLAlchemy
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Application**:
   Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage

- **Home Page**: The home page displays a list of all job postings. You can click on a job title to view more details.
- **Post a Job**: Click on the "Post a Job" button to navigate to the job posting form. Fill out the form and submit it to post a new job.
- **Job Details**: Click on any job title to view detailed information about the job, including the description and requirements.

## Code Explanation

### `app.py`

- **Flask App Initialization**:
  ```python
  app = Flask(__name__)
  app.config['SECRET_KEY'] = 'your_secret_key'
  app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jobs.db'
  db = SQLAlchemy(app)
  ```
  The Flask app is initialized with a secret key and a SQLite database URI. The `SQLAlchemy` object is created to manage database interactions.

- **Job Model**:
  ```python
  class Job(db.Model):
      id = db.Column(db.Integer, primary_key=True)
      title = db.Column(db.String(100), nullable=False)
      company = db.Column(db.String(100), nullable=False)
      description = db.Column(db.Text, nullable=False)
      requirements = db.Column(db.Text, nullable=False)
      posted_date = db.Column(db.DateTime, default=datetime.utcnow)
  ```
  The `Job` model represents a job posting in the database. It includes fields for the job title, company, description, requirements, and the date it was posted.

- **Database Initialization**:
  ```python
  with app.app_context():
      db.create_all()
  ```
  This ensures that the database tables are created if they don't already exist.

- **Routes**:
  - **Index Route**:
    ```python
    @app.route('/')
    def index():
        jobs = Job.query.order_by(Job.posted_date.desc()).all()
        return render_template('index.html', jobs=jobs)
    ```
    The index route fetches all job postings from the database, orders them by the posted date, and renders them on the home page.

  - **Job Details Route**:
    ```python
    @app.route('/job/<int:id>')
    def job_details(id):
        job = Job.query.get_or_404(id)
        return render_template('job_details.html', job=job)
    ```
    This route fetches a specific job by its ID and renders a detailed view of the job.

  - **Post Job Route**:
    ```python
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
                return redirect(url_for('index'))
            except Exception as e:
                db.session.rollback()
                flash(f'Error posting job: {str(e)}', 'danger')

        return render_template('post_job.html')
    ```
    This route handles both GET and POST requests. When a POST request is made, it creates a new job posting and adds it to the database. If successful, it redirects to the home page with a success message. If an error occurs, it rolls back the transaction and displays an error message.

- **Blockchain Integration (Placeholder)**:
  ```python
  def register_job_on_blockchain(job_id, job_data):
      print(f"Registering job {job_id} on blockchain: {job_data}")
      transaction_id = "dummy_transaction_id"
      return transaction_id
  ```
  This is a placeholder function for future blockchain integration. In a real application, this function would interact with a blockchain to register the job posting.

## Templates

### `index.html`

- Displays a list of all job postings.
- Includes a link to the job posting form.
- Uses Bootstrap for styling.

### `job_details.html`

- Displays detailed information about a specific job.
- Includes a link to return to the home page.

### `post_job.html`

- A form for posting new job listings.
- Includes fields for the job title, company, description, and requirements.

## Future Enhancements

- **User Authentication**: Add user authentication to allow only authorized users to post jobs.
- **Blockchain Integration**: Implement actual blockchain integration to register job postings on a blockchain.
- **Search and Filter**: Add search and filter functionality to the job listings page.
- **Pagination**: Implement pagination for the job listings page to handle a large number of postings.

## Conclusion

This application provides a simple and effective way to post and view job listings. It is built using Flask and SQLAlchemy, making it easy to extend and customize. Future enhancements could include user authentication, blockchain integration, and additional features to improve the user experience.
