from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from web3 import Web3

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a strong secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jobs.db'  # Use SQLite for simplicity
db = SQLAlchemy(app)

# Blockchain setup
BLOCKCHAIN_NODE_URL = "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID"  # Replace with your Infura or local node URL
CONTRACT_ADDRESS = "0xYourContractAddress"  # Replace with your deployed smart contract address
PRIVATE_KEY = "your_private_key"  # Replace with your wallet's private key (keep this secure!)
WALLET_ADDRESS = "your_wallet_address"  # Replace with your wallet address

# Connect to Ethereum node
web3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_NODE_URL))

# Load smart contract ABI (replace with your contract's ABI)
CONTRACT_ABI = [
    # Replace with your contract's ABI
    {
        "inputs": [
            {"internalType": "string", "name": "title", "type": "string"},
            {"internalType": "string", "name": "company", "type": "string"},
            {"internalType": "string", "name": "description", "type": "string"},
            {"internalType": "string", "name": "requirements", "type": "string"}
        ],
        "name": "registerJob",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

# Initialize contract instance
contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

# Job model
class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    requirements = db.Column(db.Text, nullable=False)
    posted_date = db.Column(db.DateTime, default=datetime.utcnow)
    blockchain_tx_id = db.Column(db.String(66), nullable=True)  # Store blockchain transaction ID

    def __repr__(self):
        return f'<Job {self.title}>'

# Create database if it doesn't exist
with app.app_context():
    db.create_all()

# Routes
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

        # Create a new job in the database
        new_job = Job(title=title, company=company, description=description, requirements=requirements)

        try:
            # Register the job on the blockchain
            tx_hash = register_job_on_blockchain(title, company, description, requirements)

            # Save the blockchain transaction ID to the database
            new_job.blockchain_tx_id = tx_hash

            # Add the job to the database
            db.session.add(new_job)
            db.session.commit()

            flash('Job posted successfully and registered on the blockchain!', 'success')
            return redirect(url_for('index'))  # Redirect to job listings page
        except Exception as e:
            db.session.rollback()  # Rollback on error
            flash(f'Error posting job: {str(e)}', 'danger')

    return render_template('post_job.html')

# Blockchain integration
def register_job_on_blockchain(title, company, description, requirements):
    try:
        # Create a transaction to call the smart contract
        nonce = web3.eth.get_transaction_count(WALLET_ADDRESS)
        tx = contract.functions.registerJob(title, company, description, requirements).build_transaction({
            'chainId': 1,  # Mainnet (change for testnets)
            'gas': 2000000,
            'gasPrice': web3.to_wei('50', 'gwei'),
            'nonce': nonce,
        })

        # Sign the transaction
        signed_tx = web3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)

        # Send the transaction
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)

        # Wait for the transaction to be mined
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

        # Return the transaction hash
        return tx_receipt.transactionHash.hex()
    except Exception as e:
        raise Exception(f"Blockchain error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
