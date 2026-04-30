from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import logging

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'secret-key-change-in-production'
CORS(app)
JWTManager(app)

@app.route('/')
def index():
    return jsonify({'message': 'e-KYC REST API v1.0.0', 'status': 'running'})

@app.route('/api/health/ping')
def ping():
    return jsonify({'message': 'pong'})

@app.route('/api/auth/login', methods=['POST'])
def login():
    return jsonify({'status': 'success', 'access_token': 'demo-token'})

@app.route('/api/kyc/verify/<id_num>')
def verify(id_num):
    return jsonify({'status': 'success', 'user': {'id': id_num}})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
