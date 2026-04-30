# Deployment Guide

## Local Development
```bash
pip install -r requirements.txt
python setup_database.py
streamlit run app.py
```

## Docker
```bash
docker-compose up -d
```

Access:
- Streamlit: http://localhost:8501
- API: http://localhost:5000
- phpMyAdmin: http://localhost:8080
