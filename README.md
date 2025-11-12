# ERP Archive Module - AI Document Processing System

Automated document processing system for the Lyncs ERP Archive module. Extracts information from business documents (invoices, receipts, contracts) in Arabic and English, and automatically fills archive upload forms with 95%+ accuracy.

## 🎯 Features

- **Multi-language Support**: Processes both Arabic and English documents
- **High Accuracy**: 95%+ accuracy with intelligent confidence scoring
- **Automatic Classification**: Identifies document types, categories, and departments
- **Smart Tag Generation**: Auto-generates relevant tags from document content
- **Human-in-the-Loop**: Review queue for documents requiring validation
- **Scalable Architecture**: Handles 1000+ documents per day
- **RESTful API**: Easy integration with existing systems
- **Real-time & Batch Processing**: Flexible processing modes

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Testing](#testing)
- [Contributing](#contributing)

## 🏗️ Architecture

```
┌─────────────┐
│   Upload    │
│   (FastAPI) │
└──────┬──────┘
       │
       ▼
┌─────────────────┐      ┌──────────────┐
│  Document       │─────▶│  OCR Engine  │
│  Processor      │      │  (Azure/AWS) │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  Claude API     │─────▶│  Structured  │
│  (Extraction)   │      │  Output      │
└────────┬────────┘      └──────────────┘
         │
         ▼
    ┌────┴────┐
    │Confidence│
    │  Score   │
    └────┬────┘
         │
    ┌────┴─────┐
    │  ≥95%?   │
    └─┬────┬───┘
      │Yes │No
      ▼    ▼
   ┌────┐ ┌──────┐
   │Auto│ │Review│
   │Sub │ │Queue │
   └────┘ └──────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.11+
- Redis (for task queue)
- PostgreSQL (for metadata storage)
- API Keys:
  - Anthropic Claude API
  - Azure Computer Vision (recommended) or AWS Textract

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/erp-archive-processor.git
cd erp-archive-processor
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Spacy Language Models

```bash
# English model
python -m spacy download en_core_web_sm

# Arabic model (if available)
python -m spacy download ar_core_web_sm
```

### Step 5: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
AZURE_OCR_KEY=your_azure_computer_vision_key
AZURE_OCR_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# Lyncs Archive API
LYNCS_BASE_URL=https://qa-lyncs.ploapp.net
LYNCS_API_KEY=your_lyncs_api_key

# Database
DATABASE_URL=postgresql://user:password@localhost/erp_archive
REDIS_URL=redis://localhost:6379/0

# Application Settings
UPLOAD_DIR=/tmp/document_uploads
CONFIDENCE_THRESHOLD_AUTO=0.95
CONFIDENCE_THRESHOLD_REVIEW=0.80
MAX_WORKERS=10

# Security
SECRET_KEY=your_secret_key_for_jwt
API_KEY_SALT=random_salt_value

# Environment
ENVIRONMENT=development  # development, staging, production
```

### Step 6: Initialize Database

```bash
# Create database tables
python scripts/init_db.py

# Run migrations (if using Alembic)
alembic upgrade head
```

### Step 7: Start Redis

```bash
# Using Docker
docker run -d -p 6379:6379 redis:latest

# Or using local installation
redis-server
```

## ⚙️ Configuration

### Categories Configuration

Edit `config/categories.yaml`:

```yaml
categories:
  - Financial Documents
  - Legal Documents
  - HR Documents
  - Operational Documents
  - Contracts
  - Purchase Orders
  - Invoices
  - Receipts

departments:
  - Finance
  - Legal
  - HR
  - Operations
  - IT
  - Sales
  - Procurement
  - Management

document_types:
  invoice:
    default_category: Financial Documents
    default_department: Finance
    keywords:
      en: [invoice, bill, payment]
      ar: [فاتورة, سند, دفع]
  
  contract:
    default_category: Legal Documents
    default_department: Legal
    keywords:
      en: [contract, agreement, terms]
      ar: [عقد, اتفاقية, شروط]
```

## 📖 Usage

### Starting the Application

#### Development Mode

```bash
# Start the API server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# In a separate terminal, start Celery workers (for batch processing)
celery -A tasks.celery_app worker --loglevel=info
```

#### Production Mode

```bash
# Start API with Gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# Start Celery workers
celery -A tasks.celery_app worker --loglevel=info --concurrency=10
```

### Processing a Document

#### Using Python Client

```python
import asyncio
from document_processor import DocumentProcessor

async def main():
    processor = DocumentProcessor(
        anthropic_api_key="your-key",
        azure_ocr_key="your-azure-key",
        categories=["Financial Documents", "Legal Documents"],
        departments=["Finance", "Legal"]
    )
    
    result = await processor.process_document(
        file_path="invoice.pdf",
        user_id="user123"
    )
    
    print(f"Status: {result['status']}")
    print(f"Confidence: {result['confidence_score']}")

asyncio.run(main())
```

#### Using cURL

```bash
# Upload and process a document
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.pdf" \
  -F "user_id=user123"

# Check processing status
curl "http://localhost:8000/api/documents/{job_id}/status"

# Get full result
curl "http://localhost:8000/api/documents/{job_id}/result"
```

#### Using Python Requests

```python
import requests

# Upload document
with open('invoice.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/documents/upload',
        files={'file': f},
        data={'user_id': 'user123'}
    )

job_id = response.json()['job_id']

# Poll for result
import time
while True:
    status = requests.get(
        f'http://localhost:8000/api/documents/{job_id}/status'
    ).json()
    
    if status['status'] == 'completed':
        result = requests.get(
            f'http://localhost:8000/api/documents/{job_id}/result'
        ).json()
        print(result)
        break
    
    time.sleep(2)
```

### Batch Processing

```python
import requests

response = requests.post(
    'http://localhost:8000/api/documents/batch',
    json={
        'file_paths': [
            '/path/to/invoice1.pdf',
            '/path/to/invoice2.pdf',
            '/path/to/contract1.pdf'
        ],
        'user_id': 'batch_user'
    }
)

batch_info = response.json()
print(f"Batch ID: {batch_info['batch_id']}")
print(f"Processing {batch_info['total_documents']} documents")
```

### Review Queue Management

```python
import requests

# Get review queue
response = requests.get('http://localhost:8000/api/review/queue')
queue = response.json()

print(f"Documents in review: {queue['total']}")

# Approve a document
requests.post(
    'http://localhost:8000/api/review/decision',
    json={
        'document_id': 'DOC_12345',
        'action': 'approve'
    }
)

# Edit and approve
requests.post(
    'http://localhost:8000/api/review/decision',
    json={
        'document_id': 'DOC_12346',
        'action': 'edit',
        'edited_data': {
            'basic_information': {
                'document_name': 'Corrected_Name',
                'description': 'Updated description'
            }
        }
    }
)
```

## 📚 API Documentation

Once the server is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/documents/upload` | POST | Upload and process a document |
| `/api/documents/{job_id}/status` | GET | Check processing status |
| `/api/documents/{job_id}/result` | GET | Get processing result |
| `/api/documents/batch` | POST | Batch process multiple documents |
| `/api/review/queue` | GET | Get documents in review queue |
| `/api/review/decision` | POST | Submit review decision |
| `/api/categories` | GET | Get available categories |
| `/api/departments` | GET | Get available departments |
| `/api/stats` | GET | Get processing statistics |

## 🐳 Deployment

### Docker Deployment

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - AZURE_OCR_KEY=${AZURE_OCR_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/erp_archive
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./uploads:/tmp/document_uploads

  worker:
    build: .
    command: celery -A tasks.celery_app worker --loglevel=info
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - AZURE_OCR_KEY=${AZURE_OCR_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/erp_archive
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=erp_archive
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

Build and run:

```bash
docker-compose up -d
```

### Production Deployment Checklist

- [ ] Set strong `SECRET_KEY` in environment
- [ ] Enable HTTPS/SSL
- [ ] Configure proper CORS settings
- [ ] Set up database backups
- [ ] Configure monitoring (Prometheus + Grafana)
- [ ] Set up error tracking (Sentry)
- [ ] Implement rate limiting
- [ ] Configure log aggregation
- [ ] Set up CI/CD pipeline
- [ ] Document disaster recovery procedures

## 🧪 Testing

### Run Unit Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

### Test Specific Component

```bash
# Test document processor
pytest tests/test_document_processor.py

# Test API endpoints
pytest tests/test_api.py

# Test Lyncs integration
pytest tests/test_lyncs_client.py
```

### Accuracy Testing

```bash
# Run accuracy tests on labeled dataset
python scripts/test_accuracy.py --dataset=test_data/labeled_documents.json
```

## 📊 Monitoring

### Key Metrics to Monitor

1. **Processing Metrics**
   - Documents processed per hour
   - Average processing time
   - Success/failure rate

2. **Accuracy Metrics**
   - Overall extraction accuracy
   - Field-level accuracy
   - Auto-approval rate
   - Manual correction rate

3. **System Metrics**
   - API response time
   - Queue depth
   - Worker utilization
   - Database performance

### Prometheus Metrics

Access metrics at: `http://localhost:8000/metrics`

```python
# Example metrics
documents_processed_total
documents_auto_approved_total
documents_review_required_total
processing_time_seconds
extraction_confidence_score
api_requests_total
```

## 🔒 Security

### Best Practices

1. **API Security**
   - Use API keys for authentication
   - Implement rate limiting
   - Validate all inputs

2. **Data Security**
   - Encrypt data at rest
   - Encrypt data in transit (HTTPS)
   - Implement audit logging
   - Regular security audits

3. **Access Control**
   - Role-based access control (RBAC)
   - Principle of least privilege
   - Regular credential rotation

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Submit a pull request

### Code Style

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

- GitHub Issues: https://github.com/your-org/erp-archive-processor/issues
- Email: support@yourcompany.com
- Documentation: https://docs.yourcompany.com

## 📈 Roadmap

### Phase 1 (Complete)
- [x] Core document processing
- [x] Multi-language support
- [x] REST API
- [x] Basic monitoring

### Phase 2 (In Progress)
- [ ] Advanced ML models
- [ ] Real-time dashboard
- [ ] Mobile app integration
- [ ] Advanced analytics

### Phase 3 (Planned)
- [ ] Blockchain verification
- [ ] Advanced workflows
- [ ] AI-powered insights
- [ ] Multi-tenant support

## 🙏 Acknowledgments

- Anthropic Claude for AI capabilities
- Azure Computer Vision for OCR
- FastAPI framework
- Open source community

---

**Version:** 1.0.0  
**Last Updated:** January 2025  
**Maintained by:** Your Organization
# Document-Extractor
