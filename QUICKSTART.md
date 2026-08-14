# Quick Start: Docker + MLflow + AWS

## What Was Added

```
✅ Docker Containers (5 total):
   ├── collector (polls AEMO)
   ├── inference (makes predictions)
   ├── backend (FastAPI)
   ├── dashboard (React)
   └── mlflow (experiment tracking)

✅ PostgreSQL Database (replacing SQLite)

✅ MLflow Integration for model training

✅ GitHub Actions CI/CD

✅ Deployment guide (DEPLOYMENT.md)
```

---

## 5-Minute Local Test

### 1. Start Everything

```bash
cd c:\dev\Electricity-Demand-Forecasting-Using-Statistical-Machine-Learning-and-Deep-Learning-Methods

# Start all containers
docker-compose up --build
```

This takes ~2 minutes. Wait for output showing all services healthy.

### 2. Access Components

In a browser:
- **Dashboard**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

### 3. Train a Model with MLflow Tracking

Open a new terminal:

```bash
cd c:\dev\Electricity-Demand-Forecasting-Using-Statistical-Machine-Learning-and-Deep-Learning-Methods

# Set MLflow tracking URI
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"

# Train XGBoost (logs to MLflow)
python -m scripts.train_xgboost_mlflow

# Go to MLflow UI (http://localhost:5000) and see the experiment run
```

### 4. Stop

```bash
docker-compose down
```

---

## Deploy to AWS (30 minutes)

### Prerequisites
- AWS free tier account
- SSH key pair downloaded

### 1. Create EC2 Instance

1. Go to AWS Console → EC2 Dashboard
2. **Launch Instance**
   - Image: Ubuntu 22.04 LTS (free tier)
   - Instance type: t2.micro (free tier)
   - Key pair: create or use existing
   - Security group: Allow SSH (22), HTTP (80), HTTPS (443)
3. Launch and wait 1 min

### 2. Connect & Install Docker

```bash
# Replace YOUR-KEY.pem and YOUR-EC2-IP
ssh -i YOUR-KEY.pem ubuntu@YOUR-EC2-IP

# On EC2:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker-compose --version
```

### 3. Deploy Code

```bash
# On EC2:
git clone https://github.com/YOUR-USERNAME/aemo-forecast.git
cd aemo-forecast

# Copy env file
cp .env.example .env

# Edit with your EC2's public IP
# nano .env
# Change VITE_API_BASE_URL and MLFLOW_TRACKING_URI to http://YOUR-EC2-IP:PORT

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

### 4. Access from Anywhere

- **Dashboard**: http://YOUR-EC2-IP:3000
- **API**: http://YOUR-EC2-IP:8000/health
- **MLflow**: http://YOUR-EC2-IP:5000

---

## File Structure

```
docker/
  ├── Dockerfile.collector      # Collector image
  ├── Dockerfile.inference      # Inference image
  ├── Dockerfile.backend        # FastAPI image
  ├── Dockerfile.dashboard      # React image
  ├── Dockerfile.mlflow         # MLflow image
  └── entrypoint.sh             # Startup script

docker-compose.yml              # Orchestrates all containers + PostgreSQL

.env.example                     # Environment variables template

.github/workflows/
  └── test-and-build.yml        # GitHub Actions CI/CD

src/data/
  └── db_postgres.py            # PostgreSQL adapter (replaces SQLite)

scripts/
  ├── train_xgboost_mlflow.py   # XGBoost + MLflow logging
  └── train_lstm_mlflow.py      # LSTM + MLflow logging

DEPLOYMENT.md                    # Full deployment guide (detailed)
```

---

## Architecture Diagram

```
DEVELOPMENT (Your Laptop):
  Your code → Git push → GitHub Actions
                           ↓ (runs tests & builds images)
  
LOCAL TESTING:
  docker-compose up --build
  ├─ PostgreSQL (5432)
  ├─ MLflow (5000) ← logs from train_xgboost_mlflow.py
  ├─ Collector (polls AEMO)
  ├─ Inference (predicts)
  ├─ Backend API (8000)
  └─ Dashboard (3000)

PRODUCTION (AWS EC2):
  Git → EC2 instance
  ├─ PostgreSQL (5432, internal)
  ├─ MLflow (5000)
  ├─ Collector
  ├─ Inference
  ├─ Backend API (8000, exposed)
  └─ Dashboard (3000, exposed)
```

---

## Next Steps

1. **Try locally first**: `docker-compose up --build`
2. **Train with MLflow**: `python -m scripts.train_xgboost_mlflow`
3. **Deploy to AWS**: Follow DEPLOYMENT.md Phase 2
4. **Add CI/CD**: Push to GitHub, GitHub Actions runs tests automatically

---

## MLflow Basics

**What it does**: Records every model training run with:
- Hyperparameters (max_depth, learning_rate, etc.)
- Metrics (MAE, RMSE, MAPE)
- Model artifacts (saved model file)
- Run metadata (duration, status)

**Why it matters**:
- Compare "XGBoost with max_depth=6" vs "max_depth=8"
- Track which model version is in production
- Reproduce any past training run
- Automated retraining picks up from MLflow registry

**How to use**:
```python
import mlflow

mlflow.set_experiment("AEMO-Demand-Forecasting")
with mlflow.start_run():
    mlflow.log_param("max_depth", 6)      # Parameters
    mlflow.log_metric("mae", 394)          # Metrics
    mlflow.sklearn.log_model(model, "xgboost")  # Model
```

Then view in MLflow UI: http://localhost:5000

---

## Common Commands

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Train with MLflow
python -m scripts.train_xgboost_mlflow

# Check database
docker-compose exec postgres psql -U aemo_user -d aemo_db
```

---

## Cost on AWS Free Tier

- EC2 t2.micro: FREE (750 hrs/month for 12 months)
- RDS PostgreSQL: FREE (750 hrs/month for 12 months, if used)
- ECR storage: FREE (50GB/month)
- **Total: $0 for 12 months**

After 12 months: ~$30-35/month

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `docker-compose: not found` | Install Docker Desktop |
| Containers not starting | `docker-compose logs` to see errors |
| Can't connect to MLflow | Ensure `MLFLOW_TRACKING_URI=http://localhost:5000` |
| Database errors | `docker-compose down -v` (reset) then `up --build` |
| Port 3000/5000/8000 already in use | Change ports in `docker-compose.yml` |

---

For detailed instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)
