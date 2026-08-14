# Deployment Guide: Docker + MLflow + AWS

This guide walks you through deploying the AEMO demand forecasting system to AWS using Docker and MLflow for experiment tracking.

## Prerequisites

- AWS account with free tier eligibility
- Docker & Docker Compose installed locally
- Python 3.10+
- Git

## Phase 1: Local Testing with Docker Compose

### 1. Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration (optional for local testing)
# For now, defaults are fine
```

### 2. Start the Stack Locally

```bash
# Build and start all containers
docker-compose up --build

# In a separate terminal, check health
curl http://localhost:8000/health
```

This starts:
- **PostgreSQL** (localhost:5432)
- **MLflow** (localhost:5000) - access UI at http://localhost:5000
- **Collector** (runs in background)
- **Inference** (runs in background)
- **Backend API** (localhost:8000)
- **Dashboard** (localhost:3000)

### 3. Train Models with MLflow Tracking

```bash
# In the container environment, or locally with MLflow pointing to container:

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Train XGBoost with MLflow
python -m scripts.train_xgboost_mlflow

# Train LSTM with MLflow
python -m scripts.train_lstm_mlflow

# View results in MLflow UI
# Open http://localhost:5000 and click on experiments
```

### 4. Access Components

- **Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **MLflow UI**: http://localhost:5000
- **PostgreSQL**: localhost:5432

### 5. Stop Local Stack

```bash
docker-compose down

# Remove volumes if you want to reset (WARNING: deletes data)
docker-compose down -v
```

---

## Phase 2: Deploy to AWS EC2

### Step 1: Create AWS Account & EC2 Instance

1. Sign up for [AWS Free Tier](https://aws.amazon.com/free/)
2. Go to **EC2 Dashboard**
3. Click **Launch Instance**
4. Select **Ubuntu 22.04 LTS** (eligible for free tier)
5. Instance type: **t2.micro** (free tier eligible)
6. Configure security group:
   - Allow SSH (port 22) from your IP
   - Allow HTTP (port 80) from anywhere
   - Allow HTTPS (port 443) from anywhere
7. Create/select a key pair and download it (`.pem` file)
8. Launch instance

### Step 2: Connect to EC2 Instance

```bash
# Make key file readable
chmod 600 your-key.pem

# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Update system
sudo apt update && sudo apt upgrade -y
```

### Step 3: Install Docker & Dependencies

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify
docker --version
docker-compose --version
```

### Step 4: Clone Repository & Set Up

```bash
# Clone repo
git clone https://github.com/your-username/aemo-forecast.git
cd aemo-forecast

# Copy environment file
cp .env.example .env

# Edit .env with EC2-specific settings
# Replace localhost with your EC2 public IP
nano .env
```

Update `.env`:
```
POSTGRES_USER=aemo_user
POSTGRES_PASSWORD=your_secure_password  # Change this!
POSTGRES_DB=aemo_db

MLFLOW_TRACKING_URI=http://your-ec2-ip:5000
VITE_API_BASE_URL=http://your-ec2-ip:8000
```

### Step 5: Start Services on EC2

```bash
# Build and start containers
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f collector
docker-compose logs -f inference
docker-compose logs -f backend
```

### Step 6: Access from Anywhere

- **Dashboard**: http://your-ec2-ip:3000
- **API**: http://your-ec2-ip:8000/health
- **MLflow UI**: http://your-ec2-ip:5000

### Step 7: Set Up RDS (Optional - for More Durability)

Instead of PostgreSQL container, use AWS RDS:

1. Go to **RDS Dashboard**
2. Create **PostgreSQL 15** database (free tier eligible: t3.micro)
3. Note the **Endpoint** and **credentials**
4. Update `docker-compose.yml`:
   ```yaml
   # Comment out postgres service
   # postgres:
   #   image: postgres:15-alpine
   #   ...
   
   # Update collector/inference/backend to use RDS endpoint
   ```

5. Update `.env`:
   ```
   DATABASE_URL=postgresql://aemo_user:password@your-rds-endpoint:5432/aemo_db
   ```

---

## Phase 3: Set Up CI/CD (GitHub Actions)

### Step 1: Create GitHub Secrets

1. Go to your GitHub repo
2. **Settings → Secrets and variables → Actions**
3. Add secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION` (e.g., `us-east-1`)
   - `EC2_INSTANCE_ID`

### Step 2: GitHub Actions Workflow

The `.github/workflows/test-and-build.yml` file is already created.

On every push to `main`:
1. Runs pytest suite
2. Builds Docker images
3. Deploys to EC2 (requires custom deployment script)

### Step 3: Deploy Script (Optional)

Create `.github/workflows/deploy.sh`:
```bash
#!/bin/bash
set -e

EC2_USER=ubuntu
EC2_HOST=$1
KEY_PATH=$2

# Copy repo to EC2
scp -i $KEY_PATH -r . $EC2_USER@$EC2_HOST:~/aemo-forecast

# SSH and restart containers
ssh -i $KEY_PATH $EC2_USER@$EC2_HOST << 'EOF'
  cd ~/aemo-forecast
  docker-compose pull
  docker-compose up -d
  docker-compose logs collector
EOF
```

---

## Monitoring & Maintenance

### View Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs -f inference

# Real-time
docker-compose logs -f --tail=50
```

### Check Database

```bash
# SSH into EC2, then:
docker-compose exec postgres psql -U aemo_user -d aemo_db

# SQL commands
SELECT COUNT(*) FROM observations;
SELECT COUNT(*) FROM predictions;
SELECT * FROM collector_health ORDER BY polled_at DESC LIMIT 5;
```

### MLflow Model Management

1. Open http://your-ec2-ip:5000
2. Click on an experiment
3. Compare runs side-by-side
4. Register best model as "production"

### Set Up Auto-Retraining

Add a cron job to retrain weekly:

```bash
# SSH into EC2
crontab -e

# Add this line (runs every Sunday at 2 AM)
0 2 * * 0 cd /home/ubuntu/aemo-forecast && docker-compose exec -T inference python -m scripts.train_xgboost_mlflow
```

---

## Cost Breakdown (Free Tier)

| Service | Free Tier | Cost After 12 Months |
|---------|-----------|----------------------|
| EC2 t2.micro | 750 hrs/month | ~$10-15/month |
| RDS (if used) | 750 hrs/month | ~$20/month |
| ECR | 50GB/month | ~$0.10/GB |
| Data transfer | 100GB/month (from AWS) | ~$0.10/GB |
| **Total** | ~$0 | ~$30-35/month |

---

## Troubleshooting

### Containers won't start

```bash
# Check logs
docker-compose logs

# Rebuild
docker-compose down
docker-compose up --build
```

### Database connection error

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check DATABASE_URL in environment
docker-compose exec collector env | grep DATABASE_URL

# Test connection
docker-compose exec postgres psql -U aemo_user -d aemo_db -c "SELECT 1;"
```

### MLflow not tracking

```bash
# Check MLflow container
docker-compose ps mlflow

# Test MLflow API
curl http://localhost:5000/api/2.0/experiments/list

# Set MLFLOW_TRACKING_URI
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Out of disk space on EC2

```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a

# Check container sizes
docker ps -s
```

---

## Next Steps

1. ✅ Deploy local stack
2. ✅ Deploy to AWS EC2
3. ✅ Set up CI/CD
4. 🔄 Monitor live predictions
5. 🔄 Set up auto-retraining pipeline
6. 🔄 Create alerts for forecast accuracy drops
7. 🔄 Scale to multiple regions (Kubernetes)

---

## Support

For issues or questions:
- Check Docker Compose documentation
- Review AWS Free Tier limitations
- Check MLflow documentation for experiment tracking

