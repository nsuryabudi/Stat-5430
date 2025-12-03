# 🏈 NFL Matchup Predictor

A machine learning system that predicts NFL game outcomes and team statistics using historical play-by-play data. Built with XGBoost and LightGBM, it provides win probabilities and forecasts for 22 different team statistics (passing yards, rushing yards, turnovers, sacks, etc.).

**Key Features:**
- Win probability predictions (67.37% AUC)
- 22 individual stat predictions per team
- Calibrated probabilities using isotonic regression
- FastAPI backend with interactive web interface

---

## 🚀 How to Run

### Initial Setup (First Time Only)

**1. Activate Virtual Environment**
```bash
# Windows
.venv/Scripts/activate

# macOS/Linux
source .venv/bin/activate
```

**2. Install Dependencies**
```bash
pip install -r backend/requirements.txt
```

**3. Download Data** ⏳ (5-10 minutes)
```bash
cd backend/src
python fetch_data_nflreadpy.py
```

**4. Build Features**
```bash
python build_features.py
```

**5. Train Win Probability Model**
```bash
python train_win.py
```

**6. Train Statistics Models** ⏳ (10-15 minutes)
```bash
python train_stats.py
```

---

### Running the Application

**7. Start Backend Server**
```bash
# From backend/src, go back to backend directory
cd ..

# Start FastAPI server
uvicorn app.main:app --reload
```

Backend runs at: **http://127.0.0.1:8000**  
API docs at: **http://127.0.0.1:8000/docs**

**8. Start Frontend** (New Terminal)
```bash
# From project root
cd frontend

# Start HTTP server
python -m http.server 3000
```

Frontend runs at: **http://127.0.0.1:3000**

---

## 🔧 Notes

- **First-time setup** takes ~20-30 minutes (data download + training)
- **Subsequent runs** are instant (just start backend + frontend)
- **Data refresh**: Re-run `fetch_data_nflreadpy.py` weekly during NFL season
- **Model files** are ignored by git (500MB+). Regenerate using training scripts.

---

**Data Source:** [nflverse/nfl_data_py](https://github.com/nflverse/nfl_data_py)