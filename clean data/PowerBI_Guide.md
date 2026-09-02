# 📈 Power BI Guide — IPL Project Dashboard

## ✅ What You Already Have (No work needed!)

Your file `data/processed/IPL_PowerBI_Data.xlsx` already contains **9 sheets**, one per analysis table.

> [!IMPORTANT]
> **Fastest approach:** Just open Power BI Desktop and import this **one Excel file** — all your tables load instantly. No need to import CSVs separately.

---

## 🔌 Step-by-Step: Connect Power BI to Your Data

### Step 1 — Open Power BI Desktop
- Download free from: **powerbi.microsoft.com** → Download → Power BI Desktop
- Open it and click **"Get data"** from the Home tab

### Step 2 — Import the Excel Workbook
```
Home → Get Data → Excel Workbook
```
Browse to:
```
C:\Users\Javed Shaikh\Desktop\clg project - Copy\data\processed\IPL_PowerBI_Data.xlsx
```

### Step 3 — Select These Sheets in Navigator

| ✅ Sheet Name | What it Contains |
|---|---|
| `dashboard_overview` | Total matches, runs, wickets, top player/team |
| `dashboard_team_performance` | Wins, matches played, win% per team |
| `dashboard_batsman_stats` | Every batsman's runs, strike rate, 4s, 6s |
| `dashboard_bowler_stats` | Every bowler's wickets and economy rate |
| `dashboard_venue_analysis` | Matches per venue, avg first innings score |
| `dashboard_season_analysis` | Season champion, total runs/wickets |
| `dashboard_toss_analysis` | Toss win = match win? counts |
| `cleaned_matches` | Full cleaned match records (for drill-through) |

Click **Load**

### Step 4 — Go to Report View (Bar chart icon on left sidebar)

---

## 📐 Exactly What Charts to Build (8 Visuals)

### 🔲 PAGE 1 — Overview Dashboard

#### Visual 1: KPI Cards (top row)
- **Visual type:** Card
- **Source:** `dashboard_overview`
- **Fields:** `total_matches`, `total_runs`, `total_wickets`
- **Place 3 cards in a row** at the top of the page

#### Visual 2: Team Wins — Bar Chart
- **Visual type:** Clustered Bar Chart
- **Source:** `dashboard_team_performance`
- **Y-axis:** `team`
- **X-axis:** `wins`
- **Sort by:** `wins` descending
- **Color:** Data colors → pick IPL orange/yellow

#### Visual 3: Toss Analysis — Pie Chart
- **Visual type:** Pie Chart
- **Source:** `dashboard_toss_analysis`
- **Legend:** `toss_match_same` (shows True/False)
- **Values:** `matches`
- **Title:** "Does Toss Winner = Match Winner?"

---

### 🔲 PAGE 2 — Players

#### Visual 4: Top Batsmen — Bar Chart
- **Visual type:** Clustered Bar Chart
- **Source:** `dashboard_batsman_stats`
- **Y-axis:** `batsman`
- **X-axis:** `total_runs`
- **Add filter:** Top N = 15 (use Top N filter → By `total_runs`)

#### Visual 5: Top Bowlers — Bar Chart
- **Visual type:** Clustered Bar Chart
- **Source:** `dashboard_bowler_stats`
- **Y-axis:** `bowler`
- **X-axis:** `wickets`
- **Add filter:** Top N = 15

---

### 🔲 PAGE 3 — Seasons & Venues

#### Visual 6: Runs Per Season — Line Chart
- **Visual type:** Line Chart
- **Source:** `dashboard_season_analysis`
- **X-axis:** `season`
- **Y-axis:** `total_runs`
- **Title:** "IPL Run Explosion — 2008 to 2026"

#### Visual 7: Season Champions — Table
- **Visual type:** Table
- **Source:** `dashboard_season_analysis`
- **Columns:** `season`, `champion`, `matches`, `total_runs`
- **Sort by:** `season` ascending

#### Visual 8: Venue Analysis — Bar Chart
- **Visual type:** Clustered Bar Chart
- **Source:** `dashboard_venue_analysis`
- **Y-axis:** `venue`
- **X-axis:** `first_innings_score`
- **Filter:** Top 10 by `matches`

---

## 🎨 Power BI Theme Tips

| Setting | Value |
|---------|-------|
| **Theme color** | Orange `#f59e0b` (IPL color) |
| **Background** | Dark `#1c1917` for a premium look |
| **Font** | Segoe UI |
| **Canvas size** | 16:9 widescreen |

**To apply theme:** View → Themes → Customize → Set Primary Color = `#f59e0b`

---

## 🔗 Creating Relationships (Model View)

1. Click the **Model View** icon (diagram icon on left sidebar)
2. Drag `id` from `cleaned_matches` to `match_id` in any delivery table
3. This creates a **one-to-many** relationship

> [!NOTE]
> For the examiner presentation, the dashboard tables (`dashboard_*`) work independently and don't need relationships. Only set up relationships if you want drill-through from a match to its deliveries.

---

## 🗣️ How to Show the Dashboard to Your Examiner

### Presentation Order (3 minutes):

**1. Start with Overview Page (30 sec)**
> *"This is our IPL Analytics Dashboard built in Power BI. It covers 1,218 real IPL matches from 2008 to 2026. You can see the key stats — total runs, wickets, and our most successful team is Mumbai Indians."*

**2. Click Team Chart (30 sec)**
> *"Here we can see all teams ranked by wins. Mumbai Indians leads with 155 wins, followed by Chennai Super Kings with 148. Gujarat Titans has the highest win percentage at 61% despite fewer seasons played."*

**3. Click Toss Pie Chart (20 sec)**
> *"This answers an interesting question — does winning the toss help you win the match? 51.6% yes, 48.4% no. So toss gives a slight advantage but is NOT the deciding factor — which is why our ML model uses other features too."*

**4. Click Players Page (40 sec)**
> *"On our player analysis page — Virat Kohli is the all-time IPL run scorer with 9,195 runs and a strike rate of 135. For bowling, YS Chahal leads with 229 wickets. These stats come from 395,000 ball-by-ball records."*

**5. Click Season Chart (30 sec)**
> *"This line chart shows how IPL has grown — runs per season have gone from 17,000 in 2008 to 27,000 in 2026. Cricket is becoming more aggressive every year."*

**6. Show the Python code running (30 sec)**
> *"All these visuals come from our Python pipeline — I can run it live. It downloads data, cleans it, trains the ML model, and exports this Excel file automatically."*

---

## 💾 Files to Show in Your Exam

| File | What to Say |
|------|-------------|
| `gemini-code-1786516150300.py` | "This is our main pipeline — 365 lines covering all 5 stages" |
| `ipl_pipeline_clean.py` | "This is the cleaned version with full comments — easier to explain" |
| `data/processed/IPL_PowerBI_Data.xlsx` | "This is the output — 9 sheets, imported into Power BI" |
| `models/random_forest_ipl.joblib` | "This is our saved trained model — 1.8 MB" |
| Power BI Dashboard | "Live visualization of all our analysis" |
