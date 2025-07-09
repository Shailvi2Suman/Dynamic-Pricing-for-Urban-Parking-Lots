# 🚗 Dynamic Real-Time Parking Pricing Engine

A real-time dynamic pricing engine for urban parking lots, built using **Python**, **Pathway**, and **Bokeh**. The system simulates streaming data and computes live parking prices using multiple intelligent pricing models.

---

## 📌 Problem Statement

The objective is to dynamically compute and update parking prices in real-time for multiple urban lots based on a combination of:

- Occupancy patterns
- Traffic levels
- Queue length
- Special events
- Vehicle type/weight
- Nearby competition (other parking lots)

📄 Full problem statement: _See_ `problem_statement.pdf`

---

## 📂 Files

| File                          | Description                                      |
|------------------------------|--------------------------------------------------|
| `NOTEBOOK.ipynb`             | Main working notebook with all models           |
| `daily_price_final.jsonl`    | Output file with tumbling-window model results  |
| `dataset.csv`                | Input parking stream dataset (if uploaded)      |
| `README.md`                  | You're here!                                    |
| `problem_statement.pdf`      | Original project prompt                         |

---

## 🚀 Features

✅ Real-time streaming simulation using Pathway  
✅ Multiple pricing models with smooth explainable outputs  
✅ Time-windowed aggregation (daily price calculation)  
✅ Visualizations using Bokeh  
✅ Final outputs exported to `.jsonl` format

---

## 🧠 Pricing Models Implemented

| Model | Name                     | Logic Summary |
|-------|--------------------------|---------------|
| 1     | **Baseline**             | Linear with occupancy |
| 2     | **Demand-based**         | Uses queue, traffic, special day |
| 3     | **Competitive**          | Placeholder logic using surrounding lots |
| 4     | **Tumbling Window Model**| Daily smoothing using max-min occupancy |

---

## 📊 Visualization

<img src="bokeh_plot.png" alt="Daily Price Plot" width="600"/>

- Real-time Bokeh plot for dynamic pricing over time
- Multiple lot support for future extension

---

## ⚙️ How It Works

1. Parking stream ingested using `pw.demo.replay_csv(...)`
2. Stream parsed, timestamps converted
3. Model functions compute dynamic price
4. Windowed reductions applied (Model 4)
5. Output stored as JSONL
6. Bokeh used for visualizing prices

---

## 🛠️ Tech Stack

- Python 3.11+
- [Pathway](https://pathway.com/)
- Bokeh
- Pandas, NumPy

---

## ▶️ Running Locally

```bash
pip install pathway pandas bokeh
