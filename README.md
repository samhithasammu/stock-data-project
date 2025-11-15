# 📊 Stock Data Cleaning, Aggregation & Streamlit Dashboard

This project performs **data cleaning**, **schema normalization**, **parquet generation**, and **custom aggregations** using **Pandas**, followed by an interactive **Streamlit dashboard** for visualization.

It is designed as a minimal end-to-end workflow for processing stock-market datasets and building dashboards on top of cleaned data.

---

## 🧹 Data Cleaning Tasks (performed in `prepare_data.py`)

The script performs the following operations using **pandas only**:

### ✔ Load raw CSV  
- Reads all columns as strings  
- Identifies missing values (`""`, `"na"`, `"NA"`, `"null"`, `"-"`)  

### ✔ Normalize schema  
- Converts column headers to `snake_case`  
- Trims all whitespace  
- Unifies text casing (e.g., `"usd"` → `"USD"`)  
- Fixes date format → **YYYY-MM-DD**

### ✔ Convert types properly  
- Dates → `datetime`  
- Prices/volume → numeric  
- Yes/No flags → boolean  

### ✔ Deduplicate rows  
Removes exact duplicates based on key columns.

### ✔ Save cleaned output  
- `cleaned.csv`  
- `cleaned.parquet`

---

## 📈 Aggregations Created (any 3 analyses)

The script generates **three parquet-based analyses** as examples:

---

## 🖥️ Streamlit Dashboard (app.py)

The dashboard loads **cleaned.parquet** 


### ✔ Used to generate screenshots for submission  
All screenshots are inside the **screenshots/** folder.

---

## ▶️ How to Run the Project

### 1️⃣ Create the virtual environment  
python -m venv .venv

makefile
Copy code

### 2️⃣ Activate it  
**Windows:**
..venv\Scripts\activate


### 3️⃣ Install requirements  
pip install pandas numpy streamlit pyarrow matplotlib


### 4️⃣ Run the cleaning + aggregation script  
python prepare_data.py


### 5️⃣ Run the Streamlit dashboard  
streamlit run app.py


## 📸 Screenshots

Screenshots of the Streamlit charts and filters are included 

screenshots/


## ✨ Summary

This project demonstrates:

- ✔ Real-world data cleaning using **pandas**  
- ✔ Parquet file generation  
- ✔ Multiple custom aggregations  
- ✔ Interactive dashboard using **Streamlit**  
- ✔ A complete end-to-end mini data pipeline  

It is designed to be simple, readable, and suitable for academic submission.
