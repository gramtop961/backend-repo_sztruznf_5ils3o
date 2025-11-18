import os
import io
import uuid
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Try to use DuckDB if available; otherwise fall back to Pandas-only
try:
    import duckdb  # type: ignore
    DUCKDB_AVAILABLE = True
except Exception:
    duckdb = None
    DUCKDB_AVAILABLE = False

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Optional PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROC_DIR = os.path.join(DATA_DIR, 'processed')
SAMPLES_DIR = os.path.join(DATA_DIR, 'samples')
META_DB = os.path.join(DATA_DIR, 'meta.db')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# --- App ---
app = FastAPI(title="Knowlance MVP API", version="0.1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Metadata (SQLite) ---

def get_conn():
    conn = sqlite3.connect(META_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_meta():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            tenant_id TEXT,
            name TEXT,
            raw_path TEXT,
            created_at TEXT,
            schema_json TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            id TEXT PRIMARY KEY,
            dataset_id TEXT,
            processed_path TEXT,
            created_at TEXT,
            summary_json TEXT,
            FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            id TEXT PRIMARY KEY,
            dataset_id TEXT,
            created_at TEXT,
            text TEXT,
            sql TEXT,
            result_rows INTEGER
        );
        """
    )
    conn.commit()
    conn.close()


init_meta()

# --- Models ---
class PreprocessIn(BaseModel):
    dataset_id: str
    recipe: Optional[Dict[str, Any]] = None


class ChatIn(BaseModel):
    dataset_id: str
    text: str


# --- Helpers ---

def infer_schema(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_datetime64_any_dtype(dtype):
            mapping[col] = 'datetime'
        elif pd.api.types.is_integer_dtype(dtype):
            mapping[col] = 'integer'
        elif pd.api.types.is_float_dtype(dtype):
            mapping[col] = 'float'
        else:
            mapping[col] = 'string'
    return mapping


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Try to parse a 'date' column
    for cand in ['date', 'order_date', 'created_at', 'timestamp']:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors='coerce')
            break
    # Convert price to float
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    # Convert quantity to int
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        # keep as float for simplicity; fillna
        df['quantity'] = df['quantity'].fillna(0)
    # Impute numeric columns
    for col in df.select_dtypes(include=['float', 'int']).columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    # Strip strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def ensure_sample_csv():
    sample_path = os.path.join(SAMPLES_DIR, 'sales_sample.csv')
    if not os.path.exists(sample_path):
        import random
        import csv
        start = datetime.now() - timedelta(days=120)
        categories = ['Electronics', 'Apparel', 'Home', 'Beauty']
        products = {
            'Electronics': ['Headphones', 'Keyboard', 'Mouse', 'Monitor'],
            'Apparel': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers'],
            'Home': ['Lamp', 'Chair', 'Mug', 'Pillow'],
            'Beauty': ['Serum', 'Lotion', 'Shampoo', 'Conditioner']
        }
        channels = ['Online', 'Retail', 'Wholesale']
        regions = ['North', 'South', 'East', 'West']
        rows: List[List[Any]] = []
        order_id = 1000
        for i in range(500):
            date = (start + timedelta(days=random.randint(0, 120))).date().isoformat()
            cat = random.choice(categories)
            prod = random.choice(products[cat])
            price = round(random.uniform(10, 200), 2)
            qty = random.randint(1, 5)
            channel = random.choice(channels)
            region = random.choice(regions)
            rows.append([date, order_id + i, prod, cat, price, qty, channel, region])
        with open(sample_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'order_id', 'product', 'category', 'price', 'quantity', 'channel', 'region'])
            writer.writerows(rows)
    return sample_path


ensure_sample_csv()

# --- Utility to read source into pandas ---

def read_source_to_df(source: str) -> pd.DataFrame:
    if source.endswith('.parquet'):
        return pd.read_parquet(source)
    return pd.read_csv(source)


# --- Routes ---

@app.get("/")
def root():
    return {"message": "Knowlance MVP API running", "duckdb": DUCKDB_AVAILABLE}


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat(), "duckdb": DUCKDB_AVAILABLE}


@app.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    tenant_id: str = Form("tenant_demo"),
):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        dataset_id = str(uuid.uuid4())
        raw_path = os.path.join(RAW_DIR, f"{dataset_id}.csv")
        with open(raw_path, 'wb') as f:
            f.write(contents)
        df = pd.read_csv(io.BytesIO(contents))
        schema = infer_schema(df)
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO datasets (id, tenant_id, name, raw_path, created_at, schema_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                dataset_id,
                tenant_id,
                file.filename,
                raw_path,
                datetime.utcnow().isoformat(),
                json.dumps(schema),
            ),
        )
        conn.commit()
        conn.close()
        head_rows = df.head(10).to_dict(orient='records')
        return {
            "dataset_id": dataset_id,
            "raw_path": raw_path,
            "schema": schema,
            "preview": head_rows,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess")
def preprocess(payload: PreprocessIn):
    try:
        conn = get_conn()
        cur = conn.cursor()
        row = cur.execute("SELECT id, raw_path FROM datasets WHERE id = ?", (payload.dataset_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Dataset not found")
        df = pd.read_csv(row[1])
        df = basic_preprocess(df)
        snapshot_id = str(uuid.uuid4())
        processed_path = os.path.join(PROC_DIR, f"{snapshot_id}.parquet")
        df.to_parquet(processed_path, index=False)
        summary = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns),
            "schema": infer_schema(df),
        }
        cur.execute(
            "INSERT INTO snapshots (id, dataset_id, processed_path, created_at, summary_json) VALUES (?, ?, ?, ?, ?)",
            (snapshot_id, payload.dataset_id, processed_path, datetime.utcnow().isoformat(), json.dumps(summary)),
        )
        conn.commit()
        conn.close()
        return {"snapshot_id": snapshot_id, "processed_path": processed_path, "summary": summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets")
def list_datasets():
    conn = get_conn()
    cur = conn.cursor()
    ds = [dict(r) for r in cur.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()]
    for d in ds:
        d["schema"] = json.loads(d.pop("schema_json")) if d.get("schema_json") else {}
        snaps = [dict(r) for r in cur.execute("SELECT * FROM snapshots WHERE dataset_id = ? ORDER BY created_at DESC", (d['id'],)).fetchall()]
        for s in snaps:
            s["summary"] = json.loads(s.get("summary_json") or '{}')
            s.pop("summary_json", None)
        d["snapshots"] = snaps
    conn.close()
    return {"datasets": ds}


@app.get("/dashboard-metrics")
def dashboard_metrics(dataset_id: str = Query(...)):
    conn = get_conn()
    cur = conn.cursor()
    snap = cur.execute("SELECT processed_path FROM snapshots WHERE dataset_id = ? ORDER BY created_at DESC LIMIT 1", (dataset_id,)).fetchone()
    if not snap:
        ds = cur.execute("SELECT raw_path FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        source = ds[0]
    else:
        source = snap[0]

    try:
        if DUCKDB_AVAILABLE:
            con = duckdb.connect()
            if source.endswith('.parquet'):
                con.execute("CREATE VIEW dataset AS SELECT * FROM parquet_scan(?)", [source])
            else:
                con.execute("CREATE VIEW dataset AS SELECT * FROM read_csv_auto(?, header=True)", [source])
            kpi_rev = con.execute("SELECT COALESCE(SUM(price*quantity),0) FROM dataset WHERE date >= (CURRENT_DATE - INTERVAL 30 DAY)").fetchone()[0]
            rev_prev = con.execute("SELECT COALESCE(SUM(price*quantity),0) FROM dataset WHERE date < (CURRENT_DATE - INTERVAL 30 DAY) AND date >= (CURRENT_DATE - INTERVAL 60 DAY)").fetchone()[0]
            top_cat_row = con.execute("SELECT category, SUM(price*quantity) as rev FROM dataset GROUP BY 1 ORDER BY 2 DESC LIMIT 1").fetchone()
            top_category = top_cat_row[0] if top_cat_row else None
            nulls = con.execute("SELECT SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END) + SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) + SUM(CASE WHEN quantity IS NULL THEN 1 ELSE 0 END) FROM dataset").fetchone()[0]
            total_rows = con.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
            ts = con.execute("SELECT date_trunc('day', date) AS day, SUM(price*quantity) as revenue FROM dataset GROUP BY 1 ORDER BY 1").fetchdf()
            cats = con.execute("SELECT category, SUM(price*quantity) as revenue FROM dataset GROUP BY 1 ORDER BY 2 DESC LIMIT 10").fetchdf()
            con.close()
        else:
            df = read_source_to_df(source)
            # Ensure correct dtypes
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            for col in ['price', 'quantity']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            recent_cut = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=30)
            prev_cut = recent_cut - pd.Timedelta(days=30)
            kpi_rev = float(((df[df['date'] >= recent_cut]['price'] * df[df['date'] >= recent_cut]['quantity']).sum()) if 'date' in df.columns else (df['price']*df['quantity']).sum())
            rev_prev = float(((df[(df['date'] < recent_cut) & (df['date'] >= prev_cut)]['price'] * df[(df['date'] < recent_cut) & (df['date'] >= prev_cut)]['quantity']).sum()) if 'date' in df.columns else 0)
            top_category = None
            if 'category' in df.columns:
                cats_sum = df.assign(rev=df['price']*df['quantity']).groupby('category', dropna=False)['rev'].sum().sort_values(ascending=False)
                top_category = cats_sum.index[0] if len(cats_sum) else None
            nulls = sum(df[c].isna().sum() for c in df.columns if c in ['date','price','quantity'])
            total_rows = len(df)
            if 'date' in df.columns:
                ts_df = df.copy()
                ts_df['day'] = ts_df['date'].dt.floor('D')
                ts = ts_df.groupby('day', dropna=False).apply(lambda g: (g['price']*g['quantity']).sum()).reset_index(name='revenue').sort_values('day')
            else:
                ts = pd.DataFrame({'day': [], 'revenue': []})
            if 'category' in df.columns:
                cats = df.assign(revenue=df['price']*df['quantity']).groupby('category', dropna=False)['revenue'].sum().reset_index().sort_values('revenue', ascending=False).head(10)
            else:
                cats = pd.DataFrame({'category': [], 'revenue': []})
        growth = 0.0
        if rev_prev and rev_prev != 0:
            growth = (kpi_rev - rev_prev) / rev_prev * 100.0
        risk_score = min(100, int((nulls / max(1, total_rows)) * 100))
        return {
            "kpis": {
                "total_revenue_30d": float(kpi_rev or 0),
                "growth_pct": float(growth),
                "top_category": top_category,
                "risk_score": int(risk_score),
            },
            "charts": {
                "revenue_over_time": {
                    "x": (ts['day'].astype(str).tolist() if isinstance(ts, pd.DataFrame) and not ts.empty else []),
                    "y": (ts['revenue'].astype(float).tolist() if isinstance(ts, pd.DataFrame) and not ts.empty else []),
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Revenue"
                },
                "top_categories": {
                    "x": (cats['category'].astype(str).tolist() if isinstance(cats, pd.DataFrame) and not cats.empty else []),
                    "y": (cats['revenue'].astype(float).tolist() if isinstance(cats, pd.DataFrame) and not cats.empty else []),
                    "type": "bar",
                    "name": "Revenue by Category"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- NLQ mock ---

def rule_based_sql(question: str) -> str:
    q = question.lower()
    if 'monthly' in q and 'revenue' in q:
        return "SELECT date_trunc('month', date) as month, SUM(price*quantity) as revenue FROM dataset GROUP BY 1 ORDER BY 1"
    if ('top' in q and 'product' in q) or ('top' in q and 'category' in q):
        if 'product' in q:
            return "SELECT product, SUM(price*quantity) as revenue FROM dataset GROUP BY 1 ORDER BY 2 DESC LIMIT 5"
        return "SELECT category, SUM(price*quantity) as revenue FROM dataset GROUP BY 1 ORDER BY 2 DESC LIMIT 5"
    if 'predict' in q or 'forecast' in q:
        return "SELECT SUM(price*quantity)/NULLIF(COUNT(DISTINCT date),0)*30 as predicted_next_month FROM dataset WHERE date >= (CURRENT_DATE - INTERVAL 30 DAY)"
    if 'revenue' in q:
        return "SELECT SUM(price*quantity) as revenue FROM dataset"
    return "SELECT * FROM dataset LIMIT 10"


def pandas_query_from_intent(df: pd.DataFrame, question: str) -> pd.DataFrame:
    q = question.lower()
    # Ensure types
    if 'date' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['price','quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'monthly' in q and 'revenue' in q and 'date' in df.columns:
        g = df.copy()
        g['month'] = g['date'].dt.to_period('M').dt.to_timestamp()
        return g.groupby('month', dropna=False).apply(lambda x: (x['price']*x['quantity']).sum()).reset_index(name='revenue').sort_values('month')
    if 'top' in q and 'product' in q and 'product' in df.columns:
        return df.assign(revenue=df['price']*df['quantity']).groupby('product', dropna=False)['revenue'].sum().reset_index().sort_values('revenue', ascending=False).head(5)
    if 'top' in q and 'category' in q and 'category' in df.columns:
        return df.assign(revenue=df['price']*df['quantity']).groupby('category', dropna=False)['revenue'].sum().reset_index().sort_values('revenue', ascending=False).head(5)
    if 'predict' in q and 'date' in df.columns:
        recent = df[df['date'] >= (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=30))]
        if len(recent) == 0:
            return pd.DataFrame({"predicted_next_month": [0.0]})
        per_day = (recent['price']*recent['quantity']).sum() / max(1, recent['date'].dt.date.nunique())
        return pd.DataFrame({"predicted_next_month": [float(per_day*30)]})
    if 'revenue' in q:
        return pd.DataFrame({"revenue": [float((df['price']*df['quantity']).sum())]})
    return df.head(10)


@app.post("/chat")
def chat(payload: ChatIn):
    conn = get_conn()
    cur = conn.cursor()
    snap = cur.execute("SELECT processed_path FROM snapshots WHERE dataset_id = ? ORDER BY created_at DESC LIMIT 1", (payload.dataset_id,)).fetchone()
    if not snap:
        ds = cur.execute("SELECT raw_path FROM datasets WHERE id = ?", (payload.dataset_id,)).fetchone()
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        source = ds[0]
    else:
        source = snap[0]

    sql = rule_based_sql(payload.text)
    try:
        if DUCKDB_AVAILABLE:
            con = duckdb.connect()
            if source.endswith('.parquet'):
                con.execute("CREATE VIEW dataset AS SELECT * FROM parquet_scan(?)", [source])
            else:
                con.execute("CREATE VIEW dataset AS SELECT * FROM read_csv_auto(?, header=True)", [source])
            df = con.execute(sql).fetchdf()
            con.close()
        else:
            df = read_source_to_df(source)
            df = pandas_query_from_intent(df, payload.text)
        # Save query meta
        qid = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO queries (id, dataset_id, created_at, text, sql, result_rows) VALUES (?, ?, ?, ?, ?, ?)",
            (qid, payload.dataset_id, datetime.utcnow().isoformat(), payload.text, sql, int(df.shape[0])),
        )
        conn.commit()
        chart = None
        if df.shape[1] >= 2:
            chart = {
                "x": df.iloc[:, 0].astype(str).tolist(),
                "y": df.iloc[:, 1].astype(float).tolist() if pd.api.types.is_numeric_dtype(df.iloc[:,1]) else [],
                "type": "bar" if 'top' in payload.text.lower() else "scatter",
                "name": "Result"
            }
        return {
            "answer_text": "Here are the results.",
            "sql": sql if DUCKDB_AVAILABLE else None,
            "results": {
                "columns": df.columns.tolist(),
                "rows": df.head(50).to_dict(orient='records'),
            },
            "chart_spec": chart,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export")
def export_pdf(dataset_id: str, snapshot_id: Optional[str] = None):
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(status_code=501, detail="PDF export not available (install reportlab)")
    conn = get_conn()
    cur = conn.cursor()
    if snapshot_id:
        snap = cur.execute("SELECT processed_path FROM snapshots WHERE id = ?", (snapshot_id,)).fetchone()
        if not snap:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        source = snap[0]
    else:
        snap = cur.execute("SELECT processed_path FROM snapshots WHERE dataset_id = ? ORDER BY created_at DESC LIMIT 1", (dataset_id,)).fetchone()
        if not snap:
            ds = cur.execute("SELECT raw_path FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            if not ds:
                raise HTTPException(status_code=404, detail="Dataset not found")
            source = ds[0]
        else:
            source = snap[0]
    try:
        if DUCKDB_AVAILABLE:
            con = duckdb.connect()
            if source.endswith('.parquet'):
                con.execute("CREATE VIEW dataset AS SELECT * FROM parquet_scan(?)", [source])
            else:
                con.execute("CREATE VIEW dataset AS SELECT * FROM read_csv_auto(?, header=True)", [source])
            kpi = con.execute("SELECT SUM(price*quantity) as revenue FROM dataset").fetchone()[0]
            con.close()
        else:
            df = read_source_to_df(source)
            kpi = float((df['price']*df['quantity']).sum()) if {'price','quantity'}.issubset(df.columns) else 0.0
        # build simple PDF
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setTitle("Knowlance Report")
        c.setFont("Helvetica-Bold", 18)
        c.drawString(72, 720, "Knowlance Analytics Report")
        c.setFont("Helvetica", 12)
        c.drawString(72, 690, f"Generated: {datetime.utcnow().isoformat()}")
        c.drawString(72, 670, f"Dataset ID: {dataset_id}")
        c.drawString(72, 650, f"Total Revenue: {float(kpi or 0):.2f}")
        c.showPage()
        c.save()
        buffer.seek(0)
        headers = {"Content-Disposition": f"attachment; filename=report_{dataset_id}.pdf"}
        return StreamingResponse(buffer, media_type="application/pdf", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Convenience route to register sample as a dataset
@app.post("/load-sample")
def load_sample():
    sample_csv = ensure_sample_csv()
    df = pd.read_csv(sample_csv)
    dataset_id = str(uuid.uuid4())
    raw_path = os.path.join(RAW_DIR, f"{dataset_id}.csv")
    df.to_csv(raw_path, index=False)
    schema = infer_schema(df)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO datasets (id, tenant_id, name, raw_path, created_at, schema_json) VALUES (?, ?, ?, ?, ?, ?)",
        (
            dataset_id,
            'tenant_demo',
            'sales_sample.csv',
            raw_path,
            datetime.utcnow().isoformat(),
            json.dumps(schema),
        ),
    )
    conn.commit()
    conn.close()
    return {"dataset_id": dataset_id, "raw_path": raw_path, "schema": schema, "preview": df.head(10).to_dict(orient='records')}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
