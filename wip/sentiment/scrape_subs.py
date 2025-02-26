# Scrapes Reddit posts and comments from college basketball subreddits using DuckDB
import os
import sys
import json
import datetime
import re
import io
import subprocess
from pathlib import Path
import duckdb
from pyzstd import ZstdFile, DParameter

ROOT = Path("D:/temp_files")
DUCKDB_FILE = "reddit_local_processed.duckdb"

TARGET_SUBREDDITS = {
    "CollegeBasketball", "Basketball", "MarchMadness", "Bracketology", 
    "acc", "TheB1G", "bigxii", "BigEast", "sec", "Pac12"
}

POST_FIELDS = [
    "id", "name", "kind", "created_utc", "subreddit", "author", "title", 
    "selftext", "body", "score", "num_comments", "permalink", "url", 
    "retrieved_on", "parent_id", "author_flair_text", "distinguished", 
    "edited", "is_original_content", "is_self", "locked", "spoiler", 
    "stickied", "upvote_ratio"
]

MIN_CREATED_UTC = int(datetime.datetime(2007, 1, 1, tzinfo=datetime.timezone.utc).timestamp())

# Initializes DuckDB database with tables for posts and comments
def init_db(conn):
    field_types = {
        "id": "VARCHAR PRIMARY KEY", "name": "VARCHAR", "kind": "VARCHAR",
        "created_utc": "BIGINT", "score": "BIGINT", "num_comments": "BIGINT",
        "retrieved_on": "BIGINT", "edited": "VARCHAR", "is_original_content": "BOOLEAN",
        "is_self": "BOOLEAN", "locked": "BOOLEAN", "spoiler": "BOOLEAN",
        "stickied": "BOOLEAN", "upvote_ratio": "DOUBLE"
    }
    cols = [f"{f} {field_types.get(f, 'VARCHAR')}" for f in POST_FIELDS]
    conn.execute(f"CREATE TABLE IF NOT EXISTS posts ({', '.join(cols)})")

# Processes compressed Reddit data files and extracts relevant posts
def process_file(path: Path):
    try:
        with ZstdFile(path, "rb", level_or_option={DParameter.windowLogMax: 31}) as zf:
            with io.TextIOWrapper(zf, encoding="utf-8") as txt:
                for line in txt:
                    yield line
    except Exception:
        cmd = ["zstd", "-dcq", "--long=31", "--memory=2048MB", str(path)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with io.TextIOWrapper(proc.stdout, encoding="utf-8") as txt:
            for line in txt:
                yield line
        proc.wait()

# Ingests filtered Reddit data into DuckDB database
def ingest(path: Path, conn) -> bool:
    placeholders = ", ".join(["?"] * len(POST_FIELDS))
    sql = f"INSERT OR IGNORE INTO posts ({', '.join(POST_FIELDS)}) VALUES ({placeholders})"
    kept = 0
    
    for line in process_file(path):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        try:
            ts = int(item.get("created_utc") or 0)
        except (TypeError, ValueError):
            ts = 0
            
        if item.get("subreddit") not in TARGET_SUBREDDITS or ts < MIN_CREATED_UTC:
            continue
            
        row = [json.dumps(item.get(f)) if isinstance(item.get(f), (dict, list))
               else item.get(f) for f in POST_FIELDS]
        conn.execute(sql, row)
        kept += 1
    
    return kept > 0

# Orchestrates Reddit data processing for college basketball subreddits
def main():
    if not ROOT.is_dir():
        sys.exit("Archive root missing")
    
    db = duckdb.connect(DUCKDB_FILE)
    init_db(db)

    pat = re.compile(r"^(RS|RC)_(\d{4})-(\d{1})\.zst$", re.I)
    archives = [p for p in ROOT.rglob("*.zst") if pat.match(p.name)]
    archives.sort()

    for p in archives:
        try:
            if ingest(p, db):
                try:
                    p.unlink()
                except OSError:
                    pass
        except Exception:
            continue

    db.close()

if __name__ == "__main__":
    main()
