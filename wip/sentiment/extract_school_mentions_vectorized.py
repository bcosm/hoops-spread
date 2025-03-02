# Extracts school mentions from Reddit posts using vectorized pattern matching
import duckdb
import json
import os
import re
from typing import Dict, List

# Loads school alias mappings from JSONL file
def load_aliases(jsonl_path: str) -> Dict[str, List[str]]:
    aliases = {}
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    aliases[data['school']] = data['aliases']
    except Exception:
        pass
    return aliases

# Creates regex patterns for school alias matching with word boundaries
def create_patterns(aliases: Dict[str, List[str]]) -> List[tuple]:
    patterns = []
    for school, alias_list in aliases.items():
        escaped_aliases = []
        for alias in alias_list:
            if alias and alias.strip():
                escaped = re.escape(alias.lower().strip())
                if re.match(r'^[a-zA-Z0-9\s\-\'\.]+$', alias):
                    escaped_aliases.append(f'\\b{escaped}\\b')
                else:
                    escaped_aliases.append(escaped)
        
        if escaped_aliases:
            pattern = '(' + '|'.join(escaped_aliases) + ')'
            patterns.append((school, pattern))
    
    return patterns

# Sets up DuckDB database schema for storing school mentions and statistics
def setup_database(output_db_path: str):
    conn = duckdb.connect(output_db_path)
    
    conn.execute("DROP TABLE IF EXISTS school_mentions")
    conn.execute("DROP TABLE IF EXISTS posts_full")
    conn.execute("DROP TABLE IF EXISTS school_stats")
    
    conn.execute("""
        CREATE TABLE school_mentions (
            mention_id VARCHAR PRIMARY KEY,
            school_name VARCHAR NOT NULL,
            post_id VARCHAR NOT NULL,
            post_type VARCHAR NOT NULL,
            mention_source VARCHAR NOT NULL,
            created_utc BIGINT,
            subreddit VARCHAR,
            author VARCHAR,
            score BIGINT
        )
    """)
    
    conn.execute("CREATE INDEX idx_school_mentions_school ON school_mentions(school_name)")
    conn.execute("CREATE INDEX idx_school_mentions_post ON school_mentions(post_id)")
    
    conn.execute("""
        CREATE TABLE posts_full (
            id VARCHAR PRIMARY KEY,
            name VARCHAR,
            kind VARCHAR,
            created_utc BIGINT,
            subreddit VARCHAR,
            author VARCHAR,
            title VARCHAR,
            selftext VARCHAR,
            body VARCHAR,
            score BIGINT,
            num_comments BIGINT,
            permalink VARCHAR,
            url VARCHAR,
            retrieved_on BIGINT,
            parent_id VARCHAR,
            author_flair_text VARCHAR,
            distinguished VARCHAR,
            edited VARCHAR,
            is_original_content BOOLEAN,
            is_self BOOLEAN,
            locked BOOLEAN,
            spoiler BOOLEAN,
            stickied BOOLEAN,
            upvote_ratio DOUBLE
        )
    """)
    
    conn.execute("CREATE INDEX idx_posts_full_id ON posts_full(id)")
    
    conn.execute("""
        CREATE TABLE school_stats (
            school_name VARCHAR PRIMARY KEY,
            total_mentions BIGINT,
            unique_posts BIGINT,
            unique_comments BIGINT,
            earliest_mention BIGINT,
            latest_mention BIGINT,
            top_subreddits VARCHAR
        )
    """)
    
    return conn

def process_extraction(input_db_path: str, output_db_path: str, aliases: Dict[str, List[str]]):
    output_conn = setup_database(output_db_path)
    output_conn.execute(f"ATTACH '{input_db_path}' AS input_db (READ_ONLY)")
    
    output_conn.execute("INSERT INTO posts_full SELECT * FROM input_db.posts")
    
    patterns = create_patterns(aliases)
    
    for school_name, pattern in patterns:
        try:
            safe_school_id = re.sub(r'[^a-zA-Z0-9_]', '_', school_name)
            
            # Title mentions
            output_conn.execute(f"""
                INSERT INTO school_mentions (mention_id, school_name, post_id, post_type, mention_source, created_utc, subreddit, author, score)
                SELECT 
                    id || '_title_' || '{safe_school_id}' as mention_id,
                    ? as school_name,
                    id as post_id,
                    'post' as post_type,
                    'title' as mention_source,
                    created_utc,
                    subreddit,
                    author,
                    score
                FROM posts_full 
                WHERE title IS NOT NULL AND regexp_matches(LOWER(title), ?, 'i')
            """, [school_name, pattern])
            
            # Selftext mentions
            output_conn.execute(f"""
                INSERT INTO school_mentions (mention_id, school_name, post_id, post_type, mention_source, created_utc, subreddit, author, score)
                SELECT 
                    id || '_selftext_' || '{safe_school_id}' as mention_id,
                    ? as school_name,
                    id as post_id,
                    'post' as post_type,
                    'selftext' as mention_source,
                    created_utc,
                    subreddit,
                    author,
                    score
                FROM posts_full 
                WHERE selftext IS NOT NULL AND regexp_matches(LOWER(selftext), ?, 'i')
            """, [school_name, pattern])
            
            # Body mentions
            output_conn.execute(f"""
                INSERT INTO school_mentions (mention_id, school_name, post_id, post_type, mention_source, created_utc, subreddit, author, score)
                SELECT 
                    id || '_body_' || '{safe_school_id}' as mention_id,
                    ? as school_name,
                    id as post_id,
                    CASE WHEN parent_id IS NOT NULL THEN 'comment' ELSE 'post' END as post_type,
                    'body' as mention_source,
                    created_utc,
                    subreddit,
                    author,
                    score
                FROM posts_full 
                WHERE body IS NOT NULL AND regexp_matches(LOWER(body), ?, 'i')
            """, [school_name, pattern])
            
        except Exception:
            continue
    
    # Generate stats
    output_conn.execute("""
        INSERT INTO school_stats (school_name, total_mentions, unique_posts, unique_comments, 
                                earliest_mention, latest_mention, top_subreddits)
        SELECT 
            school_name,
            COUNT(*) as total_mentions,
            COUNT(DISTINCT CASE WHEN post_type = 'post' THEN post_id END) as unique_posts,
            COUNT(DISTINCT CASE WHEN post_type = 'comment' THEN post_id END) as unique_comments,
            MIN(created_utc) as earliest_mention,
            MAX(created_utc) as latest_mention,
            '[]' as top_subreddits
        FROM school_mentions
        GROUP BY school_name
    """)
    
    # Update subreddit stats
    schools = output_conn.execute("SELECT school_name FROM school_stats WHERE total_mentions > 0").fetchall()
    for (school,) in schools:
        top_subreddits = output_conn.execute("""
            SELECT subreddit, COUNT(*) as mentions
            FROM school_mentions 
            WHERE school_name = ?
            GROUP BY subreddit
            ORDER BY mentions DESC
            LIMIT 10
        """, [school]).fetchall()
        
        top_subreddits_json = json.dumps([{"subreddit": sr, "mentions": count} for sr, count in top_subreddits])
        output_conn.execute("UPDATE school_stats SET top_subreddits = ? WHERE school_name = ?", 
                           [top_subreddits_json, school])
    
    output_conn.execute("DETACH input_db")
    output_conn.close()

# Orchestrates school mention extraction from Reddit data using aliases
def main():
    input_db = "reddit_local_processed.duckdb"
    output_db = "reddit_school_mentions_ultra_optimized.duckdb"
    aliases_path = os.path.join("data", "processed_with_school_codes", "school_aliases_normalized.jsonl")
    
    if not os.path.exists(input_db) or not os.path.exists(aliases_path):
        return
    
    aliases = load_aliases(aliases_path)
    if aliases:
        process_extraction(input_db, output_db, aliases)

if __name__ == "__main__":
    main()
