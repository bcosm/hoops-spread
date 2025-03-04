# Performs memory-efficient sentiment analysis on Reddit posts with time series EMA calculations
import duckdb
import pandas as pd
import numpy as np
import torch
import json
import time
import os
import gc
import psutil
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
import hashlib

# Configuration parameters for sentiment analysis models and processing
@dataclass
class Config:
    distilbert_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    sarcasm_model: str = "cardiffnlp/twitter-roberta-base-irony"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_length: int = 256
    ema_alpha: float = 0.1
    confidence_ema_alpha: float = 0.05
    vader_threshold: float = 0.3
    flair_threshold: float = 0.6
    sarcasm_threshold: float = 0.4
    schools_per_chunk: int = 10
    max_memory_gb: float = 4.0
    aggressive_cleanup: bool = True

# Monitors memory usage and triggers cleanup when limits are reached
class MemoryMonitor:
    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory_gb = max_memory_gb
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        return self.process.memory_info().rss / 1024**3
    
    def check_memory_warning(self) -> bool:
        memory_gb = self.get_memory_usage()
        if memory_gb > self.max_memory_gb * 0.8:
            return True
        return False
    
    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Multi-model sentiment analyzer using DistilBERT, VADER, and Flair
class SentimentAnalyzer:
                                                                     
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)
        self.content_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._load_models()
        
    def _load_models(self):
        self.vader = SentimentIntensityAnalyzer()
        
        self.flair_model = TextClassifier.load('en-sentiment')
        self.has_flair = True
        
        self.distilbert_tokenizer = AutoTokenizer.from_pretrained(self.config.distilbert_model)
        torch_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        
        self.distilbert_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.distilbert_model,
            torch_dtype=torch_dtype
        )
        self.distilbert_model.to(self.device)
        self.distilbert_model.eval()
        
        try:
            self.sarcasm_tokenizer = AutoTokenizer.from_pretrained(self.config.sarcasm_model)
            self.sarcasm_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.sarcasm_model,
                torch_dtype=torch_dtype
            )
            self.sarcasm_model.to(self.device)
            self.sarcasm_model.eval()
            self.has_sarcasm_model = True
        except Exception:
            self.has_sarcasm_model = False
    
    def _get_content_hash(self, text: str) -> str:
        normalized = text.lower().strip()
        normalized = ' '.join(normalized.split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _analyze_with_vader(self, text: str) -> Tuple[float, float, bool]:
        scores = self.vader.polarity_scores(text)
        sentiment = scores['compound']
        confidence = abs(sentiment)
        should_continue = confidence >= self.config.vader_threshold
        return sentiment, confidence, should_continue
    
    def _analyze_with_flair(self, text: str) -> Tuple[float, float, bool]:
        sentence = Sentence(text)
        self.flair_model.predict(sentence)
        label = sentence.labels[0]
        flair_confidence = label.score
        
        if label.value == 'POSITIVE':
            sentiment = flair_confidence
        else:
            sentiment = -flair_confidence
        should_continue = flair_confidence >= self.config.flair_threshold
        
        return sentiment, flair_confidence, should_continue
    
    def _analyze_with_distilbert(self, texts: List[str]) -> List[Tuple[float, float]]:
        if not texts:
            return []
        
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            inputs = self.distilbert_tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.distilbert_model(**inputs)
                
                probabilities = F.softmax(outputs.logits, dim=-1)
                probs_np = probabilities.cpu().numpy()
                
                del outputs, probabilities, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                for probs in probs_np:
                    negative = float(probs[0])
                    positive = float(probs[1])
                    sentiment = positive - negative
                    confidence = float(np.max(probs))
                    results.append((sentiment, confidence))
        
        return results
    
    def _detect_sarcasm(self, texts: List[str]) -> List[float]:
        if not self.has_sarcasm_model or not texts:
            return [0.0] * len(texts)
        
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            inputs = self.sarcasm_tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.sarcasm_model(**inputs)
                
                probabilities = F.softmax(outputs.logits, dim=-1)
                probs_np = probabilities.cpu().numpy()
                
                del outputs, probabilities, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                for probs in probs_np:
                    sarcasm_score = float(probs[1])
                    results.append(sarcasm_score)
        return results

    def analyze_texts_streaming(self, texts: List[str]) -> List[Tuple[float, float]]:
        if not texts:
            return []
        
        results = []
        vader_processed = 0
        flair_processed = 0
        distilbert_processed = 0
        sarcasm_processed = 0
        
        distilbert_texts = []
        distilbert_indices = []
        sarcasm_candidates = []
        sarcasm_indices = []
        
        for i, text in enumerate(texts):
            if not text or len(text.strip()) < 3:
                results.append((0.0, 0.0))
                continue
            
            content_hash = self._get_content_hash(text)
            if content_hash in self.content_cache:
                sentiment, confidence = self.content_cache[content_hash]
                results.append((sentiment, confidence))
                self.cache_hits += 1
                continue
            
            self.cache_misses += 1
            
            vader_sentiment, vader_confidence, continue_to_flair = self._analyze_with_vader(text)
            vader_processed += 1
            
            if not continue_to_flair:
                final_sentiment = vader_sentiment
                final_confidence = vader_confidence
            else:
                flair_sentiment, flair_confidence, continue_to_distilbert = self._analyze_with_flair(text)
                flair_processed += 1
                
                if not continue_to_distilbert:
                    final_sentiment = flair_sentiment
                    final_confidence = flair_confidence
                else:
                    distilbert_texts.append(text)
                    distilbert_indices.append(i)
                    final_sentiment = flair_sentiment
                    final_confidence = flair_confidence
                
                if abs(final_sentiment) > self.config.sarcasm_threshold:
                    sarcasm_candidates.append(text)
                    sarcasm_indices.append(i)
            
            self.content_cache[content_hash] = (final_sentiment, final_confidence)
            results.append((final_sentiment, final_confidence))
        
        if distilbert_texts:
            distilbert_results = self._analyze_with_distilbert(distilbert_texts)
            distilbert_processed = len(distilbert_texts)
            
            for idx, (sentiment, confidence) in zip(distilbert_indices, distilbert_results):
                results[idx] = (sentiment, confidence)
                content_hash = self._get_content_hash(texts[idx])
                self.content_cache[content_hash] = (sentiment, confidence)
        
        if sarcasm_candidates and self.has_sarcasm_model:
            sarcasm_scores = self._detect_sarcasm(sarcasm_candidates)
            sarcasm_processed = len(sarcasm_candidates)
            
            for idx, sarcasm_score in zip(sarcasm_indices, sarcasm_scores):
                if sarcasm_score > 0.5:
                    old_sentiment, confidence = results[idx]
                    new_sentiment = -old_sentiment * 0.8
                    results[idx] = (new_sentiment, confidence)
                    content_hash = self._get_content_hash(texts[idx])
                    self.content_cache[content_hash] = (new_sentiment, confidence)
        
        if self.config.aggressive_cleanup:
            self.memory_monitor.cleanup()
        
        return results

# Generates time series sentiment data with EMA smoothing for school mentions
class TimeSeriesGenerator:
    def __init__(self, db_path: str, config: Config):
        self.db_path = db_path
        self.config = config
        self.conn = duckdb.connect(db_path, read_only=True)
        self.analyzer = SentimentAnalyzer(config)
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)
        self._get_metadata()
    
    def _get_metadata(self):
        date_query = """
        SELECT 
            MIN(DATE_TRUNC('day', TIMESTAMP 'epoch' + created_utc * INTERVAL '1 second')) as min_date,
            MAX(DATE_TRUNC('day', TIMESTAMP 'epoch' + created_utc * INTERVAL '1 second')) as max_date,
            COUNT(*) as total_mentions
        FROM school_mentions
        """
        result = self.conn.execute(date_query).fetchone()
        self.start_date = result[0] if result[0] else date(2020, 1, 1)
        self.end_date = result[1] if result[1] else date.today()
        self.total_mentions = result[2]
        
        schools_query = "SELECT DISTINCT school_name FROM school_mentions ORDER BY school_name"
        self.schools = [row[0] for row in self.conn.execute(schools_query).fetchall()]
    
    def _load_schools_chunk(self, school_chunk: List[str]) -> pd.DataFrame:
        school_params = ', '.join(['?' for _ in school_chunk])
        
        query = f"""
        SELECT 
            sm.mention_id,
            sm.school_name,
            sm.post_id,
            sm.post_type,
            sm.mention_source,
            sm.created_utc,
            p.title,
            p.selftext,
            p.body,
            p.score,
            DATE_TRUNC('day', TIMESTAMP 'epoch' + sm.created_utc * INTERVAL '1 second') as date
        FROM school_mentions sm
        JOIN posts_full p ON sm.post_id = p.id
        WHERE sm.school_name IN ({school_params})
        ORDER BY sm.school_name, sm.created_utc
        """
        
        chunk_df = self.conn.execute(query, school_chunk).fetchdf()
        chunk_df['date'] = pd.to_datetime(chunk_df['date']).dt.date
        return chunk_df
    
    def _process_school_chunk(self, school_chunk: List[str]) -> List[pd.DataFrame]:
        chunk_df = self._load_schools_chunk(school_chunk)
        
        if len(chunk_df) == 0:
            return []
        chunk_results = []
        
        for school in school_chunk:
            school_mentions = chunk_df[chunk_df['school_name'] == school].copy()
            
            if len(school_mentions) == 0:
                continue
            
            school_ts = self._generate_school_timeseries_streaming(school, school_mentions)
            if not school_ts.empty:
                chunk_results.append(school_ts)
            
            if self.config.aggressive_cleanup:
                del school_mentions
                self.memory_monitor.cleanup()
        
        del chunk_df
        self.memory_monitor.cleanup()
        return chunk_results
    
    def _generate_school_timeseries_streaming(self, school: str, school_mentions: pd.DataFrame) -> pd.DataFrame:
        daily_groups = school_mentions.groupby('date')
        daily_results = []
        
        for date, group in daily_groups:
            sentiment, confidence, word_count = self._calculate_weighted_sentiment_streaming(group)
            daily_results.append({
                'date': date,
                'school_name': school,
                'daily_sentiment_raw': sentiment,
                'daily_confidence_raw': confidence,
                'daily_word_count': word_count,
                'mention_count': len(group)
            })
        
        if not daily_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(daily_results)
        df = df.sort_values('date')
        df = self._fill_missing_dates_and_apply_ema(df, school)
        return df
    
    def _calculate_weighted_sentiment_streaming(self, mentions_df: pd.DataFrame) -> Tuple[float, float, int]:
        if len(mentions_df) == 0:
            return 0.0, 0.0, 0
        
        texts = []
        weights = []
        
        for _, row in mentions_df.iterrows():
            if row['mention_source'] == 'title':
                mention_text = row['title'] or ''
            elif row['mention_source'] == 'selftext':
                mention_text = row['selftext'] or ''
            else:
                mention_text = row['body'] or ''
            
            if not mention_text:
                continue
            
            texts.append(mention_text)
            
            text_weight = len(mention_text.split())
            score_weight = max(1, row['score']) if pd.notna(row['score']) and row['score'] > 0 else 1
            combined_weight = text_weight * np.log(1 + score_weight)
            weights.append(combined_weight)
        
        if not texts:
            return 0.0, 0.0, 0
        
        sentiment_results = self.analyzer.analyze_texts_streaming(texts)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        sentiments = [result[0] for result in sentiment_results]
        confidences = [result[1] for result in sentiment_results]
        
        weighted_sentiment = np.average(sentiments, weights=weights)
        weighted_confidence = np.average(confidences, weights=weights)
        total_words = sum(len(text.split()) for text in texts)
        
        return weighted_sentiment, weighted_confidence, total_words
    
    def _fill_missing_dates_and_apply_ema(self, df: pd.DataFrame, school: str) -> pd.DataFrame:
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        full_df = pd.DataFrame({
            'date': [d.date() for d in date_range],
            'school_name': school
        })
        
        df = full_df.merge(df, on=['date', 'school_name'], how='left')
        
        df['daily_sentiment_raw'] = df['daily_sentiment_raw'].fillna(0.0)
        df['daily_confidence_raw'] = df['daily_confidence_raw'].fillna(0.0)
        df['daily_word_count'] = df['daily_word_count'].fillna(0)
        df['mention_count'] = df['mention_count'].fillna(0)
        
        alpha_s = self.config.ema_alpha
        alpha_c = self.config.confidence_ema_alpha
        
        df['sentiment_ema'] = df['daily_sentiment_raw'].ewm(alpha=alpha_s, adjust=False).mean()
        df['confidence_ema'] = df['daily_confidence_raw'].ewm(alpha=alpha_c, adjust=False).mean()
        
        return df
    
    def generate_all_timeseries_memory_efficient(self) -> pd.DataFrame:
        all_timeseries = []
        
        school_chunks = [
            self.schools[i:i + self.config.schools_per_chunk]
            for i in range(0, len(self.schools), self.config.schools_per_chunk)
        ]
        
        for i, chunk in enumerate(school_chunks):
            try:
                chunk_results = self._process_school_chunk(chunk)
                all_timeseries.extend(chunk_results)
                
                if self.memory_monitor.check_memory_warning():
                    self.memory_monitor.cleanup()
                
            except Exception:
                continue
        
        if not all_timeseries:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_timeseries, ignore_index=True)
        del all_timeseries
        self.memory_monitor.cleanup()
        
        combined_df = self._normalize_scores(combined_df)
        return combined_df
    
    def _normalize_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        sentiment_min = df['sentiment_ema'].min()
        sentiment_max = df['sentiment_ema'].max()
        
        if sentiment_max > sentiment_min:
            df['sentiment_normalized'] = 2 * ((df['sentiment_ema'] - sentiment_min) / (sentiment_max - sentiment_min)) - 1
        else:
            df['sentiment_normalized'] = 0.0
        
        confidence_min = df['confidence_ema'].min()
        confidence_max = df['confidence_ema'].max()
        
        if confidence_max > confidence_min:
            df['confidence_normalized'] = (df['confidence_ema'] - confidence_min) / (confidence_max - confidence_min)
        else:
            df['confidence_normalized'] = 0.5
        
        return df
    
    def export_timeseries(self, df: pd.DataFrame, output_path: str):
        base_path = output_path.replace('.csv', '')
        
        csv_path = f"{base_path}.csv"
        df.to_csv(csv_path, index=False)
        
        parquet_path = f"{base_path}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        summary_path = f"{base_path}_summary.json"
        summary = {
            'total_schools': df['school_name'].nunique(),
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat()
            },
            'total_records': len(df),
            'records_with_mentions': len(df[df['mention_count'] > 0]),
            'config': {
                'schools_per_chunk': self.config.schools_per_chunk,
                'max_memory_gb': self.config.max_memory_gb,
                'batch_size': self.config.batch_size,
                'aggressive_cleanup': self.config.aggressive_cleanup,
                'device': self.config.device,
                'distilbert_model': self.config.distilbert_model,
                'sarcasm_model': self.config.sarcasm_model,
                'vader_threshold': self.config.vader_threshold,
                'flair_threshold': self.config.flair_threshold,
                'sarcasm_threshold': self.config.sarcasm_threshold
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

# Orchestrates complete sentiment analysis pipeline with memory management
def main():
    config = Config(
        distilbert_model="distilbert-base-uncased-finetuned-sst-2-english",
        sarcasm_model="cardiffnlp/twitter-roberta-base-irony",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        max_length=256,
        vader_threshold=0.3,
        flair_threshold=0.6,
        sarcasm_threshold=0.4,
        schools_per_chunk=10,
        max_memory_gb=4.0,
        aggressive_cleanup=True,
        ema_alpha=0.1,
        confidence_ema_alpha=0.05
    )
    
    base_path = r"c:\Users\bcosm\Documents\ncaa_basketball_point_spread"
    db_path = os.path.join(base_path, "reddit_school_mentions_ultra_optimized.duckdb")
    output_path = os.path.join(base_path, "school_sentiment_timeseries.csv")
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    try:
        generator = TimeSeriesGenerator(db_path, config)
        start_time = time.time()
        
        timeseries_df = generator.generate_all_timeseries_memory_efficient()
        
        end_time = time.time()
        duration = end_time - start_time
        
        generator.export_timeseries(timeseries_df, output_path)
        
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = duration % 60
        
        print(f"Processing complete:")
        print(f"  Schools: {timeseries_df['school_name'].nunique()}")
        print(f"  Date range: {timeseries_df['date'].min()} to {timeseries_df['date'].max()}")
        print(f"  Records: {len(timeseries_df):,}")
        print(f"  Time: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        raise
    
    finally:
        if 'generator' in locals():
            generator.conn.close()

if __name__ == "__main__":
    main()
