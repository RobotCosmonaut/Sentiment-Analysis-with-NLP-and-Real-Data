#
#   CS 7319 Software Architecture
#   Homework 2 Bonus
#   Sentiment Analysis with NLP and Real Data
#   Ron Denny
#   rdenny@smu.edu
#   
#   This script was created in Microsoft VSCode and Claude.ai was utilized in the script development
#

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import pandas as pd
import time
from urllib.parse import urljoin, urlparse
import warnings
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt_tab', quiet=True)

class BBCTextScraper:
    def __init__(self):
        self.base_url = "https://www.bbc.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
    
    def get_article_links(self, section='news', max_links=10):
        """
        Scrape article links from BBC sections
        """
        try:
            if section == 'news':
                url = f"{self.base_url}/news"
            else:
                url = f"{self.base_url}/{section}"
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            # Find article links (BBC uses various selectors)
            article_selectors = [
                'a[href*="/news/"]',
                'a[href*="/sport/"]',
                'a[href*="/culture/"]'
            ]
            
            for selector in article_selectors:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href')
                    if href:
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            full_url = urljoin(self.base_url, href)
                        else:
                            full_url = href
                        
                        # Filter for actual article URLs
                        if '/news/' in full_url or '/sport/' in full_url or '/culture/' in full_url:
                            links.append(full_url)
                        
                        if len(links) >= max_links:
                            break
                
                if len(links) >= max_links:
                    break
            
            # Remove duplicates and return
            return list(set(links))[:max_links]
            
        except Exception as e:
            print(f"Error getting article links: {e}")
            return []
    
    def scrape_article_text(self, url):
        """
        Extract text content from a BBC article
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # BBC article selectors (may change over time)
            text_selectors = [
                'div[data-component="text-block"]',
                'p[data-component="text-block"]',
                '.story-body p',
                'article p',
                '.gel-body-copy'
            ]
            
            text_content = []
            title = ""
            
            # Extract title
            title_selectors = ['h1', '.story-headline', 'h1.story-body__h1']
            for selector in title_selectors:
                title_element = soup.select_one(selector)
                if title_element:
                    title = title_element.get_text().strip()
                    break
            
            # Extract article text
            for selector in text_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    if text and len(text) > 50:  # Filter out short snippets
                        text_content.append(text)
            
            full_text = " ".join(text_content)
            
            return {
                'url': url,
                'title': title,
                'text': full_text,
                'word_count': len(full_text.split())
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def preprocess_text(self, text):
        """
        Preprocess text: tokenization, lowercasing, stopword removal
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        filtered_tokens = [
            word for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return filtered_tokens
    
    def classify_sentiment(self, text):
        """
        Classify sentiment using VADER sentiment analyzer with Mixed category
        """
        if not text:
            return {'sentiment': 'Neutral', 'scores': {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}}
        
        # Get sentiment scores
        scores = self.sia.polarity_scores(text)
        
        # Enhanced classification with Mixed category
        pos_score = scores['pos']
        neg_score = scores['neg']
        compound = scores['compound']
        
        # Mixed: significant both positive and negative sentiment
        if pos_score > 0.3 and neg_score > 0.3:
            sentiment = 'Mixed'
        # Strong positive or negative
        elif compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'sentiment': sentiment,
            'scores': scores
        }
    
    def analyze_articles(self, section='news', max_articles=10):
        """
        Complete pipeline: scrape, preprocess, and analyze sentiment
        """
        print(f"Collecting article links from BBC {section}...")
        article_links = self.get_article_links(section, max_articles)
        
        if not article_links:
            print("No article links found!")
            return []
        
        print(f"Found {len(article_links)} articles. Scraping content...")
        
        results = []
        for i, url in enumerate(article_links, 1):
            print(f"Processing article {i}/{len(article_links)}: {url}")
            
            # Scrape article
            article_data = self.scrape_article_text(url)
            if not article_data:
                continue
            
            # Preprocess text
            processed_tokens = self.preprocess_text(article_data['text'])
            
            # Classify sentiment
            sentiment_result = self.classify_sentiment(article_data['text'])
            
            # Store results
            result = {
                'url': article_data['url'],
                'title': article_data['title'],
                'original_text': article_data['text'][:500] + "..." if len(article_data['text']) > 500 else article_data['text'],
                'word_count': article_data['word_count'],
                'processed_tokens_count': len(processed_tokens),
                'processed_tokens_sample': processed_tokens[:20],  # First 20 tokens
                'sentiment': sentiment_result['sentiment'],
                'sentiment_scores': sentiment_result['scores']
            }
            
            results.append(result)
            
            # Be respectful to the server
            time.sleep(1)
        
        return results

def load_existing_data():
    """
    Load existing data from CSV file if it exists
    """
    try:
        df = pd.read_csv('bbc_sentiment_analysis.csv')
        print(f"Loaded existing data: {len(df)} articles from bbc_sentiment_analysis.csv")
        
        # Convert sentiment_scores string back to dict if needed
        if 'sentiment_scores' in df.columns:
            import ast
            df['sentiment_scores'] = df['sentiment_scores'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        
        return df
    except FileNotFoundError:
        print("No existing data file found (bbc_sentiment_analysis.csv)")
        return None
    except Exception as e:
        print(f"Error loading existing data: {e}")
        return None

def print_summary_and_verdict(df):
    """
    Print aggregate totals and verdict as requested
    """
    # Count sentiments
    sentiment_counts = df['sentiment'].value_counts()
    
    # Get counts (default to 0 if category doesn't exist)
    positive = sentiment_counts.get('Positive', 0)
    negative = sentiment_counts.get('Negative', 0)
    mixed = sentiment_counts.get('Mixed', 0)
    neutral = sentiment_counts.get('Neutral', 0)
    
    # Determine verdict
    if positive > negative:
        verdict = "Happier"
    elif negative > positive:
        verdict = "Sadder"
    else:
        verdict = "Tied"
    
    # Print summary in requested format
    print(f"Positive={positive} Negative={negative} Mixed={mixed} Neutral={neutral} Verdict: {verdict}")
    
    return {
        'Positive': positive,
        'Negative': negative,
        'Mixed': mixed,
        'Neutral': neutral,
        'Verdict': verdict
    }

def create_bar_chart(sentiment_data):
    """
    Create a simple bar chart of sentiment distribution
    """
    # Prepare data for chart (exclude verdict)
    categories = ['Positive', 'Negative', 'Mixed', 'Neutral']
    counts = [sentiment_data[cat] for cat in categories]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']  # Green, Red, Orange, Gray
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Customize chart
    plt.title('BBC Articles Sentiment Analysis Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment Category', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Add verdict as subtitle
    verdict_text = f"Overall Verdict: {sentiment_data['Verdict']}"
    plt.figtext(0.5, 0.02, verdict_text, ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    
    # Save chart BEFORE showing it
    plt.savefig('sentiment_analysis_chart.png', dpi=300, bbox_inches='tight')
    print("Chart saved as 'sentiment_analysis_chart.png'")
    
    # Show chart (this will clear the figure when closed)
    plt.show()

def main():
    print("BBC Text Data Collection and Sentiment Analysis")
    print("=" * 50)
    
    # Check for existing data first
    df = load_existing_data()
    
    if df is not None:
        print("Using existing data from CSV file.")
        print("To scrape fresh data, delete 'bbc_sentiment_analysis.csv' and run again.")
        results = df.to_dict('records')
    else:
        print("No existing data found. Scraping fresh data from BBC.com...")
        
        # Initialize scraper
        scraper = BBCTextScraper()
        
        # You can change 'news' to other sections like 'sport', 'culture', etc.
        results = scraper.analyze_articles(section='news', max_articles=20)
        
        if not results:
            print("No articles were successfully processed.")
            return
        
        # Create DataFrame for better display
        df = pd.DataFrame(results)
    
    # Display results
    print(f"\nAnalyzed {len(results)} articles:")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\nArticle {i}:")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Word Count: {result['word_count']}")
        print(f"Processed Tokens: {result['processed_tokens_count']}")
        print(f"Sample Tokens: {result['processed_tokens_sample']}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Sentiment Scores: {result['sentiment_scores']}")
        print(f"Text Preview: {result['original_text'][:200]}...")
        print("-" * 50)
    
    # Print summary and verdict in requested format
    print(f"\n{'='*50}")
    print("SENTIMENT SUMMARY:")
    print("="*50)
    sentiment_summary = print_summary_and_verdict(df)
    
    # Detailed statistics
    print(f"\nDetailed Sentiment Distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"{sentiment}: {count} articles ({count/len(results)*100:.1f}%)")
    
    # Average sentiment scores
    avg_compound = df['sentiment_scores'].apply(lambda x: x['compound']).mean()
    avg_positive = df['sentiment_scores'].apply(lambda x: x['pos']).mean()
    avg_negative = df['sentiment_scores'].apply(lambda x: x['neg']).mean()
    avg_neutral = df['sentiment_scores'].apply(lambda x: x['neu']).mean()
    
    print(f"\nAverage Sentiment Scores:")
    print(f"Compound: {avg_compound:.3f}")
    print(f"Positive: {avg_positive:.3f}")
    print(f"Negative: {avg_negative:.3f}")
    print(f"Neutral: {avg_neutral:.3f}")
    
    # Create bar chart
    print(f"\nGenerating bar chart...")
    create_bar_chart(sentiment_summary)
    
    # Save to CSV only if we scraped new data
    if not os.path.exists('bbc_sentiment_analysis.csv'):
        df.to_csv('bbc_sentiment_analysis.csv', index=False)
        print(f"\nNew results saved to 'bbc_sentiment_analysis.csv'")

if __name__ == "__main__":
    main()