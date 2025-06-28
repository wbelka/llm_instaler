#!/usr/bin/env python3
"""
Convert Twitter dataset to Q&A format for training
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import random

def convert_twitter_to_qa(input_file, output_file, mode='analyze'):
    """Convert Twitter CSV/TSV to training format"""
    
    # Read data
    df = pd.read_csv(input_file, sep='\t' if input_file.endswith('.tsv') else ',')
    
    training_data = []
    
    if mode == 'analyze':
        # Mode 1: Analyze tweets (sentiment, topics, etc)
        prompts = [
            "Analyze this tweet:",
            "What is the sentiment of this tweet?",
            "What topics are discussed in this tweet?",
            "Summarize this tweet:",
            "What is the main message of this tweet?",
        ]
        
        for _, row in df.iterrows():
            text = row['text']
            if pd.notna(text) and len(text) > 20:
                prompt = random.choice(prompts)
                
                # Create expected analysis
                hashtags = eval(row['hashtags']) if pd.notna(row['hashtags']) else []
                hashtag_list = [tag['text'] for tag in hashtags] if hashtags else []
                
                analysis = f"This tweet discusses {', '.join(hashtag_list[:3]) if hashtag_list else 'various topics'}. "
                if row.get('is_retweet', False):
                    analysis += "It's a retweet. "
                analysis += f"The tweet is in {row.get('language', 'unknown')} language."
                
                training_data.append({
                    "instruction": prompt,
                    "input": text,
                    "output": analysis
                })
    
    elif mode == 'complete':
        # Mode 2: Text completion
        for _, row in df.iterrows():
            text = row['text']
            if pd.notna(text) and len(text) > 50:
                # Split tweet for completion
                words = text.split()
                if len(words) > 10:
                    split_point = len(words) // 2
                    prompt_text = ' '.join(words[:split_point])
                    completion = ' '.join(words[split_point:])
                    
                    training_data.append({
                        "instruction": "Complete this tweet:",
                        "input": prompt_text,
                        "output": completion
                    })
    
    elif mode == 'generate':
        # Mode 3: Generate similar tweets based on hashtags
        for _, row in df.iterrows():
            text = row['text']
            hashtags = eval(row['hashtags']) if pd.notna(row['hashtags']) else []
            
            if pd.notna(text) and hashtags:
                hashtag_list = [f"#{tag['text']}" for tag in hashtags]
                
                training_data.append({
                    "instruction": "Write a tweet about these topics:",
                    "input": ', '.join(hashtag_list),
                    "output": text
                })
    
    elif mode == 'qa':
        # Mode 4: Question answering about tweets
        for _, row in df.iterrows():
            text = row['text']
            if pd.notna(text) and len(text) > 30:
                # Generate questions about the tweet
                questions = []
                
                # Language question
                if pd.notna(row.get('language')):
                    questions.append({
                        "q": "What language is this tweet in?",
                        "a": f"This tweet is in {row['language']}."
                    })
                
                # Retweet question
                questions.append({
                    "q": "Is this a retweet?",
                    "a": "Yes, this is a retweet." if row.get('is_retweet', False) else "No, this is an original tweet."
                })
                
                # Hashtag question
                hashtags = eval(row['hashtags']) if pd.notna(row['hashtags']) else []
                if hashtags:
                    hashtag_list = [tag['text'] for tag in hashtags]
                    questions.append({
                        "q": "What hashtags does this tweet use?",
                        "a": f"This tweet uses the following hashtags: {', '.join(hashtag_list)}."
                    })
                
                # Add Q&A pairs
                for qa in questions:
                    training_data.append({
                        "instruction": qa["q"],
                        "input": text,
                        "output": qa["a"]
                    })
    
    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(training_data)} examples")
    print(f"Saved to: {output_file}")
    
    # Show sample
    if training_data:
        print("\nSample entry:")
        print(json.dumps(training_data[0], indent=2, ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser(description='Convert Twitter data to training format')
    parser.add_argument('input', help='Input CSV/TSV file')
    parser.add_argument('-o', '--output', help='Output JSON file', default='twitter_training.json')
    parser.add_argument('-m', '--mode', choices=['analyze', 'complete', 'generate', 'qa'], 
                       default='analyze', help='Conversion mode')
    
    args = parser.parse_args()
    
    convert_twitter_to_qa(args.input, args.output, args.mode)

if __name__ == '__main__':
    main()