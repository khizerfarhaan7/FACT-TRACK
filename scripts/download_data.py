"""
Dataset Downloader and Preprocessor for FactTrack

Downloads real news datasets and prepares them for BERT training.
Fixes bias detection by ensuring balanced, quality data.
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import config
from modules.preprocess import clean_text
import random

random.seed(config.RANDOM_SEED)


def download_agnews():
    """
    Download AG News dataset (120k news articles, 4 categories)
    """
    print("\n" + "="*60)
    print("Downloading AG News Dataset...")
    print("="*60)
    
    try:
        dataset = load_dataset("ag_news")
        
        # Combine train and test
        all_data = []
        
        for split in ['train', 'test']:
            for item in tqdm(dataset[split], desc=f"Processing {split}"):
                all_data.append({
                    'text': item['text'],
                    'category_id': item['label'],
                    'source': 'ag_news'
                })
        
        df = pd.DataFrame(all_data)
        
        # Map AG News categories to our categories
        ag_news_mapping = {
            0: 'world_news',
            1: 'sports',
            2: 'business',
            3: 'technology'
        }
        
        df['category'] = df['category_id'].map(ag_news_mapping)
        
        print(f"✓ Downloaded {len(df)} articles from AG News")
        print(f"  Categories: {df['category'].value_counts().to_dict()}")
        
        return df[['text', 'category', 'source']]
        
    except Exception as e:
        print(f"✗ Error downloading AG News: {e}")
        return pd.DataFrame()


def download_20newsgroups():
    """
    Download 20 Newsgroups dataset for additional categories
    """
    print("\n" + "="*60)
    print("Downloading 20 Newsgroups Dataset...")
    print("="*60)
    
    try:
        dataset = load_dataset("SetFit/20_newsgroups")
        
        all_data = []
        
        for split in ['train', 'test']:
            for item in tqdm(dataset[split], desc=f"Processing {split}"):
                all_data.append({
                    'text': item['text'],
                    'category_id': item['label'],
                    'label_name': item['label_text'],
                    'source': '20newsgroups'
                })
        
        df = pd.DataFrame(all_data)
        
        # Map 20 newsgroups to our categories
        newsgroup_mapping = {
            'rec.sport.baseball': 'sports',
            'rec.sport.hockey': 'sports',
            'rec.autos': 'automotive',
            'rec.motorcycles': 'automotive',
            'sci.space': 'science',
            'sci.med': 'health',
            'sci.electronics': 'technology',
            'comp.graphics': 'technology',
            'comp.sys.ibm.pc.hardware': 'technology',
            'comp.sys.mac.hardware': 'technology',
            'comp.windows.x': 'technology',
            'talk.politics.guns': 'politics',
            'talk.politics.mideast': 'politics',
            'talk.politics.misc': 'politics',
            'talk.religion.misc': 'opinion',
            'soc.religion.christian': 'opinion',
            'alt.atheism': 'opinion',
            'misc.forsale': 'business',
        }
        
        df['category'] = df['label_name'].map(newsgroup_mapping)
        df = df.dropna(subset=['category'])
        
        print(f"✓ Downloaded {len(df)} articles from 20 Newsgroups")
        print(f"  Categories: {df['category'].nunique()} unique")
        
        return df[['text', 'category', 'source']]
        
    except Exception as e:
        print(f"✗ Error downloading 20 Newsgroups: {e}")
        return pd.DataFrame()


def create_synthetic_categories():
    """
    Create synthetic data for categories not covered by public datasets
    """
    print("\n" + "="*60)
    print("Generating Synthetic Data for Additional Categories...")
    print("="*60)
    
    synthetic_data = []
    
    # Templates for each category
    templates = {
        'environment': [
            "Scientists warn about rising sea levels due to climate change. New research shows accelerating ice melt in polar regions.",
            "Renewable energy adoption reaches record highs as solar and wind power become more affordable and efficient.",
            "Deforestation rates decline in Amazon rainforest following new conservation policies and international agreements.",
            "Air quality improves in major cities after implementing stricter emission standards and promoting public transportation.",
        ],
        'crime': [
            "Police report decrease in burglary rates following community watch program implementation in residential areas.",
            "Federal agents arrest members of cybercrime ring responsible for major data breaches affecting millions.",
            "Local authorities investigate string of vehicle thefts in suburban neighborhoods, urge residents to lock cars.",
            "Court sentences white-collar criminal to prison for securities fraud and embezzlement of company funds.",
        ],
        'economy': [
            "Federal Reserve announces interest rate decision following economic data showing inflation trends and employment.",
            "Stock market indices reach new heights as investor confidence grows amid positive earnings reports.",
            "Unemployment rate drops to lowest level in years as job creation exceeds expectations across sectors.",
            "Consumer spending increases during holiday season, boosting retail sales and economic growth indicators.",
        ],
        'education': [
            "University announces expanded scholarship program to increase access for students from diverse backgrounds.",
            "School district implements new STEM curriculum to prepare students for technology-driven job market.",
            "Study finds remote learning effectiveness varies by age group and access to technology resources.",
            "Teachers receive training in latest educational methods and digital tools to enhance classroom experience.",
        ],
        'finance': [
            "Investment firms report strong quarterly returns as portfolio diversification strategies prove successful.",
            "Cryptocurrency values fluctuate amid regulatory discussions and adoption by mainstream financial institutions.",
            "Personal savings rates increase as consumers focus on financial security and emergency funds.",
            "Banking sector announces digital transformation initiatives to improve customer service and security measures.",
        ],
        'food': [
            "New restaurant opens featuring farm-to-table concept with locally sourced seasonal ingredients and organic produce.",
            "Food safety regulations updated to address modern supply chain challenges and consumer protection needs.",
            "Celebrity chef launches cookbook featuring healthy recipes and sustainable cooking practices for home cooks.",
            "Farmers market attendance grows as consumers seek fresh produce and support for local agriculture.",
        ],
        'lifestyle': [
            "Interior design trends embrace minimalism and sustainable materials for eco-friendly home decoration.",
            "Wellness experts recommend balanced approach to health combining exercise, nutrition, and mental wellbeing.",
            "Fashion week showcases latest collections featuring diverse designers and innovative sustainable fabrics.",
            "Home organization methods gain popularity as people seek to simplify living spaces and reduce clutter.",
        ],
        'local_news': [
            "City council approves budget for infrastructure improvements including road repairs and park renovations.",
            "Community center offers new programs for residents including fitness classes and educational workshops.",
            "Local business celebrates anniversary with special events and promotions for longtime customers and neighbors.",
            "School board discusses plans for facility upgrades and curriculum enhancements for upcoming academic year.",
        ],
        'opinion': [
            "Editorial argues for balanced approach to policy issues considering multiple stakeholder perspectives.",
            "Columnist reflects on societal changes and cultural shifts affecting communities across the nation.",
            "Op-ed discusses importance of civic engagement and informed participation in democratic processes.",
            "Commentary explores evolving workplace dynamics and future of remote work arrangements.",
        ],
        'real_estate': [
            "Housing market shows signs of stabilization as inventory levels improve and prices moderate in key cities.",
            "Mortgage rates fluctuate following economic indicators and central bank monetary policy decisions.",
            "Commercial real estate sector adapts to changing office space needs amid hybrid work arrangements.",
            "Property developers announce new residential projects featuring modern amenities and sustainable design.",
        ],
        'travel': [
            "Tourism industry rebounds as international travel restrictions ease and vacation bookings increase significantly.",
            "Travel guide highlights top destinations for adventure seekers featuring outdoor activities and natural wonders.",
            "Airlines expand route networks and increase flight frequencies to meet growing passenger demand worldwide.",
            "Hotel chains introduce enhanced safety protocols and contactless services for guest comfort and convenience.",
        ],
    }
    
    for category, texts in templates.items():
        for text in texts:
            synthetic_data.append({
                'text': text,
                'category': category,
                'source': 'synthetic'
            })
    
    df = pd.DataFrame(synthetic_data)
    print(f"✓ Generated {len(df)} synthetic articles")
    print(f"  Categories: {df['category'].nunique()} unique")
    
    return df


def create_bias_dataset():
    """
    Create balanced bias detection dataset
    This is CRITICAL for fixing the bias detection issue
    """
    print("\n" + "="*60)
    print("Creating Balanced Bias Detection Dataset...")
    print("="*60)
    
    # Create biased examples (various types of bias)
    biased_examples = [
        # Political bias
        "The corrupt politicians continue to ignore the will of hard-working Americans while enriching themselves.",
        "This radical socialist agenda will destroy our economy and way of life if we don't stop it now.",
        "The incompetent administration has failed at every turn, proving they're unfit to lead our great nation.",
        "These dangerous extremists are pushing policies that threaten our fundamental freedoms and values.",
        
        # Sensationalist bias
        "SHOCKING revelation exposes the TRUTH they don't want you to know about this scandal!",
        "You won't BELIEVE what happened next in this absolutely INSANE situation!",
        "BREAKING: Devastating news rocks the world as experts warn of catastrophic consequences ahead!",
        "This TERRIFYING trend is spreading like wildfire and could affect YOUR family next!",
        
        # Emotional/loaded language
        "The heartbreaking tragedy left the devastated community reeling in utter despair and hopelessness.",
        "Heroic whistleblower courageously exposes the evil wrongdoings of the villainous corporation.",
        "The barbaric attack by ruthless extremists shows their complete disregard for innocent human life.",
        "Beloved icon's stunning triumph over adversity inspires millions around the world.",
        
        # One-sided framing
        "Critics of the legislation conveniently ignore all the benefits while focusing only on minor flaws.",
        "While opponents raise baseless concerns, supporters clearly demonstrate the overwhelming advantages.",
        "The so-called experts fail to mention the obvious problems with their flawed methodology.",
        "Despite what the biased media claims, the reality is completely different from their narrative.",
        
        # Additional biased examples for balance
        "The failed policy has been an absolute disaster causing irreparable harm to countless victims.",
        "This ridiculous proposal is nothing but a thinly veiled attempt to push their extremist agenda.",
        "The incompetent leadership's reckless decisions have led to a complete catastrophe.",
        "These corrupt officials are clearly lying to the public while hiding their true intentions.",
        "The so-called reform is actually a dangerous power grab by unelected bureaucrats.",
        "This terrible idea will undoubtedly lead to chaos and destruction if implemented.",
    ]
    
    # Create non-biased examples (neutral, factual reporting)
    not_biased_examples = [
        # Factual political reporting
        "The Senate voted 62-38 to pass the infrastructure bill after three months of negotiations.",
        "The governor announced plans to increase education funding by 5% in next year's budget.",
        "Congress is scheduled to vote on the legislation next week following committee hearings.",
        "The mayor held a press conference to discuss the city's response to recent developments.",
        
        # Neutral news reporting
        "The company reported quarterly earnings that exceeded analyst expectations by 3 percent.",
        "Researchers published their findings in the peer-reviewed journal after two years of study.",
        "The weather service forecasts rain for the weekend with temperatures in the mid-60s.",
        "Local authorities confirmed that the road closure will last through the end of the month.",
        
        # Balanced reporting
        "Supporters argue the policy will create jobs, while critics express concerns about costs.",
        "The proposal has both advantages and disadvantages according to independent analysis.",
        "Experts disagree on the potential impact, with some predicting benefits and others noting risks.",
        "The initiative has received mixed reviews from stakeholders on both sides of the issue.",
        
        # Factual descriptions
        "The event attracted approximately 500 attendees according to organizer estimates.",
        "The study included 1,200 participants across five different age groups over six months.",
        "Officials reported that response times averaged 8 minutes during the previous quarter.",
        "The survey found that 62% of respondents supported the measure while 38% opposed it.",
        
        # Additional neutral examples for balance
        "The committee will review the proposal and make recommendations within 30 days.",
        "Stock prices fluctuated throughout the day, closing 1.2% higher than the previous session.",
        "The conference featured speakers from industry, academia, and government sectors.",
        "Data shows enrollment increased by 8% compared to the same period last year.",
        "The organization announced plans to open three new locations by the end of 2024.",
        "Temperatures are expected to remain above average for the remainder of the week.",
    ]
    
    # Create DataFrame
    bias_data = []
    
    for text in biased_examples:
        bias_data.append({
            'text': text,
            'bias_label': 'biased',
            'source': 'curated'
        })
    
    for text in not_biased_examples:
        bias_data.append({
            'text': text,
            'bias_label': 'not_biased',
            'source': 'curated'
        })
    
    df = pd.DataFrame(bias_data)
    
    # Balance the dataset (50/50 split)
    biased_count = (df['bias_label'] == 'biased').sum()
    not_biased_count = (df['bias_label'] == 'not_biased').sum()
    
    print(f"✓ Created bias dataset with {len(df)} examples")
    print(f"  Biased: {biased_count} ({biased_count/len(df)*100:.1f}%)")
    print(f"  Not Biased: {not_biased_count} ({not_biased_count/len(df)*100:.1f}%)")
    
    if abs(biased_count - not_biased_count) > 5:
        print("  ⚠️ WARNING: Classes are imbalanced. Will oversample during training.")
    else:
        print("  ✓ Classes are balanced!")
    
    return df


def combine_and_process_datasets():
    """
    Combine all datasets and create final training data
    """
    print("\n" + "="*60)
    print("Combining and Processing All Datasets...")
    print("="*60)
    
    # Download category datasets
    agnews_df = download_agnews()
    newsgroups_df = download_20newsgroups()
    synthetic_df = create_synthetic_categories()
    
    # Combine category data
    category_dfs = [df for df in [agnews_df, newsgroups_df, synthetic_df] if not df.empty]
    
    if category_dfs:
        category_df = pd.concat(category_dfs, ignore_index=True)
    else:
        print("✗ No category data available!")
        return None, None
    
    # Clean text
    print("\nCleaning category texts...")
    category_df['text'] = category_df['text'].apply(lambda x: clean_text(str(x)))
    
    # Filter by length
    category_df = category_df[
        (category_df['text'].str.len() >= config.MIN_TEXT_LENGTH) &
        (category_df['text'].str.len() <= config.MAX_TEXT_LENGTH)
    ]
    
    # Sample to target size if needed
    if len(category_df) > config.TARGET_CATEGORY_SAMPLES:
        category_df = category_df.groupby('category', group_keys=False).apply(
            lambda x: x.sample(min(len(x), config.TARGET_CATEGORY_SAMPLES // len(config.CATEGORIES)))
        ).reset_index(drop=True)
    
    print(f"\n✓ Final category dataset: {len(category_df)} articles")
    print(f"  Categories: {category_df['category'].nunique()} unique")
    print("\nCategory distribution:")
    print(category_df['category'].value_counts())
    
    # Create bias dataset
    bias_df = create_bias_dataset()
    
    # Expand bias dataset if needed
    if len(bias_df) < config.TARGET_BIAS_SAMPLES:
        # Oversample to reach target
        n_repeats = config.TARGET_BIAS_SAMPLES // len(bias_df) + 1
        bias_df = pd.concat([bias_df] * n_repeats, ignore_index=True)
        bias_df = bias_df.sample(n=config.TARGET_BIAS_SAMPLES, random_state=config.RANDOM_SEED)
    
    print(f"\n✓ Final bias dataset: {len(bias_df)} examples")
    print("\nBias distribution:")
    print(bias_df['bias_label'].value_counts())
    
    # Save processed datasets
    print("\nSaving processed datasets...")
    category_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, 'category_data.csv'), index=False)
    bias_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, 'bias_data.csv'), index=False)
    
    print(f"✓ Saved to {config.PROCESSED_DATA_DIR}/")
    
    return category_df, bias_df


def main():
    """
    Main function to download and prepare all data
    """
    print("\n" + "="*80)
    print("FactTrack Data Download and Preparation")
    print("="*80)
    print("\nThis will download real news datasets and prepare them for training.")
    print("This may take several minutes depending on your internet connection.")
    print("\nNote: First-time download will cache datasets for future use.")
    
    # Create directories
    for directory in [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.CACHE_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Download and process all datasets
    category_df, bias_df = combine_and_process_datasets()
    
    if category_df is not None and bias_df is not None:
        print("\n" + "="*80)
        print("✓ Data Download Complete!")
        print("="*80)
        print(f"\nCategory Dataset: {len(category_df)} articles across {category_df['category'].nunique()} categories")
        print(f"Bias Dataset: {len(bias_df)} examples (balanced 50/50)")
        print(f"\nData saved to: {config.PROCESSED_DATA_DIR}/")
        print("\nNext step: Run 'python train.py' to train the models")
    else:
        print("\n" + "="*80)
        print("✗ Data Download Failed")
        print("="*80)
        print("\nPlease check your internet connection and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

