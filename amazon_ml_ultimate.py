"""
Amazon ML Challenge 2025 - ULTIMATE SOLUTION
Target: SMAPE < 15% (Competition Level Performance)
✓ Multi-Level Ensemble with Advanced Features
✓ Comprehensive Hyperparameter Tuning
✓ Advanced Feature Engineering (1000+ features)
✓ Production-Ready Performance
"""

import os
import pandas as pd
import numpy as np
import re
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.neural_network import MLPRegressor

# Advanced models
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False


def calculate_smape(actual, predicted):
    """Calculate SMAPE - The challenge evaluation metric"""
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    diff = np.abs(actual - predicted)
    mask = denominator != 0
    smape = np.zeros_like(actual, dtype=float)
    smape[mask] = diff[mask] / denominator[mask]
    return np.mean(smape) * 100


class UltimateFeatureEngineer:
    """Ultimate feature engineering with 1000+ optimized features"""
    
    def __init__(self):
        # Multiple TF-IDF configurations for comprehensive text analysis
        self.tfidf_word = TfidfVectorizer(
            max_features=400,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words='english',
            sublinear_tf=True,
            lowercase=True
        )
        
        self.tfidf_char = TfidfVectorizer(
            max_features=200,
            analyzer='char',
            ngram_range=(3, 6),
            min_df=2
        )
        
        self.tfidf_bigram = TfidfVectorizer(
            max_features=300,
            ngram_range=(2, 2),
            min_df=3,
            max_df=0.9,
            stop_words='english'
        )
        
        self.count_vec = CountVectorizer(
            max_features=150,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8
        )
        
        # Dimensionality reduction
        self.svd_word = TruncatedSVD(n_components=100, random_state=42)
        self.svd_char = TruncatedSVD(n_components=60, random_state=42)
        self.pca = PCA(n_components=50, random_state=42)
        
        # Enhanced brand and category lists
        self.premium_brands = [
            'apple', 'samsung', 'sony', 'lg', 'dell', 'hp', 'lenovo', 'asus',
            'canon', 'nikon', 'panasonic', 'philips', 'bosch', 'nike', 'adidas',
            'puma', 'logitech', 'microsoft', 'intel', 'amd', 'nvidia', 'beats',
            'bose', 'jbl', 'sennheiser', 'gucci', 'prada', 'louis vuitton',
            'tiffany', 'rolex', 'omega', 'cartier', 'hermes', 'chanel', 'dior'
        ]
        
        self.categories = {
            'electronics': ['laptop', 'phone', 'tablet', 'computer', 'camera', 
                          'tv', 'monitor', 'headphone', 'speaker', 'mouse', 'keyboard',
                          'smartphone', 'ipad', 'macbook', 'wireless', 'bluetooth',
                          'gaming', 'console', 'playstation', 'xbox', 'nintendo'],
            'clothing': ['shirt', 'pants', 'jeans', 'dress', 'shoes', 'jacket', 
                        'sweater', 'hoodie', 't-shirt', 'sneakers', 'boots',
                        'watch', 'jewelry', 'ring', 'necklace', 'bracelet'],
            'home': ['furniture', 'chair', 'table', 'sofa', 'bed', 'lamp', 
                    'kitchen', 'decor', 'pillow', 'blanket', 'curtain',
                    'appliance', 'refrigerator', 'washer', 'dryer'],
            'beauty': ['makeup', 'cosmetic', 'perfume', 'shampoo', 'lotion', 
                      'cream', 'serum', 'foundation', 'lipstick', 'skincare'],
            'sports': ['fitness', 'gym', 'exercise', 'yoga', 'sports', 'outdoor',
                      'running', 'training', 'equipment', 'weights', 'bike'],
            'toys': ['toy', 'game', 'puzzle', 'doll', 'lego', 'action figure'],
            'books': ['book', 'novel', 'textbook', 'magazine', 'kindle'],
            'automotive': ['car', 'bike', 'vehicle', 'automotive', 'motor', 'tire'],
            'food': ['food', 'snack', 'chocolate', 'candy', 'drink', 'beverage',
                    'coffee', 'tea', 'sauce', 'spice', 'cooking', 'wine'],
            'health': ['medicine', 'vitamin', 'supplement', 'health', 'medical'],
            'tools': ['tool', 'drill', 'saw', 'hammer', 'screwdriver', 'wrench'],
            'office': ['office', 'desk', 'chair', 'printer', 'paper', 'pen']
        }
        
        self.fitted = False
    
    def extract_ipq(self, text):
        """Enhanced IPQ extraction with comprehensive patterns"""
        text_lower = str(text).lower()
        
        patterns = [
            r'ipq[:\s]*(\d+)',
            r'pack of (\d+)',
            r'(\d+)\s*pack',
            r'quantity[:\s]*(\d+)',
            r'count[:\s]*(\d+)',
            r'set of (\d+)',
            r'(\d+)\s*piece',
            r'(\d+)\s*pcs',
            r'(\d+)\s*pc',
            r'(\d+)\s*count',
            r'(\d+)\s*units?',
            r'(\d+)\s*items?',
            r'bundle of (\d+)',
            r'lot of (\d+)',
            r'(\d+)\s*in\s*1',
            r'(\d+)\s*x\s*\d+',
            r'(\d+)\s*per\s*pack',
            r'(\d+)\s*per\s*box',
            r'(\d+)\s*per\s*case',
            r'(\d+)\s*pieces?',
            r'(\d+)\s*bottles?',
            r'(\d+)\s*cans?',
            r'(\d+)\s*tubes?',
            r'(\d+)\s*jars?'
        ]
        
        ipq = 1
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                qty = int(match.group(1))
                ipq = min(qty, 1000)
                break
        
        return ipq
    
    def extract_price_indicators(self, text):
        """Extract comprehensive price-related indicators"""
        text_lower = str(text).lower()
        features = {}
        
        # Storage capacity
        storage_patterns = [
            r'(\d+\.?\d*)\s*(gb|tb|mb)', r'(\d+\.?\d*)\s*(gigabyte|terabyte|megabyte)'
        ]
        storage_value = 0
        for pattern in storage_patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                if 'tb' in unit or 'terabyte' in unit:
                    value *= 1024
                elif 'mb' in unit or 'megabyte' in unit:
                    value /= 1024
                storage_value = max(storage_value, value)
        
        features['storage_gb'] = storage_value
        features['has_storage'] = int(storage_value > 0)
        features['storage_log'] = np.log1p(storage_value)
        
        # RAM
        ram_match = re.search(r'(\d+)\s*(gb|gigabyte)\s*ram', text_lower)
        features['ram_gb'] = int(ram_match.group(1)) if ram_match else 0
        features['has_ram'] = int(features['ram_gb'] > 0)
        
        # Screen size
        screen_match = re.search(r'(\d+\.?\d*)\s*inch', text_lower)
        features['screen_inches'] = float(screen_match.group(1)) if screen_match else 0
        features['has_screen'] = int(features['screen_inches'] > 0)
        
        # Weight
        weight_patterns = [
            r'(\d+\.?\d*)\s*(kg|kilogram)', r'(\d+\.?\d*)\s*(g|gram)', 
            r'(\d+\.?\d*)\s*(lb|pound)', r'(\d+\.?\d*)\s*(oz|ounce)'
        ]
        weight_kg = 0
        for pattern in weight_patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                if 'g' in unit and 'kg' not in unit:
                    value /= 1000
                elif 'lb' in unit or 'pound' in unit:
                    value *= 0.453592
                elif 'oz' in unit or 'ounce' in unit:
                    value *= 0.0283495
                weight_kg = max(weight_kg, value)
        
        features['weight_kg'] = weight_kg
        features['has_weight'] = int(weight_kg > 0)
        features['weight_log'] = np.log1p(weight_kg)
        
        # Volume/Capacity
        volume_patterns = [
            r'(\d+\.?\d*)\s*(ml|milliliter)', r'(\d+\.?\d*)\s*(l|liter)', 
            r'(\d+\.?\d*)\s*(fl oz|fluid ounce)', r'(\d+\.?\d*)\s*(oz|ounce)'
        ]
        volume_l = 0
        for pattern in volume_patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                if 'ml' in unit:
                    value /= 1000
                elif 'fl oz' in unit or 'fluid ounce' in unit:
                    value *= 0.0295735
                elif 'oz' in unit and 'fl oz' not in unit:
                    value *= 0.0295735
                volume_l = max(volume_l, value)
        
        features['volume_l'] = volume_l
        features['has_volume'] = int(volume_l > 0)
        features['volume_log'] = np.log1p(volume_l)
        
        # Megapixels
        mp_match = re.search(r'(\d+)\s*mp', text_lower)
        features['megapixels'] = int(mp_match.group(1)) if mp_match else 0
        features['has_megapixels'] = int(features['megapixels'] > 0)
        
        # Battery capacity
        battery_match = re.search(r'(\d+)\s*(mah|mah)', text_lower)
        features['battery_mah'] = int(battery_match.group(1)) if battery_match else 0
        features['has_battery'] = int(features['battery_mah'] > 0)
        
        # Resolution
        features['is_4k'] = int('4k' in text_lower or '2160p' in text_lower)
        features['is_hd'] = int('hd' in text_lower or '1080p' in text_lower)
        features['is_8k'] = int('8k' in text_lower or '4320p' in text_lower)
        
        # Connectivity
        features['has_wireless'] = int('wireless' in text_lower or 'wifi' in text_lower)
        features['has_bluetooth'] = int('bluetooth' in text_lower)
        features['has_usb'] = int('usb' in text_lower)
        features['has_nfc'] = int('nfc' in text_lower)
        
        return features
    
    def extract_quality_features(self, text):
        """Extract comprehensive quality indicators"""
        text_lower = str(text).lower()
        features = {}
        
        # Brand analysis
        features['is_premium_brand'] = int(any(brand in text_lower for brand in self.premium_brands))
        features['brand_count'] = sum(1 for brand in self.premium_brands if brand in text_lower)
        
        # Price tier indicators
        luxury_indicators = ['luxury', 'premium', 'deluxe', 'exclusive', 'limited edition', 'professional']
        budget_indicators = ['budget', 'economy', 'basic', 'starter', 'entry-level', 'affordable', 'value']
        
        features['luxury_count'] = sum(indicator in text_lower for indicator in luxury_indicators)
        features['budget_count'] = sum(indicator in text_lower for indicator in budget_indicators)
        features['is_luxury'] = int(features['luxury_count'] > features['budget_count'])
        
        # Quality terms
        quality_terms = ['warranty', 'guarantee', 'certified', 'authentic', 'original', 'genuine', 'professional']
        features['quality_count'] = sum(term in text_lower for term in quality_terms)
        features['has_warranty'] = int('warranty' in text_lower or 'guarantee' in text_lower)
        
        # Technology indicators
        tech_terms = ['smart', 'digital', 'electronic', 'automatic', 'programmable', 'touchscreen', 'led', 'oled', 'lcd']
        features['tech_count'] = sum(term in text_lower for term in tech_terms)
        features['is_tech_product'] = int(features['tech_count'] > 2)
        
        # Material indicators
        materials = {
            'metal': ['metal', 'steel', 'aluminum', 'brass', 'copper', 'titanium'],
            'premium_metal': ['gold', 'silver', 'platinum', 'stainless steel'],
            'wood': ['wood', 'wooden', 'oak', 'pine', 'mahogany', 'walnut'],
            'leather': ['leather', 'genuine leather', 'full grain'],
            'plastic': ['plastic', 'pvc', 'acrylic', 'polycarbonate'],
            'glass': ['glass', 'crystal', 'sapphire'],
            'fabric': ['cotton', 'silk', 'wool', 'cashmere', 'linen']
        }
        
        for material_type, keywords in materials.items():
            features[f'material_{material_type}'] = int(any(kw in text_lower for kw in keywords))
        
        features['premium_material_count'] = sum([
            features['material_premium_metal'], features['material_leather'], 
            features['material_wood'], features['material_fabric']
        ])
        
        return features
    
    def extract_category_features(self, text):
        """Extract enhanced category features"""
        text_lower = str(text).lower()
        features = {}
        
        # Category detection with confidence
        for category, keywords in self.categories.items():
            match_count = sum(1 for kw in keywords if kw in text_lower)
            features[f'cat_{category}'] = int(match_count > 0)
            features[f'cat_{category}_strength'] = match_count
            features[f'cat_{category}_confidence'] = match_count / len(keywords) if keywords else 0
        
        # Primary category
        max_category = max(self.categories.keys(), 
                          key=lambda cat: features[f'cat_{cat}_strength'])
        features['primary_category_strength'] = features[f'cat_{max_category}_strength']
        features['primary_category_confidence'] = features[f'cat_{max_category}_confidence']
        
        # Category combinations
        electronics_terms = sum(1 for cat in ['electronics', 'sports', 'tools'] if features[f'cat_{cat}'])
        features['high_tech_category'] = int(electronics_terms > 0)
        
        luxury_terms = sum(1 for cat in ['clothing', 'beauty', 'home'] if features[f'cat_{cat}'])
        features['lifestyle_category'] = int(luxury_terms > 0)
        
        return features
    
    def extract_text_statistics(self, text):
        """Comprehensive text statistics"""
        text = str(text)
        text_lower = text.lower()
        words = text.split()
        chars = list(text)
        
        features = {}
        
        # Basic statistics
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        features['char_count'] = len(chars)
        features['unique_words'] = len(set(words))
        features['unique_chars'] = len(set(chars))
        
        # Ratios
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        features['unique_char_ratio'] = len(set(chars)) / len(chars) if chars else 0
        
        # Character analysis
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['lowercase_ratio'] = sum(1 for c in text if c.islower()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['special_char_ratio'] = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text) if text else 0
        features['space_ratio'] = sum(1 for c in text if c.isspace()) / len(text) if text else 0
        
        # Word analysis
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['max_word_length'] = max([len(w) for w in words]) if words else 0
        features['min_word_length'] = min([len(w) for w in words]) if words else 0
        features['word_length_std'] = np.std([len(w) for w in words]) if len(words) > 1 else 0
        
        # Sentence analysis
        sentences = re.split(r'[.!?]+', text)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
        
        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['comma_count'] = text.count(',')
        features['period_count'] = text.count('.')
        features['colon_count'] = text.count(':')
        features['semicolon_count'] = text.count(';')
        
        # Advanced text features
        features['has_bullet_points'] = int('bullet point' in text_lower)
        features['has_product_description'] = int('product description' in text_lower)
        features['has_value_unit'] = int('value:' in text_lower and 'unit:' in text_lower)
        features['has_specifications'] = int('specification' in text_lower)
        
        # IPQ features
        ipq = self.extract_ipq(text)
        features['ipq'] = ipq
        features['ipq_log'] = np.log1p(ipq)
        features['ipq_squared'] = ipq ** 2
        features['ipq_sqrt'] = np.sqrt(ipq)
        features['ipq_is_1'] = int(ipq == 1)
        features['ipq_is_multiple'] = int(ipq > 1)
        features['ipq_is_large'] = int(ipq > 10)
        
        # Numeric analysis
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            numbers = [float(n) for n in numbers if float(n) < 1000000]
            features['num_count'] = len(numbers)
            features['num_max'] = max(numbers) if numbers else 0
            features['num_min'] = min(numbers) if numbers else 0
            features['num_mean'] = np.mean(numbers) if numbers else 0
            features['num_median'] = np.median(numbers) if numbers else 0
            features['num_std'] = np.std(numbers) if len(numbers) > 1 else 0
            features['num_range'] = max(numbers) - min(numbers) if len(numbers) > 1 else 0
            features['num_large_count'] = sum(1 for n in numbers if n > 100)
            features['num_small_count'] = sum(1 for n in numbers if n < 10)
        else:
            features.update({
                'num_count': 0, 'num_max': 0, 'num_min': 0, 
                'num_mean': 0, 'num_median': 0, 'num_std': 0,
                'num_range': 0, 'num_large_count': 0, 'num_small_count': 0
            })
        
        return features
    
    def prepare_features(self, df, fit=False):
        """Prepare complete feature matrix with 1000+ features"""
        print(f"🔧 Engineering 1000+ features for {len(df):,} samples...")
        
        # Handcrafted features
        all_features = []
        for text in df['catalog_content']:
            features = {}
            features.update(self.extract_text_statistics(text))
            features.update(self.extract_price_indicators(text))
            features.update(self.extract_quality_features(text))
            features.update(self.extract_category_features(text))
            all_features.append(features)
        
        text_features_df = pd.DataFrame(all_features)
        
        # Text vectorization
        if fit:
            tfidf_word_matrix = self.tfidf_word.fit_transform(df['catalog_content'].fillna(''))
            tfidf_char_matrix = self.tfidf_char.fit_transform(df['catalog_content'].fillna(''))
            tfidf_bigram_matrix = self.tfidf_bigram.fit_transform(df['catalog_content'].fillna(''))
            count_matrix = self.count_vec.fit_transform(df['catalog_content'].fillna(''))
            
            # Apply dimensionality reduction
            tfidf_word_svd = self.svd_word.fit_transform(tfidf_word_matrix)
            tfidf_char_svd = self.svd_char.fit_transform(tfidf_char_matrix)
            
            # PCA on handcrafted features
            pca_features = self.pca.fit_transform(text_features_df.fillna(0))
            
            self.fitted = True
        else:
            tfidf_word_matrix = self.tfidf_word.transform(df['catalog_content'].fillna(''))
            tfidf_char_matrix = self.tfidf_char.transform(df['catalog_content'].fillna(''))
            tfidf_bigram_matrix = self.tfidf_bigram.transform(df['catalog_content'].fillna(''))
            count_matrix = self.count_vec.transform(df['catalog_content'].fillna(''))
            
            tfidf_word_svd = self.svd_word.transform(tfidf_word_matrix)
            tfidf_char_svd = self.svd_char.transform(tfidf_char_matrix)
            pca_features = self.pca.transform(text_features_df.fillna(0))
        
        # Create DataFrames
        tfidf_word_df = pd.DataFrame(
            tfidf_word_matrix.toarray(),
            columns=[f'tfidf_w_{i}' for i in range(tfidf_word_matrix.shape[1])]
        )
        
        tfidf_char_df = pd.DataFrame(
            tfidf_char_matrix.toarray(),
            columns=[f'tfidf_c_{i}' for i in range(tfidf_char_matrix.shape[1])]
        )
        
        tfidf_bigram_df = pd.DataFrame(
            tfidf_bigram_matrix.toarray(),
            columns=[f'tfidf_b_{i}' for i in range(tfidf_bigram_matrix.shape[1])]
        )
        
        count_df = pd.DataFrame(
            count_matrix.toarray(),
            columns=[f'count_{i}' for i in range(count_matrix.shape[1])]
        )
        
        tfidf_word_svd_df = pd.DataFrame(
            tfidf_word_svd,
            columns=[f'svd_w_{i}' for i in range(tfidf_word_svd.shape[1])]
        )
        
        tfidf_char_svd_df = pd.DataFrame(
            tfidf_char_svd,
            columns=[f'svd_c_{i}' for i in range(tfidf_char_svd.shape[1])]
        )
        
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f'pca_{i}' for i in range(pca_features.shape[1])]
        )
        
        # Combine all features
        features = pd.concat([
            text_features_df.reset_index(drop=True),
            tfidf_word_df.reset_index(drop=True),
            tfidf_char_df.reset_index(drop=True),
            tfidf_bigram_df.reset_index(drop=True),
            count_df.reset_index(drop=True),
            tfidf_word_svd_df.reset_index(drop=True),
            tfidf_char_svd_df.reset_index(drop=True),
            pca_df.reset_index(drop=True)
        ], axis=1)
        
        print(f"✓ Generated {features.shape[1]} features")
        return features


class UltimateEnsembleModel:
    """Ultimate ensemble model with advanced techniques"""
    
    def __init__(self, use_tuning=True):
        self.feature_engineer = UltimateFeatureEngineer()
        self.scaler = RobustScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        self.models = []
        self.stacking_models = []
        self.weights = []
        self.use_tuning = use_tuning and HAS_OPTUNA
        self.use_log_transform = True
        
    def optimize_lgbm(self, X_train, y_train, X_val, y_val):
        """Optimize LightGBM with comprehensive search"""
        if not self.use_tuning:
            return {
                'n_estimators': 1500,
                'learning_rate': 0.02,
                'max_depth': 15,
                'num_leaves': 100,
                'min_child_samples': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 1.0,
                'reg_lambda': 2.0,
                'random_state': 42,
                'verbose': -1
            }
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 1000, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 10, 20),
                'num_leaves': trial.suggest_int('num_leaves', 50, 200),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
                'random_state': 42,
                'verbose': -1
            }
            
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            y_val_orig = np.expm1(y_val)
            pred_orig = np.expm1(pred)
            smape = calculate_smape(y_val_orig, pred_orig)
            
            return smape
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)
        return study.best_params
    
    def optimize_xgb(self, X_train, y_train, X_val, y_val):
        """Optimize XGBoost with comprehensive search"""
        if not self.use_tuning:
            return {
                'n_estimators': 1500,
                'learning_rate': 0.02,
                'max_depth': 15,
                'min_child_weight': 1,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'gamma': 1.0,
                'reg_alpha': 1.0,
                'reg_lambda': 3.0,
                'random_state': 42,
                'verbosity': 0
            }
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 1000, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 10, 20),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 3.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                'random_state': 42,
                'verbosity': 0
            }
            
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            y_val_orig = np.expm1(y_val)
            pred_orig = np.expm1(pred)
            smape = calculate_smape(y_val_orig, pred_orig)
            
            return smape
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)
        return study.best_params
    
    def train_stacking_models(self, X_train, y_train, X_val, y_val):
        """Train advanced stacking meta-learners"""
        stacking_models = [
            ('Ridge', Ridge(alpha=0.1)),
            ('ElasticNet', ElasticNet(alpha=0.01, l1_ratio=0.3)),
            ('Huber', HuberRegressor(epsilon=1.2, max_iter=300)),
            ('MLP', MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=1000, random_state=42)),
            ('RandomForest', RandomForestRegressor(n_estimators=200, random_state=42)),
            ('ExtraTrees', ExtraTreesRegressor(n_estimators=200, random_state=42))
        ]
        
        best_model = None
        best_score = float('inf')
        
        for name, model in stacking_models:
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            y_val_orig = np.expm1(y_val)
            pred_orig = np.expm1(pred)
            smape = calculate_smape(y_val_orig, pred_orig)
            
            if smape < best_score:
                best_score = smape
                best_model = (name, model)
        
        return best_model
    
    def train(self, train_df, validation_split=0.15):
        """Train ultimate ensemble model"""
        print("\n" + "="*80)
        print("🚀 AMAZON ML CHALLENGE - ULTIMATE SOLUTION TRAINING")
        print("🎯 TARGET: SMAPE < 15%")
        print("="*80)
        
        start_time = time.time()
        
        # Feature engineering
        X = self.feature_engineer.prepare_features(train_df, fit=True)
        y = train_df['price'].values
        
        # Advanced preprocessing
        if self.use_log_transform:
            y = np.log1p(y)
            print("✓ Using log1p transformation for target")
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=True
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Power transformation
        X_train_transformed = self.power_transformer.fit_transform(X_train_scaled)
        X_val_transformed = self.power_transformer.transform(X_val_scaled)
        
        print(f"\n📊 Advanced Data Split:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        print(f"   Features: {X_train.shape[1]:,}")
        
        # Hyperparameter optimization
        model_configs = []
        
        if HAS_LGBM:
            print(f"\n🔧 {'Optimizing' if self.use_tuning else 'Configuring'} LightGBM...")
            lgbm_params = self.optimize_lgbm(X_train_transformed, y_train, X_val_transformed, y_val)
            model_configs.append(('LightGBM', LGBMRegressor(**lgbm_params)))
        
        if HAS_XGB:
            print(f"\n🔧 {'Optimizing' if self.use_tuning else 'Configuring'} XGBoost...")
            xgb_params = self.optimize_xgb(X_train_transformed, y_train, X_val_transformed, y_val)
            model_configs.append(('XGBoost', XGBRegressor(**xgb_params)))
        
        if HAS_CATBOOST:
            catboost_params = {
                'iterations': 1500,
                'learning_rate': 0.02,
                'depth': 12,
                'l2_leaf_reg': 5,
                'random_seed': 42,
                'verbose': False
            }
            model_configs.append(('CatBoost', CatBoostRegressor(**catboost_params)))
        
        if not model_configs:
            raise RuntimeError("No models available!")
        
        # Train each model
        val_predictions = []
        model_scores = []
        
        for name, model in model_configs:
            print(f"\n{'='*80}")
            print(f"📊 Training {name}...")
            print(f"{'='*80}")
            
            model_start = time.time()
            model.fit(X_train_transformed, y_train)
            model_time = time.time() - model_start
            
            self.models.append((name, model))
            
            # Validate
            pred = model.predict(X_val_transformed)
            val_predictions.append(pred)
            
            if self.use_log_transform:
                y_val_orig = np.expm1(y_val)
                pred_orig = np.expm1(pred)
                smape = calculate_smape(y_val_orig, pred_orig)
                mae = mean_absolute_error(y_val_orig, pred_orig)
            else:
                smape = calculate_smape(y_val, pred)
                mae = mean_absolute_error(y_val, pred)
            
            model_scores.append(smape)
            
            print(f"   ✓ Validation SMAPE: {smape:.4f}%")
            print(f"   ✓ Validation MAE: ${mae:.2f}")
            print(f"   ✓ Training time: {model_time:.1f}s")
        
        # Advanced ensemble optimization
        print(f"\n{'='*80}")
        print("🔧 Ultimate Ensemble Optimization...")
        print(f"{'='*80}")
        
        # Stacking
        if len(val_predictions) > 1:
            stacking_features = np.column_stack(val_predictions)
            best_stacking_model = self.train_stacking_models(stacking_features, y_train, stacking_features, y_val)
            self.stacking_models.append(best_stacking_model)
            
            # Evaluate stacking
            stacking_pred = best_stacking_model[1].predict(stacking_features)
            if self.use_log_transform:
                stacking_pred_orig = np.expm1(stacking_pred)
                y_val_orig = np.expm1(y_val)
                stacking_smape = calculate_smape(y_val_orig, stacking_pred_orig)
            else:
                stacking_smape = calculate_smape(y_val, stacking_pred)
            
            print(f"   ✓ Best stacking model: {best_stacking_model[0]}")
            print(f"   ✓ Stacking SMAPE: {stacking_smape:.4f}%")
            
            # Weighted ensemble
            best_smape = float('inf')
            best_weights = None
            
            for w1 in np.linspace(0.1, 0.9, 9):
                for w2 in np.linspace(0.1, 0.9, 9):
                    if len(val_predictions) == 2:
                        weights = [w1, 1-w1]
                    else:
                        w3 = 1 - w1 - w2
                        if w3 < 0 or w3 > 1:
                            continue
                        weights = [w1, w2, w3]
                    
                    ensemble_pred = sum(w * pred for w, pred in zip(weights, val_predictions))
                    
                    if self.use_log_transform:
                        ensemble_pred_orig = np.expm1(ensemble_pred)
                        y_val_orig = np.expm1(y_val)
                        smape = calculate_smape(y_val_orig, ensemble_pred_orig)
                    else:
                        smape = calculate_smape(y_val, ensemble_pred)
                    
                    if smape < best_smape:
                        best_smape = smape
                        best_weights = weights
            
            self.weights = best_weights
            print(f"   ✓ Best weights: {[f'{w:.3f}' for w in self.weights]}")
            print(f"   ✓ Weighted Ensemble SMAPE: {best_smape:.4f}%")
            
            # Choose best approach
            if stacking_smape < best_smape:
                print(f"   🏆 Using Stacking (SMAPE: {stacking_smape:.4f}%)")
                final_score = stacking_smape
                self.use_stacking = True
            else:
                print(f"   🏆 Using Weighted Ensemble (SMAPE: {best_smape:.4f}%)")
                final_score = best_smape
                self.use_stacking = False
        else:
            self.weights = [1.0]
            final_score = model_scores[0]
            self.use_stacking = False
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("✅ ULTIMATE TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"   Individual model scores:")
        for (name, _), score in zip(self.models, model_scores):
            print(f"      {name}: {score:.4f}%")
        print(f"   🏆 Final Ensemble SMAPE: {final_score:.4f}%")
        print(f"   🎯 Target (<15%): {'✅ ACHIEVED' if final_score < 15 else '❌ NOT ACHIEVED'}")
        print(f"   ⏱️  Total training time: {total_time/60:.1f} minutes")
        print(f"{'='*80}\n")
        
        return {'val_smape': final_score, 'model_scores': model_scores}
    
    def predict(self, test_df):
        """Generate predictions using the best ensemble method"""
        if not self.models:
            raise ValueError("Model not trained! Call train() first.")
        
        print(f"🔮 Generating predictions for {len(test_df):,} samples...")
        
        X = self.feature_engineer.prepare_features(test_df, fit=False)
        X_scaled = self.scaler.transform(X)
        X_transformed = self.power_transformer.transform(X_scaled)
        
        # Get predictions from each model
        predictions = []
        for name, model in self.models:
            pred = model.predict(X_transformed)
            predictions.append(pred)
            print(f"   ✓ {name} predictions generated")
        
        # Choose prediction method
        if self.use_stacking and self.stacking_models:
            # Use stacking
            stacking_features = np.column_stack(predictions)
            final_pred = self.stacking_models[0][1].predict(stacking_features)
            print(f"   ✓ Stacking ensemble with {self.stacking_models[0][0]}")
        elif len(predictions) > 1:
            # Use weighted ensemble
            final_pred = sum(w * pred for w, pred in zip(self.weights, predictions))
            print(f"   ✓ Weighted ensemble: {[f'{w:.3f}' for w in self.weights]}")
        else:
            final_pred = predictions[0]
        
        # Inverse transform
        if self.use_log_transform:
            final_pred = np.expm1(final_pred)
        
        # Ensure positive prices
        final_pred = np.maximum(final_pred, 0.01)
        
        print(f"   ✓ Price range: ${final_pred.min():.2f} - ${final_pred.max():.2f}")
        
        return final_pred


def main():
    """Main execution pipeline"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    AMAZON ML CHALLENGE 2025                                  ║
    ║                    ULTIMATE SOLUTION                                         ║
    ║                    🎯 TARGET: SMAPE < 15%                                    ║
    ║                    Advanced Ensemble + Stacking + 1000+ Features            ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DATASET_FOLDER = 'dataset/'
    train_path = os.path.join(DATASET_FOLDER, 'train.csv')
    test_path = os.path.join(DATASET_FOLDER, 'test.csv')
    output_path = os.path.join(DATASET_FOLDER, 'test_out_ultimate.csv')
    
    # Check files exist
    if not os.path.exists(train_path):
        print(f"❌ ERROR: {train_path} not found!")
        return
    
    # Load data
    print("📂 Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"\n📊 Dataset Information:")
    print(f"   Train samples: {len(train_df):,}")
    print(f"   Test samples: {len(test_df):,}")
    
    print(f"\n💰 Price Statistics:")
    print(f"   Min: ${train_df['price'].min():.2f}")
    print(f"   Max: ${train_df['price'].max():.2f}")
    print(f"   Mean: ${train_df['price'].mean():.2f}")
    print(f"   Median: ${train_df['price'].median():.2f}")
    
    # Train model
    print(f"\n🔧 Using hyperparameter tuning: {HAS_OPTUNA}")
    predictor = UltimateEnsembleModel(use_tuning=HAS_OPTUNA)
    metrics = predictor.train(train_df, validation_split=0.15)
    
    # Generate predictions
    print(f"\n{'='*80}")
    print("🎯 GENERATING ULTIMATE TEST PREDICTIONS")
    print(f"{'='*80}")
    predictions = predictor.predict(test_df)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': np.round(predictions, 2)
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print("✅ ULTIMATE SUBMISSION FILE CREATED")
    print(f"{'='*80}")
    print(f"   📁 File: {output_path}")
    print(f"   📊 Predictions: {len(submission):,}")
    print(f"   💵 Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
    print(f"   💵 Mean price: ${submission['price'].mean():.2f}")
    print(f"\n{'='*80}")
    print(f"   🏆 Expected SMAPE on test set: ~{metrics['val_smape']:.2f}%")
    print(f"   🎯 Target (<15%): {'✅ ACHIEVED' if metrics['val_smape'] < 15 else '❌ NEEDS IMPROVEMENT'}")
    print(f"{'='*80}")
    print(f"\n   ✅ Ready to submit! Upload {output_path} to the challenge portal.")
    print(f"{'='*80}\n")
    
    # Show sample predictions
    print("📋 Sample predictions:")
    print(submission.head(10).to_string(index=False))
    
    return predictor, submission


if __name__ == "__main__":
    try:
        predictor, submission = main()
        print("\n🎉 ULTIMATE SUCCESS! Your advanced model is ready.")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 If you encounter errors, please share the error message!")
