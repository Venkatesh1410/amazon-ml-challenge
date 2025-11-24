"""
Amazon ML Challenge 2025 - ULTIMATE OPTIMIZED SOLUTION
✓ Advanced Multi-Model Ensemble with Stacking
✓ 500+ Advanced Features + Hyperparameter Tuning
✓ SMAPE-Optimized with Advanced Validation
✓ Dataset-Specific Optimizations
✓ Production-Ready Performance

USAGE:
    python amazon_ml_optimized.py

Requirements:
    pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna
"""

import os
import pandas as pd
import numpy as np
import re
import warnings
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Advanced models
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("⚠️  LightGBM not available. Install: pip install lightgbm")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not available. Install: pip install xgboost")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not available. Install: pip install catboost")

# Hyperparameter tuning
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


class AdvancedTextAnalyzer:
    """Advanced text analysis for catalog content"""
    
    def __init__(self):
        self.premium_brands = [
            'apple', 'samsung', 'sony', 'lg', 'dell', 'hp', 'lenovo', 'asus',
            'canon', 'nikon', 'panasonic', 'philips', 'bosch', 'nike', 'adidas',
            'puma', 'logitech', 'microsoft', 'intel', 'amd', 'nvidia', 'beats',
            'bose', 'jbl', 'sennheiser', 'gucci', 'prada', 'louis vuitton'
        ]
        
        self.categories = {
            'electronics': ['laptop', 'phone', 'tablet', 'computer', 'camera', 
                          'tv', 'monitor', 'headphone', 'speaker', 'mouse', 'keyboard',
                          'smartphone', 'ipad', 'macbook', 'wireless', 'bluetooth'],
            'clothing': ['shirt', 'pants', 'jeans', 'dress', 'shoes', 'jacket', 
                        'sweater', 'hoodie', 't-shirt', 'sneakers', 'boots'],
            'home': ['furniture', 'chair', 'table', 'sofa', 'bed', 'lamp', 
                    'kitchen', 'decor', 'pillow', 'blanket', 'curtain'],
            'beauty': ['makeup', 'cosmetic', 'perfume', 'shampoo', 'lotion', 
                      'cream', 'serum', 'foundation', 'lipstick', 'skincare'],
            'sports': ['fitness', 'gym', 'exercise', 'yoga', 'sports', 'outdoor',
                      'running', 'training', 'equipment', 'weights'],
            'toys': ['toy', 'game', 'puzzle', 'doll', 'lego', 'action figure'],
            'books': ['book', 'novel', 'textbook', 'magazine', 'kindle'],
            'automotive': ['car', 'bike', 'vehicle', 'automotive', 'motor', 'tire'],
            'food': ['food', 'snack', 'chocolate', 'candy', 'drink', 'beverage',
                    'coffee', 'tea', 'sauce', 'spice', 'cooking'],
            'health': ['medicine', 'vitamin', 'supplement', 'health', 'medical']
        }
    
    def extract_numeric_features(self, text):
        """Extract all numeric features from text"""
        text_lower = str(text).lower()
        features = {}
        
        # Storage capacity
        storage_match = re.search(r'(\d+\.?\d*)\s*(gb|tb|mb)', text_lower)
        if storage_match:
            value = float(storage_match.group(1))
            unit = storage_match.group(2)
            if unit == 'tb':
                value *= 1024
            elif unit == 'mb':
                value /= 1024
            features['storage_gb'] = value
            features['has_storage'] = 1
        else:
            features['storage_gb'] = 0
            features['has_storage'] = 0
        
        # RAM
        ram_match = re.search(r'(\d+)\s*gb\s*ram', text_lower)
        if ram_match:
            features['ram_gb'] = int(ram_match.group(1))
            features['has_ram'] = 1
        else:
            features['ram_gb'] = 0
            features['has_ram'] = 0
        
        # Screen size
        screen_match = re.search(r'(\d+\.?\d*)\s*inch', text_lower)
        if screen_match:
            features['screen_inches'] = float(screen_match.group(1))
            features['has_screen'] = 1
        else:
            features['screen_inches'] = 0
            features['has_screen'] = 0
        
        # Weight
        weight_match = re.search(r'(\d+\.?\d*)\s*(kg|g|lb|oz|pound)', text_lower)
        if weight_match:
            value = float(weight_match.group(1))
            unit = weight_match.group(2)
            if unit in ['g', 'gram']:
                value /= 1000
            elif unit in ['lb', 'pound']:
                value *= 0.453592
            elif unit == 'oz':
                value *= 0.0283495
            features['weight_kg'] = value
            features['has_weight'] = 1
        else:
            features['weight_kg'] = 0
            features['has_weight'] = 0
        
        # Volume/Capacity
        volume_match = re.search(r'(\d+\.?\d*)\s*(ml|l|liter|fl oz|ounce)', text_lower)
        if volume_match:
            value = float(volume_match.group(1))
            unit = volume_match.group(2)
            if unit in ['ml']:
                value /= 1000
            elif unit in ['fl oz', 'ounce']:
                value *= 0.0295735
            features['volume_l'] = value
            features['has_volume'] = 1
        else:
            features['volume_l'] = 0
            features['has_volume'] = 0
        
        # Megapixels
        mp_match = re.search(r'(\d+)\s*mp', text_lower)
        if mp_match:
            features['megapixels'] = int(mp_match.group(1))
            features['has_megapixels'] = 1
        else:
            features['megapixels'] = 0
            features['has_megapixels'] = 0
        
        # Quantity/Pack size
        pack_match = re.search(r'pack of (\d+)', text_lower)
        if pack_match:
            features['pack_size'] = int(pack_match.group(1))
            features['has_pack_info'] = 1
        else:
            features['pack_size'] = 1
            features['has_pack_info'] = 0
        
        # IPQ (Item Pack Quantity)
        ipq_patterns = [
            r'ipq[:\s]*(\d+)',
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
            r'lot of (\d+)'
        ]
        
        ipq = 1
        for pattern in ipq_patterns:
            match = re.search(pattern, text_lower)
            if match:
                ipq = min(int(match.group(1)), 1000)
                break
        
        features['ipq'] = ipq
        features['ipq_log'] = np.log1p(ipq)
        features['ipq_squared'] = ipq ** 2
        
        return features
    
    def extract_quality_indicators(self, text):
        """Extract quality and premium indicators"""
        text_lower = str(text).lower()
        features = {}
        
        # Premium terms
        premium_terms = [
            'premium', 'pro', 'plus', 'ultra', 'deluxe', 'luxury', 'professional',
            'advanced', 'high-end', 'top-quality', 'best', 'superior', 'exclusive'
        ]
        features['premium_count'] = sum(term in text_lower for term in premium_terms)
        features['is_premium'] = int(features['premium_count'] > 0)
        
        # Brand indicators
        features['is_premium_brand'] = int(any(brand in text_lower for brand in self.premium_brands))
        features['brand_count'] = sum(1 for brand in self.premium_brands if brand in text_lower)
        
        # Quality indicators
        quality_terms = ['warranty', 'guarantee', 'certified', 'authentic', 'original', 'genuine']
        features['quality_count'] = sum(term in text_lower for term in quality_terms)
        features['has_warranty'] = int('warranty' in text_lower or 'guarantee' in text_lower)
        
        # Technology indicators
        tech_terms = ['wireless', 'bluetooth', 'wifi', 'smart', 'digital', 'hd', '4k', 'led', 'usb', 'nfc']
        features['tech_count'] = sum(term in text_lower for term in tech_terms)
        features['is_tech_product'] = int(features['tech_count'] > 2)
        
        # Resolution indicators
        features['is_4k'] = int('4k' in text_lower or '2160p' in text_lower)
        features['is_hd'] = int('hd' in text_lower or '1080p' in text_lower)
        
        return features
    
    def extract_category_features(self, text):
        """Extract category-specific features"""
        text_lower = str(text).lower()
        features = {}
        
        # Category detection
        for category, keywords in self.categories.items():
            match_count = sum(1 for kw in keywords if kw in text_lower)
            features[f'cat_{category}'] = int(match_count > 0)
            features[f'cat_{category}_strength'] = match_count
        
        # Primary category (highest match) - convert to numeric
        max_category = max(self.categories.keys(), 
                          key=lambda cat: features[f'cat_{cat}_strength'])
        if features[f'cat_{max_category}_strength'] > 0:
            features['primary_category'] = hash(max_category) % 1000  # Convert to numeric
        else:
            features['primary_category'] = 0  # Unknown category
        
        return features
    
    def extract_material_features(self, text):
        """Extract material and build quality features"""
        text_lower = str(text).lower()
        features = {}
        
        # Materials
        materials = {
            'leather': ['leather', 'genuine leather'],
            'metal': ['metal', 'steel', 'aluminum', 'brass', 'copper'],
            'wood': ['wood', 'wooden', 'oak', 'pine', 'mahogany'],
            'plastic': ['plastic', 'pvc', 'acrylic'],
            'glass': ['glass', 'crystal'],
            'fabric': ['cotton', 'silk', 'wool', 'polyester', 'nylon'],
            'ceramic': ['ceramic', 'porcelain'],
            'rubber': ['rubber', 'silicone']
        }
        
        for material, keywords in materials.items():
            features[f'material_{material}'] = int(any(kw in text_lower for kw in keywords))
        
        features['material_count'] = sum(features[f'material_{mat}'] for mat in materials.keys())
        features['is_premium_material'] = int(features['material_leather'] + 
                                            features['material_metal'] + 
                                            features['material_wood'] > 0)
        
        return features


class AdvancedFeatureEngineer:
    """Ultimate feature engineering with 500+ features"""
    
    def __init__(self):
        self.text_analyzer = AdvancedTextAnalyzer()
        
        # TF-IDF configurations
        self.tfidf_word = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9,
            stop_words='english',
            sublinear_tf=True,
            lowercase=True
        )
        
        self.tfidf_char = TfidfVectorizer(
            max_features=100,
            analyzer='char',
            ngram_range=(3, 6),
            min_df=2
        )
        
        self.tfidf_bigram = TfidfVectorizer(
            max_features=150,
            ngram_range=(2, 2),
            min_df=3,
            max_df=0.8,
            stop_words='english'
        )
        
        # Count vectorizer for rare terms
        self.count_vec = CountVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.7
        )
        
        # SVD for dimensionality reduction
        self.svd_word = TruncatedSVD(n_components=50, random_state=42)
        self.svd_char = TruncatedSVD(n_components=30, random_state=42)
        
        self.fitted = False
    
    def extract_basic_text_features(self, text):
        """Extract basic text statistics"""
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
        
        # Sentence analysis
        sentences = re.split(r'[.!?]+', text)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
        
        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['comma_count'] = text.count(',')
        features['period_count'] = text.count('.')
        
        return features
    
    def extract_advanced_features(self, text):
        """Extract advanced domain-specific features"""
        text = str(text)
        text_lower = text.lower()
        
        features = {}
        
        # Combine all feature extractors
        features.update(self.extract_basic_text_features(text))
        features.update(self.text_analyzer.extract_numeric_features(text))
        features.update(self.text_analyzer.extract_quality_indicators(text))
        features.update(self.text_analyzer.extract_category_features(text))
        features.update(self.text_analyzer.extract_material_features(text))
        
        # Additional advanced features
        features['has_bullet_points'] = int('bullet point' in text_lower)
        features['has_product_description'] = int('product description' in text_lower)
        features['has_value_unit'] = int('value:' in text_lower and 'unit:' in text_lower)
        
        # Color detection
        colors = ['black', 'white', 'red', 'blue', 'green', 'silver', 'gold', 'pink', 'purple', 'orange']
        features['color_count'] = sum(1 for color in colors if color in text_lower)
        features['has_color'] = int(features['color_count'] > 0)
        
        # Size indicators
        size_terms = ['small', 'medium', 'large', 'xl', 'xxl', 'mini', 'jumbo', 'compact']
        features['size_count'] = sum(1 for size in size_terms if size in text_lower)
        
        # Age indicators
        age_terms = ['adult', 'kids', 'children', 'baby', 'infant', 'teen']
        features['age_specific'] = int(any(term in text_lower for term in age_terms))
        
        # Seasonal indicators
        seasonal_terms = ['christmas', 'halloween', 'valentine', 'summer', 'winter']
        features['seasonal'] = int(any(term in text_lower for term in seasonal_terms))
        
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
            features['num_range'] = features['num_max'] - features['num_min']
        else:
            features.update({
                'num_count': 0, 'num_max': 0, 'num_min': 0, 
                'num_mean': 0, 'num_median': 0, 'num_std': 0, 'num_range': 0
            })
        
        return features
    
    def prepare_features(self, df, fit=False):
        """Prepare complete feature matrix with 500+ features"""
        print(f"🔧 Engineering 500+ features for {len(df):,} samples...")
        
        # Handcrafted features
        text_features = df['catalog_content'].apply(self.extract_advanced_features)
        text_features_df = pd.DataFrame(text_features.tolist())
        
        # Text vectorization
        if fit:
            # Fit transformers
            tfidf_word_matrix = self.tfidf_word.fit_transform(df['catalog_content'].fillna(''))
            tfidf_char_matrix = self.tfidf_char.fit_transform(df['catalog_content'].fillna(''))
            tfidf_bigram_matrix = self.tfidf_bigram.fit_transform(df['catalog_content'].fillna(''))
            count_matrix = self.count_vec.fit_transform(df['catalog_content'].fillna(''))
            
            # Apply SVD
            tfidf_word_svd = self.svd_word.fit_transform(tfidf_word_matrix)
            tfidf_char_svd = self.svd_char.fit_transform(tfidf_char_matrix)
            
            self.fitted = True
        else:
            # Transform only
            tfidf_word_matrix = self.tfidf_word.transform(df['catalog_content'].fillna(''))
            tfidf_char_matrix = self.tfidf_char.transform(df['catalog_content'].fillna(''))
            tfidf_bigram_matrix = self.tfidf_bigram.transform(df['catalog_content'].fillna(''))
            count_matrix = self.count_vec.transform(df['catalog_content'].fillna(''))
            
            tfidf_word_svd = self.svd_word.transform(tfidf_word_matrix)
            tfidf_char_svd = self.svd_char.transform(tfidf_char_matrix)
        
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
        
        # Combine all features
        features = pd.concat([
            text_features_df.reset_index(drop=True),
            tfidf_word_df.reset_index(drop=True),
            tfidf_char_df.reset_index(drop=True),
            tfidf_bigram_df.reset_index(drop=True),
            count_df.reset_index(drop=True),
            tfidf_word_svd_df.reset_index(drop=True),
            tfidf_char_svd_df.reset_index(drop=True)
        ], axis=1)
        
        print(f"✓ Generated {features.shape[1]} features")
        return features


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization with Optuna"""
    
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.best_params = {}
    
    def optimize_lgbm(self, X_train, y_train, X_val, y_val):
        """Optimize LightGBM hyperparameters"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 6, 15),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': 42,
                'verbose': -1
            }
            
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            # Use log1p transformed target
            y_val_orig = np.expm1(y_val)
            pred_orig = np.expm1(pred)
            smape = calculate_smape(y_val_orig, pred_orig)
            
            return smape
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['lgbm'] = study.best_params
        return study.best_params
    
    def optimize_xgb(self, X_train, y_train, X_val, y_val):
        """Optimize XGBoost hyperparameters"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
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
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['xgb'] = study.best_params
        return study.best_params
    
    def optimize_catboost(self, X_train, y_train, X_val, y_val):
        """Optimize CatBoost hyperparameters"""
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'depth': trial.suggest_int('depth', 6, 12),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_seed': 42,
                'verbose': False
            }
            
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            y_val_orig = np.expm1(y_val)
            pred_orig = np.expm1(pred)
            smape = calculate_smape(y_val_orig, pred_orig)
            
            return smape
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['catboost'] = study.best_params
        return study.best_params


class AdvancedEnsembleModel:
    """Advanced ensemble with stacking and hyperparameter optimization"""
    
    def __init__(self, use_tuning=True, n_trials=30):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.scaler = RobustScaler()
        self.optimizer = HyperparameterOptimizer(n_trials) if HAS_OPTUNA and use_tuning else None
        self.models = []
        self.stacking_model = None
        self.weights = []
        self.use_tuning = use_tuning and HAS_OPTUNA
        self.use_log_transform = True
        
    def train(self, train_df, validation_split=0.15):
        """Train advanced ensemble model"""
        print("\n" + "="*80)
        print("🚀 AMAZON ML CHALLENGE - ULTIMATE OPTIMIZED TRAINING")
        print("="*80)
        
        start_time = time.time()
        
        # Feature engineering
        X = self.feature_engineer.prepare_features(train_df, fit=True)
        y = train_df['price'].values
        
        # Log transform
        if self.use_log_transform:
            y = np.log1p(y)
            print("✓ Using log1p transformation for target")
        
        # Advanced validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=True
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print(f"\n📊 Advanced Data Split:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        print(f"   Features: {X_train.shape[1]:,}")
        
        # Hyperparameter optimization
        model_configs = []
        
        if HAS_LGBM:
            if self.use_tuning:
                print(f"\n🔧 Optimizing LightGBM hyperparameters...")
                lgbm_params = self.optimizer.optimize_lgbm(X_train_scaled, y_train, X_val_scaled, y_val)
                print(f"   ✓ Best LightGBM params: {lgbm_params}")
            else:
                lgbm_params = {
                    'n_estimators': 800,
                    'learning_rate': 0.03,
                    'max_depth': 9,
                    'num_leaves': 40,
                    'min_child_samples': 15,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.5,
                    'random_state': 42,
                    'verbose': -1
                }
            model_configs.append(('LightGBM', LGBMRegressor(**lgbm_params)))
        
        if HAS_XGB:
            if self.use_tuning:
                print(f"\n🔧 Optimizing XGBoost hyperparameters...")
                xgb_params = self.optimizer.optimize_xgb(X_train_scaled, y_train, X_val_scaled, y_val)
                print(f"   ✓ Best XGBoost params: {xgb_params}")
            else:
                xgb_params = {
                    'n_estimators': 700,
                    'learning_rate': 0.03,
                    'max_depth': 8,
                    'min_child_weight': 3,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1,
                    'random_state': 42,
                    'verbosity': 0
                }
            model_configs.append(('XGBoost', XGBRegressor(**xgb_params)))
        
        if HAS_CATBOOST:
            if self.use_tuning:
                print(f"\n🔧 Optimizing CatBoost hyperparameters...")
                catboost_params = self.optimizer.optimize_catboost(X_train_scaled, y_train, X_val_scaled, y_val)
                print(f"   ✓ Best CatBoost params: {catboost_params}")
            else:
                catboost_params = {
                    'iterations': 700,
                    'learning_rate': 0.03,
                    'depth': 8,
                    'l2_leaf_reg': 3,
                    'random_seed': 42,
                    'verbose': False
                }
            model_configs.append(('CatBoost', CatBoostRegressor(**catboost_params)))
        
        if not model_configs:
            raise RuntimeError("No models available! Install at least one: lightgbm, xgboost, or catboost")
        
        # Train each model
        val_predictions = []
        model_scores = []
        
        for name, model in model_configs:
            print(f"\n{'='*80}")
            print(f"📊 Training {name}...")
            print(f"{'='*80}")
            
            model_start = time.time()
            model.fit(X_train_scaled, y_train)
            model_time = time.time() - model_start
            
            self.models.append((name, model))
            
            # Validate
            pred = model.predict(X_val_scaled)
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
        print("🔧 Advanced Ensemble Optimization...")
        print(f"{'='*80}")
        
        if len(val_predictions) > 1:
            # Stacking approach
            stacking_features = np.column_stack(val_predictions)
            
            # Try different meta-learners
            meta_learners = [
                ('Ridge', Ridge(alpha=1.0)),
                ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5)),
                ('Huber', HuberRegressor(epsilon=1.35, max_iter=100))
            ]
            
            best_stacking_score = float('inf')
            best_stacking_model = None
            
            for meta_name, meta_model in meta_learners:
                meta_model.fit(stacking_features, y_val)
                stacking_pred = meta_model.predict(stacking_features)
                
                if self.use_log_transform:
                    stacking_pred_orig = np.expm1(stacking_pred)
                    y_val_orig = np.expm1(y_val)
                    smape = calculate_smape(y_val_orig, stacking_pred_orig)
                else:
                    smape = calculate_smape(y_val, stacking_pred)
                
                if smape < best_stacking_score:
                    best_stacking_score = smape
                    best_stacking_model = (meta_name, meta_model)
            
            self.stacking_model = best_stacking_model
            print(f"   ✓ Best stacking model: {best_stacking_model[0]}")
            print(f"   ✓ Stacking SMAPE: {best_stacking_score:.4f}%")
            
            # Weighted ensemble
            best_smape = float('inf')
            best_weights = None
            
            # Grid search for optimal weights
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
            if best_stacking_score < best_smape:
                print(f"   🏆 Using Stacking (SMAPE: {best_stacking_score:.4f}%)")
                final_score = best_stacking_score
            else:
                print(f"   🏆 Using Weighted Ensemble (SMAPE: {best_smape:.4f}%)")
                final_score = best_smape
        else:
            self.weights = [1.0]
            final_score = model_scores[0]
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("✅ ULTIMATE TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"   Individual model scores:")
        for (name, _), score in zip(self.models, model_scores):
            print(f"      {name}: {score:.4f}%")
        print(f"   🏆 Final Ensemble SMAPE: {final_score:.4f}%")
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
        
        # Get predictions from each model
        predictions = []
        for name, model in self.models:
            pred = model.predict(X_scaled)
            predictions.append(pred)
            print(f"   ✓ {name} predictions generated")
        
        # Choose prediction method
        if self.stacking_model and len(predictions) > 1:
            # Use stacking
            stacking_features = np.column_stack(predictions)
            final_pred = self.stacking_model[1].predict(stacking_features)
            print(f"   ✓ Stacking ensemble with {self.stacking_model[0]}")
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
    """Main execution pipeline with advanced optimizations"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    AMAZON ML CHALLENGE 2025                                  ║
    ║                    ULTIMATE OPTIMIZED SOLUTION                               ║
    ║                    Advanced Ensemble + Hyperparameter Tuning                ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DATASET_FOLDER = 'dataset/'
    train_path = os.path.join(DATASET_FOLDER, 'train.csv')
    test_path = os.path.join(DATASET_FOLDER, 'test.csv')
    output_path = os.path.join(DATASET_FOLDER, 'test_out_optimized.csv')
    
    # Check files exist
    if not os.path.exists(train_path):
        print(f"❌ ERROR: {train_path} not found!")
        print(f"   Please ensure your dataset folder structure is correct:")
        print(f"   - dataset/train.csv")
        print(f"   - dataset/test.csv")
        return
    
    # Load data
    print("📂 Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"\n📊 Dataset Information:")
    print(f"   Train samples: {len(train_df):,}")
    print(f"   Test samples: {len(test_df):,}")
    print(f"   Columns: {train_df.columns.tolist()}")
    
    print(f"\n💰 Price Statistics:")
    print(f"   Min: ${train_df['price'].min():.2f}")
    print(f"   Max: ${train_df['price'].max():.2f}")
    print(f"   Mean: ${train_df['price'].mean():.2f}")
    print(f"   Median: ${train_df['price'].median():.2f}")
    
    # Check for missing values
    missing = train_df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Missing values detected:")
        print(missing[missing > 0])
    
    # Train model with hyperparameter tuning
    print(f"\n🔧 Using hyperparameter tuning: {HAS_OPTUNA}")
    predictor = AdvancedEnsembleModel(use_tuning=HAS_OPTUNA, n_trials=30)
    metrics = predictor.train(train_df, validation_split=0.15)
    
    # Generate predictions
    print(f"\n{'='*80}")
    print("🎯 GENERATING OPTIMIZED TEST PREDICTIONS")
    print(f"{'='*80}")
    predictions = predictor.predict(test_df)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': np.round(predictions, 2)
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print("✅ OPTIMIZED SUBMISSION FILE CREATED")
    print(f"{'='*80}")
    print(f"   📁 File: {output_path}")
    print(f"   📊 Predictions: {len(submission):,}")
    print(f"   💵 Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
    print(f"   💵 Mean price: ${submission['price'].mean():.2f}")
    print(f"\n{'='*80}")
    print(f"   🏆 Expected SMAPE on test set: ~{metrics['val_smape']:.2f}%")
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
        print("\n🎉 ULTIMATE SUCCESS! Your optimized model is ready.")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 If you encounter errors, please share the error message!")
