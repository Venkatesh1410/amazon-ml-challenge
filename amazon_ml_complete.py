"""
Amazon ML Challenge 2025 - COMPLETE SOLUTION
Target: SMAPE < 15% (Competition Level Performance)
✓ Text + Image Features
✓ Advanced Ensemble + Stacking
✓ Comprehensive Hyperparameter Tuning
✓ Production-Ready Performance
"""

import os
import pandas as pd
import numpy as np
import re
import warnings
import time
import io
from datetime import datetime
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.neural_network import MLPRegressor

# Image processing
try:
    import requests
    from PIL import Image
    import cv2
    HAS_IMAGE = True
except ImportError:
    HAS_IMAGE = False
    print("⚠️  Image processing not available. Install: pip install pillow opencv-python requests")

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


class ImageFeatureExtractor:
    """Extract features from product images"""
    
    def __init__(self):
        self.image_cache = {}
        
    def extract_image_features(self, image_url, sample_id):
        """Extract comprehensive image features"""
        features = {}
        
        try:
            if image_url in self.image_cache:
                image = self.image_cache[image_url]
            else:
                # Download and process image
                response = requests.get(image_url, timeout=10)
                image = Image.open(io.BytesIO(response.content))
                image = image.convert('RGB')
                self.image_cache[image_url] = image
            
            # Basic image properties
            features['image_width'] = image.width
            features['image_height'] = image.height
            features['image_aspect_ratio'] = image.width / image.height if image.height > 0 else 0
            features['image_area'] = image.width * image.height
            features['image_is_square'] = int(abs(image.width - image.height) < 10)
            features['image_is_landscape'] = int(image.width > image.height)
            features['image_is_portrait'] = int(image.height > image.width)
            
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Color analysis
            features['image_mean_r'] = np.mean(img_array[:, :, 0])
            features['image_mean_g'] = np.mean(img_array[:, :, 1])
            features['image_mean_b'] = np.mean(img_array[:, :, 2])
            features['image_std_r'] = np.std(img_array[:, :, 0])
            features['image_std_g'] = np.std(img_array[:, :, 1])
            features['image_std_b'] = np.std(img_array[:, :, 2])
            
            # Brightness and contrast
            gray = np.mean(img_array, axis=2)
            features['image_brightness'] = np.mean(gray)
            features['image_contrast'] = np.std(gray)
            
            # Dominant colors (simplified)
            features['image_is_dark'] = int(features['image_brightness'] < 100)
            features['image_is_bright'] = int(features['image_brightness'] > 200)
            features['image_is_colorful'] = int(features['image_contrast'] > 50)
            
            # Edge detection (simplified)
            edges = cv2.Canny(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), 50, 150)
            features['image_edge_density'] = np.mean(edges > 0)
            
        except Exception as e:
            # Default values if image processing fails
            features = {
                'image_width': 0, 'image_height': 0, 'image_aspect_ratio': 0,
                'image_area': 0, 'image_is_square': 0, 'image_is_landscape': 0,
                'image_is_portrait': 0, 'image_mean_r': 0, 'image_mean_g': 0,
                'image_mean_b': 0, 'image_std_r': 0, 'image_std_g': 0, 'image_std_b': 0,
                'image_brightness': 0, 'image_contrast': 0, 'image_is_dark': 0,
                'image_is_bright': 0, 'image_is_colorful': 0, 'image_edge_density': 0
            }
        
        return features


class CompleteFeatureEngineer:
    """Complete feature engineering with text + image features"""
    
    def __init__(self):
        self.image_extractor = ImageFeatureExtractor() if HAS_IMAGE else None
        
        # Memory-optimized TF-IDF configurations
        self.tfidf_word = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.85,
            stop_words='english',
            sublinear_tf=True,
            lowercase=True,
            smooth_idf=True
        )
        
        self.tfidf_char = TfidfVectorizer(
            max_features=200,
            analyzer='char',
            ngram_range=(3, 5),
            min_df=10,
            max_df=0.9
        )
        
        self.tfidf_bigram = TfidfVectorizer(
            max_features=300,
            ngram_range=(2, 2),
            min_df=8,
            max_df=0.8,
            stop_words='english',
            sublinear_tf=True
        )
        
        # Additional TF-IDF for specific patterns
        self.tfidf_brand = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=10,
            max_df=0.75,
            lowercase=True
        )
        
        # Enhanced brand and category lists
        self.premium_brands = [
            'apple', 'samsung', 'sony', 'lg', 'dell', 'hp', 'lenovo', 'asus',
            'canon', 'nikon', 'panasonic', 'philips', 'bosch', 'nike', 'adidas',
            'puma', 'logitech', 'microsoft', 'intel', 'amd', 'nvidia', 'beats',
            'bose', 'jbl', 'sennheiser', 'gucci', 'prada', 'louis vuitton',
            'tiffany', 'rolex', 'omega', 'cartier', 'hermes', 'chanel', 'dior',
            'macbook', 'iphone', 'ipad', 'airpods', 'galaxy', 'playstation',
            'xbox', 'kindle', 'echo', 'alexa', 'nest', 'tesla', 'bmw', 'mercedes'
        ]
        
        # Price indicator brands (typically expensive)
        self.luxury_brands = [
            'gucci', 'prada', 'louis vuitton', 'tiffany', 'rolex', 'omega', 
            'cartier', 'hermes', 'chanel', 'dior', 'versace', 'armani',
            'burberry', 'ralph lauren', 'calvin klein', 'hugo boss'
        ]
        
        # Budget brands (typically cheaper)
        self.budget_brands = [
            'great value', 'store brand', 'generic', 'no name', 'house brand',
            'economy', 'budget', 'value', 'basic', 'simple'
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
            'sports': ['fitness', 'gym', 'exercise', 'yoga', 'sports', 'outdoor'],
            'toys': ['toy', 'game', 'puzzle', 'doll', 'lego', 'action figure'],
            'books': ['book', 'novel', 'textbook', 'magazine', 'kindle'],
            'food': ['food', 'snack', 'chocolate', 'candy', 'drink', 'coffee', 'tea'],
            'health': ['medicine', 'vitamin', 'supplement', 'health', 'medical']
        }
        
        self.fitted = False
    
    def extract_ipq(self, text):
        """Enhanced IPQ extraction"""
        text_lower = str(text).lower()
        
        patterns = [
            r'ipq[:\s]*(\d+)', r'pack of (\d+)', r'(\d+)\s*pack',
            r'quantity[:\s]*(\d+)', r'count[:\s]*(\d+)', r'set of (\d+)',
            r'(\d+)\s*piece', r'(\d+)\s*pcs', r'(\d+)\s*pc',
            r'(\d+)\s*count', r'(\d+)\s*units?', r'(\d+)\s*items?',
            r'bundle of (\d+)', r'lot of (\d+)', r'(\d+)\s*in\s*1',
            r'(\d+)\s*x\s*\d+', r'(\d+)\s*per\s*pack', r'(\d+)\s*per\s*box'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return min(int(match.group(1)), 1000)
        return 1
    
    def extract_price_indicators(self, text):
        """Extract price-related indicators"""
        text_lower = str(text).lower()
        features = {}
        
        # Storage, RAM, Screen, Weight, Volume, etc.
        storage_match = re.search(r'(\d+\.?\d*)\s*(gb|tb|mb)', text_lower)
        if storage_match:
            value = float(storage_match.group(1))
            if 'tb' in storage_match.group(2):
                value *= 1024
            elif 'mb' in storage_match.group(2):
                value /= 1024
            features['storage_gb'] = value
        else:
            features['storage_gb'] = 0
        
        ram_match = re.search(r'(\d+)\s*gb\s*ram', text_lower)
        features['ram_gb'] = int(ram_match.group(1)) if ram_match else 0
        
        screen_match = re.search(r'(\d+\.?\d*)\s*inch', text_lower)
        features['screen_inches'] = float(screen_match.group(1)) if screen_match else 0
        
        weight_match = re.search(r'(\d+\.?\d*)\s*(kg|g|lb|oz)', text_lower)
        if weight_match:
            value = float(weight_match.group(1))
            unit = weight_match.group(2)
            if unit in ['g']:
                value /= 1000
            elif unit in ['lb']:
                value *= 0.453592
            elif unit == 'oz':
                value *= 0.0283495
            features['weight_kg'] = value
        else:
            features['weight_kg'] = 0
        
        volume_match = re.search(r'(\d+\.?\d*)\s*(ml|l|fl oz)', text_lower)
        if volume_match:
            value = float(volume_match.group(1))
            if volume_match.group(2) == 'ml':
                value /= 1000
            elif 'fl oz' in volume_match.group(2):
                value *= 0.0295735
            features['volume_l'] = value
        else:
            features['volume_l'] = 0
        
        # Technology indicators
        features['is_4k'] = int('4k' in text_lower)
        features['is_hd'] = int('hd' in text_lower)
        features['has_wireless'] = int('wireless' in text_lower or 'wifi' in text_lower)
        features['has_bluetooth'] = int('bluetooth' in text_lower)
        
        return features
    
    def extract_quality_features(self, text):
        """Extract quality indicators"""
        text_lower = str(text).lower()
        features = {}
        
        # Brand analysis
        features['is_premium_brand'] = int(any(brand in text_lower for brand in self.premium_brands))
        features['is_luxury_brand'] = int(any(brand in text_lower for brand in self.luxury_brands))
        features['is_budget_brand'] = int(any(brand in text_lower for brand in self.budget_brands))
        features['brand_count'] = sum(1 for brand in self.premium_brands if brand in text_lower)
        
        # Quality terms
        premium_terms = ['premium', 'pro', 'plus', 'ultra', 'deluxe', 'luxury', 'professional', 'high-end', 'gourmet']
        features['premium_count'] = sum(term in text_lower for term in premium_terms)
        
        quality_terms = ['warranty', 'guarantee', 'certified', 'authentic', 'original', 'genuine', 'licensed']
        features['quality_count'] = sum(term in text_lower for term in quality_terms)
        
        tech_terms = ['smart', 'digital', 'wireless', 'bluetooth', 'touchscreen', 'ai', 'automated']
        features['tech_count'] = sum(term in text_lower for term in tech_terms)
        
        # Material indicators
        material_terms = ['organic', 'natural', 'handmade', 'artisan', 'crafted', 'leather', 'wood', 'metal', 'glass']
        features['material_count'] = sum(term in text_lower for term in material_terms)
        
        # Size indicators
        size_terms = ['large', 'small', 'mini', 'jumbo', 'family', 'bulk', 'economy', 'travel']
        features['size_count'] = sum(term in text_lower for term in size_terms)
        
        return features
    
    def extract_category_features(self, text):
        """Extract category features"""
        text_lower = str(text).lower()
        features = {}
        
        for category, keywords in self.categories.items():
            match_count = sum(1 for kw in keywords if kw in text_lower)
            features[f'cat_{category}'] = int(match_count > 0)
            features[f'cat_{category}_strength'] = match_count
        
        return features
    
    def extract_text_features(self, text):
        """Extract comprehensive text features"""
        text = str(text)
        text_lower = text.lower()
        words = text.split()
        
        features = {}
        
        # Basic statistics
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        features['unique_words'] = len(set(words))
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        
        # Character analysis
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['special_char_ratio'] = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text) if text else 0
        
        # IPQ
        ipq = self.extract_ipq(text)
        features['ipq'] = ipq
        features['ipq_log'] = np.log1p(ipq)
        
        # Numeric analysis
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            numbers = [float(n) for n in numbers if float(n) < 1000000]
            features['num_count'] = len(numbers)
            features['num_max'] = max(numbers) if numbers else 0
            features['num_mean'] = np.mean(numbers) if numbers else 0
        else:
            features.update({'num_count': 0, 'num_max': 0, 'num_mean': 0})
        
        return features
    
    def prepare_features(self, df, fit=False):
        """Prepare complete feature matrix with text + image features"""
        print(f"🔧 Engineering complete features for {len(df):,} samples...")
        
        # Process in chunks to reduce memory usage
        chunk_size = 5000
        all_text_features = []
        all_image_features = []
        
        for chunk_start in range(0, len(df), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df))
            chunk_df = df.iloc[chunk_start:chunk_end]
            
            chunk_text_features = []
            chunk_image_features = []
            
            for idx, row in chunk_df.iterrows():
                # Text features
                text_features = {}
                text_features.update(self.extract_text_features(row['catalog_content']))
                text_features.update(self.extract_price_indicators(row['catalog_content']))
                text_features.update(self.extract_quality_features(row['catalog_content']))
                text_features.update(self.extract_category_features(row['catalog_content']))
                chunk_text_features.append(text_features)
                
                # Image features
                if self.image_extractor and 'image_link' in row:
                    image_features = self.image_extractor.extract_image_features(
                        row['image_link'], row.get('sample_id', idx)
                    )
                else:
                    image_features = {f'image_{k}': 0 for k in [
                        'width', 'height', 'aspect_ratio', 'area', 'is_square', 'is_landscape',
                        'mean_r', 'mean_g', 'mean_b', 'brightness', 'contrast', 'is_dark'
                    ]}
                chunk_image_features.append(image_features)
            
            all_text_features.extend(chunk_text_features)
            all_image_features.extend(chunk_image_features)
            print(f"   Processed {chunk_end:,} samples...")
        
        text_features_df = pd.DataFrame(all_text_features)
        image_features_df = pd.DataFrame(all_image_features)
        
        # TF-IDF features
        if fit:
            tfidf_word_matrix = self.tfidf_word.fit_transform(df['catalog_content'].fillna(''))
            tfidf_char_matrix = self.tfidf_char.fit_transform(df['catalog_content'].fillna(''))
            tfidf_bigram_matrix = self.tfidf_bigram.fit_transform(df['catalog_content'].fillna(''))
            tfidf_brand_matrix = self.tfidf_brand.fit_transform(df['catalog_content'].fillna(''))
            self.fitted = True
        else:
            tfidf_word_matrix = self.tfidf_word.transform(df['catalog_content'].fillna(''))
            tfidf_char_matrix = self.tfidf_char.transform(df['catalog_content'].fillna(''))
            tfidf_bigram_matrix = self.tfidf_bigram.transform(df['catalog_content'].fillna(''))
            tfidf_brand_matrix = self.tfidf_brand.transform(df['catalog_content'].fillna(''))
        
        # Create TF-IDF DataFrames (keep sparse for memory efficiency)
        tfidf_word_df = pd.DataFrame.sparse.from_spmatrix(
            tfidf_word_matrix,
            columns=[f'tfidf_w_{i}' for i in range(tfidf_word_matrix.shape[1])]
        )
        
        tfidf_char_df = pd.DataFrame.sparse.from_spmatrix(
            tfidf_char_matrix,
            columns=[f'tfidf_c_{i}' for i in range(tfidf_char_matrix.shape[1])]
        )
        
        tfidf_bigram_df = pd.DataFrame.sparse.from_spmatrix(
            tfidf_bigram_matrix,
            columns=[f'tfidf_b_{i}' for i in range(tfidf_bigram_matrix.shape[1])]
        )
        
        tfidf_brand_df = pd.DataFrame.sparse.from_spmatrix(
            tfidf_brand_matrix,
            columns=[f'tfidf_br_{i}' for i in range(tfidf_brand_matrix.shape[1])]
        )
        
        # Convert to dense only for final combination
        tfidf_word_df = tfidf_word_df.sparse.to_dense()
        tfidf_char_df = tfidf_char_df.sparse.to_dense()
        tfidf_bigram_df = tfidf_bigram_df.sparse.to_dense()
        tfidf_brand_df = tfidf_brand_df.sparse.to_dense()
        
        # Combine all features
        features = pd.concat([
            text_features_df.reset_index(drop=True),
            image_features_df.reset_index(drop=True),
            tfidf_word_df.reset_index(drop=True),
            tfidf_char_df.reset_index(drop=True),
            tfidf_bigram_df.reset_index(drop=True),
            tfidf_brand_df.reset_index(drop=True)
        ], axis=1)
        
        print(f"✓ Generated {features.shape[1]} features (Text + Image + TF-IDF)")
        return features


class CompleteEnsembleModel:
    """Complete ensemble model with text + image features"""
    
    def __init__(self, use_tuning=True):
        self.feature_engineer = CompleteFeatureEngineer()
        self.scaler = RobustScaler()
        self.models = []
        self.stacking_models = []
        self.weights = []
        self.use_tuning = use_tuning and HAS_OPTUNA
        self.use_log_transform = True
        
    def optimize_model(self, model_name, X_train, y_train, X_val, y_val):
        """Optimize individual model"""
        if not self.use_tuning:
            if model_name == 'lgbm':
                return {
                    'n_estimators': 1500,
                    'learning_rate': 0.02,
                    'max_depth': 12,
                    'num_leaves': 100,
                    'min_child_samples': 10,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 1.0,
                    'reg_lambda': 2.0,
                    'min_split_gain': 0.0,
                    'random_state': 42,
                    'verbose': -1,
                    'force_col_wise': True
                }
            elif model_name == 'xgb':
                return {
                    'n_estimators': 1500,
                    'learning_rate': 0.02,
                    'max_depth': 12,
                    'min_child_weight': 3,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 1.0,
                    'reg_alpha': 1.0,
                    'reg_lambda': 3.0,
                    'max_delta_step': 1,
                    'scale_pos_weight': 1,
                    'random_state': 42,
                    'verbosity': 0
                }
            else:  # catboost
                return {
                    'iterations': 1500,
                    'learning_rate': 0.02,
                    'depth': 10,
                    'l2_leaf_reg': 5,
                    'border_count': 64,
                    'random_strength': 1.0,
                    'one_hot_max_size': 2,
                    'leaf_estimation_method': 'Newton',
                    'bootstrap_type': 'Bayesian',
                    'bagging_temperature': 1.0,
                    'random_seed': 42,
                    'verbose': False
                }
        
        def objective(trial):
            if model_name == 'lgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 1500, 3000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
                    'max_depth': trial.suggest_int('max_depth', 12, 20),
                    'num_leaves': trial.suggest_int('num_leaves', 100, 300),
                    'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
                    'subsample': trial.suggest_float('subsample', 0.4, 0.8),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 5.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
                    'random_state': 42,
                    'verbose': -1
                }
                model = LGBMRegressor(**params)
            elif model_name == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 1500, 3000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
                    'max_depth': trial.suggest_int('max_depth', 12, 20),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                    'subsample': trial.suggest_float('subsample', 0.4, 0.8),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
                    'gamma': trial.suggest_float('gamma', 1.0, 5.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 5.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                    'random_state': 42,
                    'verbosity': 0
                }
                model = XGBRegressor(**params)
            else:  # catboost
                params = {
                    'iterations': trial.suggest_int('iterations', 1500, 3000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
                    'depth': trial.suggest_int('depth', 10, 15),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5, 20),
                    'random_seed': 42,
                    'verbose': False
                }
                model = CatBoostRegressor(**params)
            
            # Cross-validation
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_vl = X_train[train_idx], X_train[val_idx]
                y_tr, y_vl = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_tr, y_tr)
                pred = model.predict(X_vl)
                
                y_vl_orig = np.expm1(y_vl)
                pred_orig = np.expm1(pred)
                smape = calculate_smape(y_vl_orig, pred_orig)
                scores.append(smape)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        return study.best_params
    
    def train_stacking_models(self, X_train, y_train, X_val, y_val):
        """Train stacking meta-learners"""
        stacking_models = [
            ('Ridge', Ridge(alpha=0.001)),
            ('ElasticNet', ElasticNet(alpha=0.0001, l1_ratio=0.05, max_iter=1000)),
            ('Huber', HuberRegressor(epsilon=1.2, max_iter=500, alpha=0.001)),
            ('MLP', MLPRegressor(hidden_layer_sizes=(500, 200), max_iter=2000, 
                               learning_rate_init=0.001, random_state=42)),
            ('RandomForest', RandomForestRegressor(n_estimators=500, max_depth=15, 
                                                 min_samples_split=2, random_state=42)),
            ('ExtraTrees', ExtraTreesRegressor(n_estimators=500, max_depth=15, 
                                             min_samples_split=2, random_state=42))
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
        """Train complete ensemble model"""
        print("\n" + "="*80)
        print("🚀 AMAZON ML CHALLENGE - COMPLETE SOLUTION TRAINING")
        print("🎯 TARGET: SMAPE < 15%")
        print("📊 FEATURES: Text + Image + Advanced Engineering")
        print("="*80)
        
        start_time = time.time()
        
        # Feature engineering
        X = self.feature_engineer.prepare_features(train_df, fit=True)
        y = train_df['price'].values
        
        # Log transform
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
        
        print(f"\n📊 Complete Data Split:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        print(f"   Total features: {X_train.shape[1]:,}")
        
        # Train models
        model_configs = []
        
        if HAS_LGBM:
            print(f"\n🔧 {'Optimizing' if self.use_tuning else 'Configuring'} LightGBM...")
            lgbm_params = self.optimize_model('lgbm', X_train_scaled, y_train, X_val_scaled, y_val)
            model_configs.append(('LightGBM', LGBMRegressor(**lgbm_params)))
        
        if HAS_XGB:
            print(f"\n🔧 {'Optimizing' if self.use_tuning else 'Configuring'} XGBoost...")
            xgb_params = self.optimize_model('xgb', X_train_scaled, y_train, X_val_scaled, y_val)
            model_configs.append(('XGBoost', XGBRegressor(**xgb_params)))
        
        if HAS_CATBOOST:
            print(f"\n🔧 {'Optimizing' if self.use_tuning else 'Configuring'} CatBoost...")
            catboost_params = self.optimize_model('catboost', X_train_scaled, y_train, X_val_scaled, y_val)
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
        print("🔧 Complete Ensemble Optimization...")
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
        print("✅ COMPLETE TRAINING FINISHED")
        print(f"{'='*80}")
        print(f"   Individual model scores:")
        for (name, _), score in zip(self.models, model_scores):
            print(f"      {name}: {score:.4f}%")
        print(f"   🏆 Final Ensemble SMAPE: {final_score:.4f}%")
        print(f"   🎯 Target (<15%): {'✅ ACHIEVED' if final_score < 15 else '❌ NEEDS IMPROVEMENT'}")
        print(f"   ⏱️  Total training time: {total_time/60:.1f} minutes")
        print(f"{'='*80}\n")
        
        return {'val_smape': final_score, 'model_scores': model_scores}
    
    def predict(self, test_df):
        """Generate predictions"""
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
    ║                    COMPLETE SOLUTION                                          ║
    ║                    🎯 TARGET: SMAPE < 15%                                     ║
    ║                    📊 Text + Image + Advanced Features                        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DATASET_FOLDER = 'dataset/'
    train_path = os.path.join(DATASET_FOLDER, 'train.csv')
    test_path = os.path.join(DATASET_FOLDER, 'test.csv')
    output_path = os.path.join(DATASET_FOLDER, 'test_out_complete.csv')
    
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
    print(f"📸 Image processing available: {HAS_IMAGE}")
    predictor = CompleteEnsembleModel(use_tuning=HAS_OPTUNA)
    metrics = predictor.train(train_df, validation_split=0.15)
    
    # Generate predictions
    print(f"\n{'='*80}")
    print("🎯 GENERATING COMPLETE TEST PREDICTIONS")
    print(f"{'='*80}")
    predictions = predictor.predict(test_df)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': np.round(predictions, 2)
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE SUBMISSION FILE CREATED")
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
        print("\n🎉 COMPLETE SUCCESS! Your advanced model with text + image features is ready.")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 If you encounter errors, please share the error message!")
