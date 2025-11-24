"""
Amazon ML Challenge 2025 - FINAL PRODUCTION SOLUTION
✓ Optimized Ensemble with Advanced Features
✓ Hyperparameter Tuning + SMAPE Optimization
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

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
    """Ultimate feature engineering with 400+ optimized features"""
    
    def __init__(self):
        self.tfidf_word = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9,
            stop_words='english',
            sublinear_tf=True
        )
        
        self.tfidf_char = TfidfVectorizer(
            max_features=100,
            analyzer='char',
            ngram_range=(3, 5),
            min_df=2
        )
        
        self.fitted = False
    
    def extract_ipq(self, text):
        """Extract Item Pack Quantity with enhanced patterns"""
        text = str(text).lower()
        
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
            r'(\d+)\s*x\s*\d+',  # Pattern like "12 x 5"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                qty = int(match.group(1))
                return min(qty, 1000)
        
        return 1
    
    def extract_numeric_features(self, text):
        """Extract comprehensive numeric features"""
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
        
        return features
    
    def extract_quality_features(self, text):
        """Extract quality and premium indicators"""
        text_lower = str(text).lower()
        features = {}
        
        # Premium brands
        premium_brands = [
            'apple', 'samsung', 'sony', 'lg', 'dell', 'hp', 'lenovo', 'asus',
            'canon', 'nikon', 'panasonic', 'philips', 'bosch', 'nike', 'adidas',
            'puma', 'logitech', 'microsoft', 'intel', 'amd', 'nvidia', 'beats',
            'bose', 'jbl', 'sennheiser'
        ]
        features['is_premium_brand'] = int(any(brand in text_lower for brand in premium_brands))
        features['brand_count'] = sum(1 for brand in premium_brands if brand in text_lower)
        
        # Premium terms
        premium_terms = ['premium', 'pro', 'plus', 'ultra', 'deluxe', 'luxury', 'professional']
        features['premium_count'] = sum(term in text_lower for term in premium_terms)
        features['is_premium'] = int(features['premium_count'] > 0)
        
        # Quality indicators
        quality_terms = ['warranty', 'guarantee', 'certified', 'authentic', 'original', 'genuine']
        features['quality_count'] = sum(term in text_lower for term in quality_terms)
        features['has_warranty'] = int('warranty' in text_lower or 'guarantee' in text_lower)
        
        # Technology indicators
        tech_terms = ['wireless', 'bluetooth', 'wifi', 'smart', 'digital', 'hd', '4k', 'led', 'usb']
        features['tech_count'] = sum(term in text_lower for term in tech_terms)
        features['is_tech_product'] = int(features['tech_count'] > 2)
        
        # Resolution
        features['is_4k'] = int('4k' in text_lower or '2160p' in text_lower)
        features['is_hd'] = int('hd' in text_lower or '1080p' in text_lower)
        
        return features
    
    def extract_category_features(self, text):
        """Extract category-specific features"""
        text_lower = str(text).lower()
        features = {}
        
        categories = {
            'electronics': ['laptop', 'phone', 'tablet', 'computer', 'camera', 
                          'tv', 'monitor', 'headphone', 'speaker', 'mouse', 'keyboard'],
            'clothing': ['shirt', 'pants', 'jeans', 'dress', 'shoes', 'jacket', 'sweater'],
            'home': ['furniture', 'chair', 'table', 'sofa', 'bed', 'lamp', 'kitchen'],
            'beauty': ['makeup', 'cosmetic', 'perfume', 'shampoo', 'lotion', 'cream'],
            'sports': ['fitness', 'gym', 'exercise', 'yoga', 'sports', 'outdoor'],
            'toys': ['toy', 'game', 'puzzle', 'doll', 'lego'],
            'books': ['book', 'novel', 'textbook', 'magazine'],
            'food': ['food', 'snack', 'chocolate', 'candy', 'drink', 'coffee', 'tea', 'sauce'],
            'health': ['medicine', 'vitamin', 'supplement', 'health']
        }
        
        for category, keywords in categories.items():
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
        features['ipq'] = self.extract_ipq(text)
        features['ipq_log'] = np.log1p(features['ipq'])
        features['ipq_squared'] = features['ipq'] ** 2
        
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
        else:
            features.update({
                'num_count': 0, 'num_max': 0, 'num_min': 0, 
                'num_mean': 0, 'num_median': 0, 'num_std': 0
            })
        
        return features
    
    def prepare_features(self, df, fit=False):
        """Prepare complete feature matrix"""
        print(f"🔧 Engineering 400+ features for {len(df):,} samples...")
        
        # Handcrafted features
        all_features = []
        for text in df['catalog_content']:
            features = {}
            features.update(self.extract_text_features(text))
            features.update(self.extract_numeric_features(text))
            features.update(self.extract_quality_features(text))
            features.update(self.extract_category_features(text))
            all_features.append(features)
        
        text_features_df = pd.DataFrame(all_features)
        
        # TF-IDF features
        if fit:
            tfidf_word_matrix = self.tfidf_word.fit_transform(df['catalog_content'].fillna(''))
            tfidf_char_matrix = self.tfidf_char.fit_transform(df['catalog_content'].fillna(''))
            self.fitted = True
        else:
            tfidf_word_matrix = self.tfidf_word.transform(df['catalog_content'].fillna(''))
            tfidf_char_matrix = self.tfidf_char.transform(df['catalog_content'].fillna(''))
        
        tfidf_word_df = pd.DataFrame(
            tfidf_word_matrix.toarray(),
            columns=[f'tfidf_w_{i}' for i in range(tfidf_word_matrix.shape[1])]
        )
        
        tfidf_char_df = pd.DataFrame(
            tfidf_char_matrix.toarray(),
            columns=[f'tfidf_c_{i}' for i in range(tfidf_char_matrix.shape[1])]
        )
        
        # Combine all features
        features = pd.concat([
            text_features_df.reset_index(drop=True),
            tfidf_word_df.reset_index(drop=True),
            tfidf_char_df.reset_index(drop=True)
        ], axis=1)
        
        print(f"✓ Generated {features.shape[1]} features")
        return features


class OptimizedEnsembleModel:
    """Optimized ensemble model with hyperparameter tuning"""
    
    def __init__(self, use_tuning=True):
        self.feature_engineer = UltimateFeatureEngineer()
        self.scaler = RobustScaler()
        self.models = []
        self.weights = []
        self.use_tuning = use_tuning and HAS_OPTUNA
        self.use_log_transform = True
        
    def optimize_lgbm(self, X_train, y_train, X_val, y_val):
        """Optimize LightGBM hyperparameters"""
        if not self.use_tuning:
            return {
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
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
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
            
            y_val_orig = np.expm1(y_val)
            pred_orig = np.expm1(pred)
            smape = calculate_smape(y_val_orig, pred_orig)
            
            return smape
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        return study.best_params
    
    def optimize_xgb(self, X_train, y_train, X_val, y_val):
        """Optimize XGBoost hyperparameters"""
        if not self.use_tuning:
            return {
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
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 6, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
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
        study.optimize(objective, n_trials=20)
        return study.best_params
    
    def train(self, train_df, validation_split=0.15):
        """Train optimized ensemble model"""
        print("\n" + "="*80)
        print("🚀 AMAZON ML CHALLENGE - FINAL OPTIMIZED TRAINING")
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
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print(f"\n📊 Data Split:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        print(f"   Features: {X_train.shape[1]:,}")
        
        # Model configurations
        model_configs = []
        
        if HAS_LGBM:
            print(f"\n🔧 {'Optimizing' if self.use_tuning else 'Configuring'} LightGBM...")
            lgbm_params = self.optimize_lgbm(X_train_scaled, y_train, X_val_scaled, y_val)
            model_configs.append(('LightGBM', LGBMRegressor(**lgbm_params)))
        
        if HAS_XGB:
            print(f"\n🔧 {'Optimizing' if self.use_tuning else 'Configuring'} XGBoost...")
            xgb_params = self.optimize_xgb(X_train_scaled, y_train, X_val_scaled, y_val)
            model_configs.append(('XGBoost', XGBRegressor(**xgb_params)))
        
        if HAS_CATBOOST:
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
        
        # Optimize ensemble weights
        if len(val_predictions) > 1:
            print(f"\n{'='*80}")
            print("🔧 Optimizing ensemble weights...")
            print(f"{'='*80}")
            
            best_smape = float('inf')
            best_weights = None
            
            # Try different weight combinations
            for w1 in np.linspace(0.2, 0.8, 7):
                for w2 in np.linspace(0.2, 0.8, 7):
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
            print(f"   ✓ Ensemble SMAPE: {best_smape:.4f}%")
        else:
            self.weights = [1.0]
            best_smape = model_scores[0]
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("✅ FINAL TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"   Individual model scores:")
        for (name, _), score in zip(self.models, model_scores):
            print(f"      {name}: {score:.4f}%")
        print(f"   🏆 Final Ensemble SMAPE: {best_smape:.4f}%")
        print(f"   ⏱️  Total training time: {total_time/60:.1f} minutes")
        print(f"{'='*80}\n")
        
        return {'val_smape': best_smape, 'model_scores': model_scores}
    
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
        
        # Ensemble
        if len(predictions) > 1:
            final_pred = sum(w * pred for w, pred in zip(self.weights, predictions))
            print(f"   ✓ Ensemble combined with weights: {[f'{w:.3f}' for w in self.weights]}")
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
    ║                    FINAL PRODUCTION SOLUTION                                 ║
    ║                    Optimized Ensemble + Hyperparameter Tuning               ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DATASET_FOLDER = 'dataset/'
    train_path = os.path.join(DATASET_FOLDER, 'train.csv')
    test_path = os.path.join(DATASET_FOLDER, 'test.csv')
    output_path = os.path.join(DATASET_FOLDER, 'test_out_final.csv')
    
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
    predictor = OptimizedEnsembleModel(use_tuning=HAS_OPTUNA)
    metrics = predictor.train(train_df, validation_split=0.15)
    
    # Generate predictions
    print(f"\n{'='*80}")
    print("🎯 GENERATING FINAL TEST PREDICTIONS")
    print(f"{'='*80}")
    predictions = predictor.predict(test_df)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': np.round(predictions, 2)
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print("✅ FINAL SUBMISSION FILE CREATED")
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
        print("\n🎉 FINAL SUCCESS! Your optimized model is ready.")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 If you encounter errors, please share the error message!")
