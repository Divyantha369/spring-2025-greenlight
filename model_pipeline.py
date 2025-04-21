import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance  # Ensure this import exists
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import shap
import optuna
import optuna.logging
import warnings
warnings.filterwarnings('ignore')

class EnhancedMovieBoxOfficePipeline:
    """ ML pipeline for movie box office prediction with hyperparameter tuning and stacking"""
    
    def __init__(self, numeric_features, categorical_features, target='success_ratio'):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.features = numeric_features + categorical_features
        self.target = target
        self.results = {}
        self.best_params = {}
        self.verbose = 1  # Default verbosity level

    def run(self, df, test_size=0.2, tune_hyperparams=True, 
            n_trials=20, n_folds=5, alpha=1.0, base_models=None, stackable_models=None, verbose=1, **kwargs):
        """hyperparameter tuning and stacking"""
        # Set verbosity level
        self.verbose = verbose
        
        # Data preparation
        self._split_data(df, test_size)
        self._preprocess_data()
        
        # base models 
        if base_models is None:
            base_models = ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'SVR','Lasso','KNN']
        
        # hyperparameter tuning 
        if tune_hyperparams:
            self._tune_hyperparameters(base_models, n_trials, n_folds)
        
        # train base models
        self._train_base_models(base_models, **kwargs)
        
        # stacked ensemble
        if stackable_models is None:
            stackable_models = [m for m in base_models if m not in ['Lasso', 'KNN']]
        self._create_stacked_ensemble(models_to_stack=stackable_models, n_folds=n_folds, alpha=alpha)

        # permutations
        self._calculate_permutation_importance()
        
        # Calculate SHAP values
        self._calculate_shap_values()
        
        # Create standard visualizations
        self._create_visualizations()
        
        return self.results
    
    def _split_data(self, df, test_size):
        """Split data temporally for time-aware evaluation"""
        df = df.sort_values(by='release_date')
        split_idx = int(len(df) * (1 - test_size))
        self.train_df = df.iloc[:split_idx]
        self.test_df = df.iloc[split_idx:]
        if self.verbose > 0:
            print(f"Training: {self.train_df['release_date'].min()} to {self.train_df['release_date'].max()} ({len(self.train_df)} movies)")
            print(f"Testing: {self.test_df['release_date'].min()} to {self.test_df['release_date'].max()} ({len(self.test_df)} movies)")
    
    def _preprocess_data(self):
        """Preprocess features and targets with appropriate scaling and encoding"""
        # Extract features
        self.X_train = self.train_df[self.features]
        self.X_test = self.test_df[self.features]
        
        # Extract target
        self.y_train = self.train_df[self.target]
        self.y_test = self.test_df[self.target]
        
        # Scale numeric features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        
        if self.numeric_features:
            self.X_train_scaled[self.numeric_features] = self.scaler.fit_transform(
                self.X_train[self.numeric_features])
            self.X_test_scaled[self.numeric_features] = self.scaler.transform(
                self.X_test[self.numeric_features])
        
        # Get categorical features indices for models that need them
        self.cat_features_idx = [self.X_train.columns.get_loc(col) for col in self.categorical_features]
    
    def _tune_hyperparameters(self, model_types, n_trials=20, n_folds=5):
        """Tune hyperparameters using Optuna for all selected models"""
        if self.verbose > 0:
            print(f"\n=== Tuning Hyperparameters for Selected Models ===")
        
        # Set up cross-validation
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        # Define parameter spaces for each model type
        valid_model_types = ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'SVR', 'Lasso', 'KNN']
        param_spaces = {}


        if 'Lasso' in model_types:
            param_spaces['Lasso'] = {
                'alpha': optuna.distributions.FloatDistribution(0.001, 10.0, log=True),
                'max_iter': optuna.distributions.IntDistribution(1000, 5000),
                'tol': optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True),
                'selection': optuna.distributions.CategoricalDistribution(['cyclic', 'random'])
            }
    
   
        if 'KNN' in model_types:
            param_spaces['KNN'] = {
                'n_neighbors': optuna.distributions.IntDistribution(1, 50),
                'weights': optuna.distributions.CategoricalDistribution(['uniform', 'distance']),
                'algorithm': optuna.distributions.CategoricalDistribution(['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': optuna.distributions.IntDistribution(10, 50),
                'p': optuna.distributions.CategoricalDistribution([1, 2])  # Manhattan or Euclidean distance
            }
        
        if 'RandomForest' in model_types:
            param_spaces['RandomForest'] = {
                'n_estimators': optuna.distributions.IntDistribution(50, 500),
                'max_depth': optuna.distributions.IntDistribution(3, 20),
                'min_samples_split': optuna.distributions.IntDistribution(2, 20),
                'min_samples_leaf': optuna.distributions.IntDistribution(1, 20)
            }
            
        if 'XGBoost' in model_types:
            param_spaces['XGBoost'] = {
                'n_estimators': optuna.distributions.IntDistribution(50, 500),
                'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                'max_depth': optuna.distributions.IntDistribution(3, 15),
                'subsample': optuna.distributions.FloatDistribution(0.5, 1.0),
                'colsample_bytree': optuna.distributions.FloatDistribution(0.5, 1.0),
                'reg_alpha': optuna.distributions.FloatDistribution(0.001, 10.0, log=True),
                'reg_lambda': optuna.distributions.FloatDistribution(0.001, 10.0, log=True)
            }
            
        if 'LightGBM' in model_types:
            param_spaces['LightGBM'] = {
                'n_estimators': optuna.distributions.IntDistribution(50, 500),
                'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                'max_depth': optuna.distributions.IntDistribution(3, 15),
                'num_leaves': optuna.distributions.IntDistribution(8, 256),
                'subsample': optuna.distributions.FloatDistribution(0.5, 1.0),
                'colsample_bytree': optuna.distributions.FloatDistribution(0.5, 1.0),
                'reg_alpha': optuna.distributions.FloatDistribution(0.001, 10.0, log=True),
                'reg_lambda': optuna.distributions.FloatDistribution(0.001, 10.0, log=True)
            }
            
        if 'CatBoost' in model_types:
            param_spaces['CatBoost'] = {
                'iterations': optuna.distributions.IntDistribution(50, 500),
                'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                'depth': optuna.distributions.IntDistribution(3, 10),
                'l2_leaf_reg': optuna.distributions.FloatDistribution(0.1, 10.0, log=True),
                'subsample': optuna.distributions.FloatDistribution(0.5, 1.0)
            }
            
        if 'SVR' in model_types:
            param_spaces['SVR'] = {
                'C': optuna.distributions.FloatDistribution(0.1, 100.0, log=True),
                'epsilon': optuna.distributions.FloatDistribution(0.01, 1.0, log=True),
                'gamma': optuna.distributions.CategoricalDistribution(['scale', 'auto']),
                'kernel': optuna.distributions.CategoricalDistribution(['linear', 'rbf', 'poly', 'sigmoid'])
            }

        # Set Optuna logging level based on verbosity
        if self.verbose < 2:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Tune each model type
        for model_name, param_space in param_spaces.items():
            if self.verbose > 0:
                print(f"\nTuning {model_name}...")
            
            def objective(trial):
                # Create parameter dict for this trial
                params = {k: trial.suggest_categorical(k, [v]) if isinstance(v, list) else trial._suggest(k, v) 
                          for k, v in param_space.items()}
                
                # Add random_state for reproducibility where applicable
                if model_name not in ['SVR','KNN']:
                    params['random_state'] = 42
                
                # Initialize appropriate model
                if model_name == 'RandomForest':
                    model = RandomForestRegressor(**params)
                elif model_name == 'XGBoost':
                    model = xgb.XGBRegressor(**params, verbosity=0)
                elif model_name == 'LightGBM':
                    model = lgb.LGBMRegressor(**params, verbose=-1)  # Added verbose=-1
                elif model_name == 'CatBoost':
                    model = CatBoostRegressor(**params, verbose=False)
                elif model_name == 'SVR':
                    model = SVR(**params)
                elif model_name == 'Lasso':
                    model = Lasso(**params)
                elif model_name == 'KNN':
                    model = KNeighborsRegressor(**params)
                
                # Cross-validate
                cv_scores = []
                for train_idx, val_idx in tscv.split(self.X_train):
                    X_train_cv, X_val_cv = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                    y_train_cv, y_val_cv = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                    
                    # Scale features within each fold to prevent data leakage
                    X_train_cv_scaled = X_train_cv.copy()
                    X_val_cv_scaled = X_val_cv.copy()
                    
                    if self.numeric_features:
                        scaler = StandardScaler()
                        X_train_cv_scaled[self.numeric_features] = scaler.fit_transform(X_train_cv[self.numeric_features])
                        X_val_cv_scaled[self.numeric_features] = scaler.transform(X_val_cv[self.numeric_features])
                    
                    # Fit model with special handling for LightGBM and CatBoost
                    if model_name == 'LightGBM':
                        model.fit(X_train_cv_scaled, y_train_cv, categorical_feature=self.cat_features_idx)
                    elif model_name == 'CatBoost':
                        cat_features = self.cat_features_idx if self.cat_features_idx else None
                        model.fit(X_train_cv_scaled, y_train_cv, cat_features=cat_features)
                    else:
                        model.fit(X_train_cv_scaled, y_train_cv)
                    
                    # Evaluate
                    preds = model.predict(X_val_cv_scaled)
                    score = np.sqrt(mean_squared_error(y_val_cv, preds))
                    cv_scores.append(score)
                
                return np.mean(cv_scores)
            
            # Use context manager to capture unwanted output for low verbosity
            import contextlib
            import io
            
            # Create study
            study = optuna.create_study(direction='minimize')
            
            if self.verbose < 2:
                # Capture all output during optimization
                f = io.StringIO()
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            else:
                # Show all output
                study.optimize(objective, n_trials=n_trials)
            
            # Store best parameters
            self.best_params[model_name] = study.best_params
            
            if self.verbose > 0:
                print(f"Best parameters for {model_name}: {study.best_params}")
                print(f"Best RMSE: {study.best_value:.4f}")
    
    def _train_base_models(self, model_types, **kwargs):
        """Train all selected base models with optimal hyperparameters"""
        if self.verbose > 0:
            print("\n=== Training Base Models ===")
        
        # Models
        available_models = {
            'RandomForest': lambda params: RandomForestRegressor(
                **{**{'n_estimators': 100, 'random_state': 42}, **params}),
            'XGBoost': lambda params: xgb.XGBRegressor(
                **{**{'random_state': 42, 'verbosity': 0}, **params}),
            'LightGBM': lambda params: lgb.LGBMRegressor(
                **{**{'random_state': 42, 'verbose': -1}, **params}),  # Added verbose=-1
            'CatBoost': lambda params: CatBoostRegressor(
                **{**{'random_state': 42, 'verbose': False}, **params}),
            'SVR': lambda params: SVR(**params),
            'Lasso': lambda params: Lasso(
                **{**{'random_state': 42}, **params}),
            'KNN': lambda params: KNeighborsRegressor(**params)
        }
        
        # Filter to only train requested models
        models = {k: v for k, v in available_models.items() if k in model_types}
        
        # Train each model
        for name, model_initializer in models.items():
            if self.verbose > 0:
                print(f"\nTraining {name} Model...")
            
            # Use tuned parameters if available
            params = self.best_params.get(name, {})
            
            # Initialize model
            model = model_initializer(params)
            
            # Train with special handling for LightGBM and CatBoost
            if name == 'LightGBM':
                model.fit(self.X_train_scaled, self.y_train, categorical_feature=self.cat_features_idx)
            elif name == 'CatBoost':
                cat_features = self.cat_features_idx if self.cat_features_idx else None
                model.fit(self.X_train_scaled, self.y_train, cat_features=cat_features)
            else:
                model.fit(self.X_train_scaled, self.y_train)
            
            # Evaluate
            preds = model.predict(self.X_test_scaled)
            rmse = np.sqrt(mean_squared_error(self.y_test, preds))
            
            # Store results
            self.results[name] = {
                'model': model,
                'rmse': rmse,
                'predictions': preds
            }
            
            if hasattr(model, 'feature_importances_'):
                self.results[name]['feature_importances'] = pd.Series(
                    model.feature_importances_, index=self.features)
            
            if self.verbose > 0:
                print(f"{name} RMSE: {rmse:.4f}")
    
    def _create_stacked_ensemble(self, models_to_stack=None, n_folds=5, alpha=1.0):
        """Create a stacked ensemble using all trained base models"""
        if self.verbose > 0:
            print("\n=== Creating Stacked Ensemble ===")

        base_models = [model_name for model_name in self.results.keys() 
                      if model_name not in ['Lasso', 'KNN']]
        
        
        if not base_models:
            print("No base models found. Train base models first.")
            return
        
        if self.verbose > 0:
            print(f"Stacking models: {', '.join(base_models)}")
        
        # Set up cross-validation
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        # Initialize arrays for out-of-fold predictions
        oof_train = np.zeros((self.X_train.shape[0], len(base_models)))
        oof_test = np.zeros((self.X_test.shape[0], len(base_models)))
        
        # Generate out-of-fold predictions for training meta-model
        if self.verbose > 0:
            print("\nGenerating out-of-fold predictions...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            if self.verbose > 0:
                print(f"Fold {fold_idx+1}/{n_folds}")
            
            fold_X_train = self.X_train.iloc[train_idx]
            fold_y_train = self.y_train.iloc[train_idx]
            fold_X_val = self.X_train.iloc[val_idx]
            
            # Scale features within each fold to prevent data leakage
            fold_X_train_scaled = fold_X_train.copy()
            fold_X_val_scaled = fold_X_val.copy()
            
            if self.numeric_features:
                scaler = StandardScaler()
                fold_X_train_scaled[self.numeric_features] = scaler.fit_transform(fold_X_train[self.numeric_features])
                fold_X_val_scaled[self.numeric_features] = scaler.transform(fold_X_val[self.numeric_features])
            
            # Train and predict with each base model
            for i, name in enumerate(base_models):
                # Get model class
                if name == 'RandomForest':
                    model = RandomForestRegressor(**self.best_params.get(name, {'n_estimators': 100}), random_state=42)
                elif name == 'XGBoost':
                    model = xgb.XGBRegressor(**self.best_params.get(name, {}), random_state=42, verbosity=0)
                elif name == 'LightGBM':
                    model = lgb.LGBMRegressor(**self.best_params.get(name, {}), random_state=42, verbose=-1)  # Added verbose=-1
                elif name == 'CatBoost':
                    model = CatBoostRegressor(**self.best_params.get(name, {}), random_state=42, verbose=False)
                elif name == 'SVR':
                    model = SVR(**self.best_params.get(name, {}))
                else:
                    continue  # Skip unknown models
                
                # Train the model
                if name == 'LightGBM':
                    model.fit(fold_X_train_scaled, fold_y_train, categorical_feature=self.cat_features_idx)
                elif name == 'CatBoost':
                    cat_features = self.cat_features_idx if self.cat_features_idx else None
                    model.fit(fold_X_train_scaled, fold_y_train, cat_features=cat_features)
                else:
                    model.fit(fold_X_train_scaled, fold_y_train)
                
                # Generate OOF predictions
                oof_train[val_idx, i] = model.predict(fold_X_val_scaled)
                
                # Also predict on test set for final ensemble
                oof_test[:, i] += model.predict(self.X_test_scaled) / n_folds
        
        # Always use Ridge as meta-model with regularization
        meta_model = Ridge(alpha=alpha, random_state=42)
        
        # Train meta-model on out-of-fold predictions
        meta_model.fit(oof_train, self.y_train)
        
        # Make final predictions
        stacked_preds = meta_model.predict(oof_test)
        stacked_rmse = np.sqrt(mean_squared_error(self.y_test, stacked_preds))
        
        # Store results
        self.results['Stacked'] = {
            'rmse': stacked_rmse,
            'meta_model': meta_model,
            'predictions': stacked_preds,
            'model_weights': pd.Series(meta_model.coef_, index=base_models)
        }
        
        if self.verbose > 0:
            print(f"\nStacked Ensemble RMSE: {stacked_rmse:.4f}")
            
            # Print model weights
            print("\nModel weights in ensemble:")
            weights = pd.Series(meta_model.coef_, index=base_models)
            for model_name, weight in weights.items():
                print(f"{model_name}: {weight:.4f}")
            
        return self.results['Stacked']

    def _calculate_permutation_importance(self):
        """Calculate permutation-based feature importance for all trained models"""
        # Explicitly import permutation_importance locally to ensure availability
        from sklearn.inspection import permutation_importance
        
        if self.verbose > 0:
            print("\n=== Calculating Permutation Feature Importance ===")
    
        # Get all trained models (excluding stacked ensemble)
        all_models = {k: v['model'] for k, v in self.results.items() 
                 if 'model' in v and k != 'Stacked' and k not in ['Lasso', 'KNN']}
    
        for model_key, model in all_models.items():
            try:
                if self.verbose > 0:
                    print(f"Calculating permutation importance for {model_key}...")
            
                # Calculate permutation importance
                perm_importance = permutation_importance(
                    model, self.X_test_scaled, self.y_test,
                    n_repeats=10, random_state=42, n_jobs=-1
                )
            
                # Store results for later visualization
                self.results[model_key]['perm_importance'] = {
                    'importances_mean': perm_importance.importances_mean,
                    'importances_std': perm_importance.importances_std,
                    'feature_names': self.features
                }
            
            except Exception as e:
                print(f"Error calculating permutation importance for {model_key}: {str(e)}")

    def _calculate_shap_values(self):
        """Calculate SHAP values for all trained models without visualization"""
        if self.verbose > 0:
            print("\n=== Calculating SHAP Values ===")
    
        # Get all trained models
        all_models = {k: v['model'] for k, v in self.results.items() 
                 if 'model' in v and k != 'Stacked' and k not in ['Lasso', 'KNN']}
    
        for model_key, model in all_models.items():
            try:
                if self.verbose > 0:
                    print(f"Generating SHAP values for {model_key}...")
            
                # Skip SVR (not compatible with TreeExplainer)
                if model_key == 'SVR':
                    if self.verbose > 0:
                        print(f"Skipping SHAP calculation for {model_key} - not compatible with TreeExplainer")
                    continue
            
                # Define X_data before any branching logic
                X_data = None
                shap_values = None
                explainer = None
            
                # Create appropriate explainer based on model type
                if model_key in ['XGBoost', 'LightGBM', 'RandomForest', 'CatBoost']:
                    explainer = shap.TreeExplainer(model)
                    X_data = self.X_test_scaled
                    shap_values = explainer.shap_values(X_data)
                else:
                    # For other model types
                    sample_size = min(100, len(self.X_test))
                    X_data = self.X_test_scaled.sample(sample_size, random_state=42)
                    explainer = shap.KernelExplainer(model.predict, X_data)
                    shap_values = explainer.shap_values(X_data)
                
                # Store SHAP values for later use
                self.results[model_key]['shap_values'] = shap_values
                self.results[model_key]['shap_explainer'] = explainer
                self.results[model_key]['shap_data'] = X_data
                
            except Exception as e:
                print(f"Error generating SHAP values for {model_key}: {str(e)}")
    
    def _create_visualizations(self):
        """Create visualizations of model performance"""
        # Performance comparison across all models
        models = list(self.results.keys())
        rmse_values = [self.results[k]['rmse'] for k in models]
        
        # Sort models by performance
        sorted_indices = np.argsort(rmse_values)
        sorted_models = [models[i] for i in sorted_indices]
        sorted_rmse = [rmse_values[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        
        # Use different colors for Stacked model
        colors = ['#1f77b4'] * len(sorted_models)
        for i, model_name in enumerate(sorted_models):
            if model_name == 'Stacked':
                colors[i] = '#d62728'
        
        plt.bar(range(len(sorted_models)), sorted_rmse, color=colors)
        plt.title('Model Performance Comparison - RMSE (lower is better)')
        plt.ylabel('RMSE')
        plt.xticks(range(len(sorted_models)), sorted_models, rotation=45)
        
        # Annotate values
        for i, v in enumerate(sorted_rmse):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_performance.png')
        plt.close()
        
        # Visualize stacking model weights
        if 'Stacked' in self.results and 'model_weights' in self.results['Stacked']:
            plt.figure(figsize=(10, 6))
            weights = self.results['Stacked']['model_weights']
            
            # Sort weights by absolute magnitude
            weights = weights.reindex(weights.abs().sort_values(ascending=False).index)
            
            # Create color map based on sign
            colors = ['#2ca02c' if w > 0 else '#d62728' for w in weights]
            
            plt.bar(weights.index, weights.values, color=colors)
            plt.title('Model Weights in Stacked Ensemble')
            plt.ylabel('Weight')
            plt.xticks(rotation=45)
            
            # Annotate values
            for i, v in enumerate(weights.values):
                plt.text(i, v, f'{v:.3f}', ha='center', va='bottom' if v > 0 else 'top')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig('stacked_model_weights.png')
            plt.close()
            
            # Create feature importance visualization
            if any('feature_importances' in self.results[model] for model in models if model != 'Stacked'):
                plt.figure(figsize=(12, 8))
                
                # Aggregate feature importances across models
                agg_importance = pd.DataFrame()
                
                for model in models:
                    if model != 'Stacked' and 'feature_importances' in self.results[model]:
                        imp = self.results[model]['feature_importances']
                        # Normalize importances
                        imp = imp / imp.sum()
                        agg_importance[model] = imp
                
                # Calculate average importance
                if not agg_importance.empty:
                    agg_importance['Average'] = agg_importance.mean(axis=1)
                    
                    # Sort by average importance
                    agg_importance = agg_importance.sort_values('Average', ascending=False)
                    
                    # Plot top 15 features
                    top_features = agg_importance.head(15).index
                    agg_importance.loc[top_features, 'Average'].sort_values().plot(kind='barh')
                    
                    plt.title('Average Feature Importance Across Models')
                    plt.xlabel('Normalized Importance')
                    plt.tight_layout()
                    plt.savefig('average_feature_importance.png')
                    plt.close()