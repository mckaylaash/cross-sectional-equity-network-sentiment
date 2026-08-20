import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# ==========================================
# 1. LOAD AND PREPARE DATASETS
# ==========================================
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

vader_avg = pd.read_csv(os.path.join(DATA_DIR, 'vader_avg.csv'))
top200 = pd.read_csv(os.path.join(DATA_DIR, 'Top200_2022.csv'))
weekly_changes = pd.read_csv(os.path.join(DATA_DIR, 'priceChangeData.csv'))

# Standardize columns to lowercase
vader_avg.columns = vader_avg.columns.str.strip().str.lower()
top200.columns = top200.columns.str.strip().str.lower()
weekly_changes.columns = weekly_changes.columns.str.strip().str.lower()

# Ensure standard column names
vader_avg = vader_avg[['matched_tickers', 'week', 'vader_score_avg']].drop_duplicates()
vader_avg['matched_tickers'] = vader_avg['matched_tickers'].astype(str).str.strip().str.upper()
vader_avg['week'] = vader_avg['week'].astype(int)

top200['ticker'] = top200['ticker'].astype(str).str.strip().str.upper()
weekly_changes['ticker'] = weekly_changes['ticker'].astype(str).str.strip().str.upper()
weekly_changes['week'] = weekly_changes['week'].astype(int)

# Extract Centrality Metrics
centrality_cols = ['ticker', 'industry', 'eigen centrality', 'betweeness centrality', 'weighted degree']
network_df = top200[centrality_cols].drop_duplicates(subset=['ticker'])

# Merge Sentiment with Network Graph
df = pd.merge(vader_avg, network_df, left_on='matched_tickers', right_on='ticker', how='inner')

# ==========================================
# UPGRADE 1: VECTORIZED NETWORK SPILLOVER (NO INDEX CORRUPTION)
# ==========================================
metrics = {
    'spillover_eigen': 'eigen centrality',
    'spillover_betweenness': 'betweeness centrality',
    'spillover_degree': 'weighted degree'
}

for spill_col, weight_col in metrics.items():
    # Calculate product of sentiment and centrality weight
    df['temp_weighted_sent'] = df['vader_score_avg'] * df[weight_col]
    
    # Calculate sector-wide totals per week
    sector_total_sent = df.groupby(['industry', 'week'])['temp_weighted_sent'].transform('sum')
    sector_total_weight = df.groupby(['industry', 'week'])[weight_col].transform('sum')
    
    # Peer spillover = (Sector Total - Own) / (Total Sector Weight - Own Weight)
    peer_sent_sum = sector_total_sent - df['temp_weighted_sent']
    peer_weight_sum = sector_total_weight - df[weight_col]
    
    df[spill_col] = peer_sent_sum / (peer_weight_sum + 1e-6)

df = df.drop(columns=['temp_weighted_sent'])

# ==========================================
# FORWARD RETURN ALIGNMENT (PREDICT t+1)
# ==========================================
# Shift target week backwards so week t sentiment maps to week t+1 return
weekly_changes['predict_for_week'] = weekly_changes['week'] - 1

df_final = pd.merge(
    df,
    weekly_changes[['ticker', 'predict_for_week', 'price change (%)']],
    left_on=['matched_tickers', 'week'],
    right_on=['ticker', 'predict_for_week'],
    how='inner'
)

# Composite Multi-Centrality Alpha Signal
df_final['composite_alpha'] = (
    df_final['vader_score_avg'] + 
    df_final['spillover_eigen'] + 
    df_final['spillover_betweenness'] + 
    df_final['spillover_degree']
)

# ==========================================
# UPGRADE 2: CROSS-SECTIONAL FACTOR (RANK IC & L/S)
# ==========================================
print("\n" + "="*50)
print("UPGRADE 2: CROSS-SECTIONAL FACTOR PERFORMANCE")
print("="*50)

ics = []
long_short_returns = []

for w, group in df_final.groupby('week'):
    if len(group) >= 5:
        # 1. Rank Information Coefficient (IC)
        ic, _ = spearmanr(group['composite_alpha'], group['price change (%)'])
        if not np.isnan(ic):
            ics.append(ic)
            
        # 2. Quintile Long/Short Portfolio Spread
        group = group.sort_values(by='composite_alpha')
        q_size = max(1, len(group) // 5)
        short_basket = group.head(q_size)['price change (%)'].mean()
        long_basket = group.tail(q_size)['price change (%)'].mean()
        long_short_returns.append(long_basket - short_basket)

mean_ic = np.mean(ics) if ics else 0
icir = (mean_ic / np.std(ics)) if len(ics) > 1 and np.std(ics) > 0 else 0
mean_ls = np.mean(long_short_returns) if long_short_returns else 0

print(f"Mean Cross-Sectional Rank IC : {mean_ic:.4f}")
print(f"Information Ratio (ICIR)     : {icir:.2f}")
print(f"Average Weekly Long/Short Spread: {mean_ls:.2f}%")

# ==========================================
# UPGRADE 3: SHORT-SELLING ASYMMETRY TEST (OLS)
# ==========================================
print("\n" + "="*50)
print("UPGRADE 3: SHORT-SELLING ASYMMETRY TEST (OLS)")
print("="*50)

# Decompose sentiment into positive vs negative shocks
df_final['pos_sentiment'] = np.maximum(0, df_final['vader_score_avg'])
df_final['neg_sentiment'] = np.minimum(0, df_final['vader_score_avg'])

industry_dummies = pd.get_dummies(df_final['industry'], prefix='ind', drop_first=True)

X_asym = pd.concat([
    df_final[['pos_sentiment', 'neg_sentiment', 'spillover_eigen', 'spillover_betweenness']], 
    industry_dummies
], axis=1).astype(float)

X_asym = sm.add_constant(X_asym)
y_asym = df_final['price change (%)'].astype(float)

asym_model = sm.OLS(y_asym, X_asym).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
print(asym_model.summary().tables[1])

# ==========================================
# LOOKAHEAD-FREE CLASSIFIERS (BENCHMARK)
# ==========================================
print("\n" + "="*50)
print("CHRONOLOGICAL ML CLASSIFICATION (OUT-OF-SAMPLE)")
print("="*50)

df_final['target_label'] = (df_final['price change (%)'] > 0).astype(int)

features = [
    'vader_score_avg', 
    'spillover_eigen', 
    'spillover_betweenness', 
    'spillover_degree', 
    'eigen centrality', 
    'betweeness centrality', 
    'weighted degree'
]

X_ml = df_final[features]
y_ml = df_final['target_label']

# Chronological split: Train on Weeks 0 & 1, Test strictly on Week 2 forward horizon
train_mask = df_final['week'] < 2
test_mask = df_final['week'] >= 2

X_train, y_train = X_ml[train_mask], y_ml[train_mask]
X_test, y_test = X_ml[test_mask], y_ml[test_mask]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifiers = {
    "Logistic Regression (L2 Regularized)": LogisticRegression(C=0.1, max_iter=1000),
    "Random Forest (Constrained Depth)": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
}

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ==========================================
# EXPORT RESULTS TO CSV AND TXT
# ==========================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Save detailed regression coefficients to CSV
reg_summary_df = pd.DataFrame({
    'Feature': asym_model.params.index,
    'Coefficient': asym_model.params.values,
    'Std_Error': asym_model.bse.values,
    't_stat': asym_model.tvalues.values,
    'p_value': asym_model.pvalues.values
})
reg_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'regression_asymmetry_results.csv'), index=False)

# 2. Save factor evaluation metrics to CSV
factor_summary_df = pd.DataFrame([{
    'Metric': 'Mean Cross-Sectional Rank IC',
    'Value': mean_ic
}, {
    'Metric': 'Information Ratio (ICIR)',
    'Value': icir
}, {
    'Metric': 'Average Weekly Long/Short Spread (%)',
    'Value': mean_ls
}])
factor_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'factor_summary_metrics.csv'), index=False)

# 3. Save the final aligned dataset with network spillovers to CSV
df_final.to_csv(os.path.join(OUTPUT_DIR, 'final_model_features_dataset.csv'), index=False)

# 4. Save a full formatted summary report to a .txt file
with open(os.path.join(OUTPUT_DIR, 'model_results_report.txt'), 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("QUANTITATIVE RESEARCH REPORT: AL GUINDY NETWORK SENTIMENT\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("1. CROSS-SECTIONAL FACTOR METRICS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Mean Rank IC                  : {mean_ic:.4f}\n")
    f.write(f"Information Ratio (ICIR)      : {icir:.2f}\n")
    f.write(f"Average Weekly L/S Spread (%) : {mean_ls:.2f}%\n\n")
    
    f.write("2. SHORT-SELLING ASYMMETRY REGRESSION (HAC OLS)\n")
    f.write("-" * 40 + "\n")
    f.write(asym_model.summary().as_text() + "\n\n")
    
    f.write("3. CHRONOLOGICAL ML BENCHMARKS (OUT-OF-SAMPLE)\n")
    f.write("-" * 40 + "\n")
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        
        f.write(f"--- {name} ---\n")
        f.write(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}\n")
        f.write(classification_report(y_test, y_pred, zero_division=0) + "\n")

print("\n" + "=" * 50)
print("Files saved successfully to the project folder:")
print(" • regression_asymmetry_results.csv")
print(" • factor_summary_metrics.csv")
print(" • final_model_features_dataset.csv")
print(" • model_results_report.txt")
print("=" * 50)