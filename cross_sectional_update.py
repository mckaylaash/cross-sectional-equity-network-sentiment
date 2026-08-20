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

# Normalize column names (strip whitespace and standardize lowercase where applicable)
vader_avg.columns = vader_avg.columns.str.strip()
top200.columns = top200.columns.str.strip()
weekly_changes.columns = weekly_changes.columns.str.strip()

# Standardize ticker and week column names
if 'Week' in vader_avg.columns:
    vader_avg = vader_avg.rename(columns={'Week': 'week'})
if 'Week' in weekly_changes.columns:
    weekly_changes = weekly_changes.rename(columns={'Week': 'week'})

vader_avg = vader_avg[['matched_tickers', 'week', 'vader_score_avg']].drop_duplicates()
vader_avg['matched_tickers'] = vader_avg['matched_tickers'].astype(str).str.strip().str.upper()
vader_avg['week'] = vader_avg['week'].astype(int)

top200['Ticker'] = top200['Ticker'].astype(str).str.strip().str.upper()
weekly_changes['Ticker'] = weekly_changes['Ticker'].astype(str).str.strip().str.upper()
weekly_changes['week'] = weekly_changes['week'].astype(int)

# Extract Centrality Metrics from Al Guindy Top200
centrality_cols = ['Ticker', 'Industry', 'Eigen Centrality', 'Betweeness Centrality', 'Weighted Degree']
network_df = top200[centrality_cols].drop_duplicates(subset=['Ticker'])

# Merge Sentiment with Network Graph Topology
df = pd.merge(vader_avg, network_df, left_on='matched_tickers', right_on='Ticker', how='inner')

# ==========================================
# UPGRADE 1: MULTI-CENTRALITY NETWORK SPILLOVER
# ==========================================
def compute_centrality_spillovers(group):
    metrics = {
        'spillover_eigen': 'Eigen Centrality',
        'spillover_betweenness': 'Betweeness Centrality',
        'spillover_degree': 'Weighted Degree'
    }
    
    for spill_col, weight_col in metrics.items():
        total_weighted_sent = (group['vader_score_avg'] * group[weight_col]).sum()
        total_weight = group[weight_col].sum()
        
        # Peer spillover = (Sector Total - Own) / (Total Sector Weight - Own Weight)
        peer_sent = (total_weighted_sent - (group['vader_score_avg'] * group[weight_col])) / (total_weight - group[weight_col] + 1e-6)
        group[spill_col] = peer_sent
        
    return group

# Calculate spillovers and ensure 'week' stays as a flat column
df = df.groupby(['Industry', 'week'], group_keys=False).apply(compute_centrality_spillovers)
df = df.reset_index(drop=True)

# ==========================================
# FORWARD RETURN ALIGNMENT (PREDICT t+1)
# ==========================================
# Shift target week backwards so week t sentiment maps to week t+1 return
weekly_changes['Predict_For_Week'] = weekly_changes['week'] - 1

df_final = pd.merge(
    df,
    weekly_changes[['Ticker', 'Predict_For_Week', 'Price Change (%)']],
    left_on=['matched_tickers', 'week'],
    right_on=['Ticker', 'Predict_For_Week'],
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
        ic, _ = spearmanr(group['composite_alpha'], group['Price Change (%)'])
        if not np.isnan(ic):
            ics.append(ic)
            
        # 2. Quintile Long/Short Portfolio Spread
        group = group.sort_values(by='composite_alpha')
        q_size = max(1, len(group) // 5)
        short_basket = group.head(q_size)['Price Change (%)'].mean()
        long_basket = group.tail(q_size)['Price Change (%)'].mean()
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

industry_dummies = pd.get_dummies(df_final['Industry'], prefix='ind', drop_first=True)

X_asym = pd.concat([
    df_final[['pos_sentiment', 'neg_sentiment', 'spillover_eigen', 'spillover_betweenness']], 
    industry_dummies
], axis=1).astype(float)

X_asym = sm.add_constant(X_asym)
y_asym = df_final['Price Change (%)'].astype(float)

asym_model = sm.OLS(y_asym, X_asym).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
print(asym_model.summary().tables[1])

# ==========================================
# LOOKAHEAD-FREE CLASSIFIERS (BENCHMARK)
# ==========================================
print("\n" + "="*50)
print("CHRONOLOGICAL ML CLASSIFICATION (OUT-OF-SAMPLE)")
print("="*50)

df_final['target_label'] = (df_final['Price Change (%)'] > 0).astype(int)

features = [
    'vader_score_avg', 
    'spillover_eigen', 
    'spillover_betweenness', 
    'spillover_degree', 
    'Eigen Centrality', 
    'Betweeness Centrality', 
    'Weighted Degree'
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