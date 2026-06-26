import matplotlib.pyplot as plt
import pandas as pd

# Create professional visualizations for feature importance
# Select top 15 features for clarity
top_n = 15
top_features = feature_importance_df.head(top_n).copy()

# Zerve design system colors
bg_color = '#1D1D20'
text_color = '#fbfbff'
secondary_text = '#909094'
highlight_color = '#ffd400'
colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', 
          '#1F77B4', '#9467BD', '#8C564B', '#C49C94', '#E377C2']

# Chart 1: Gain-based Feature Importance
fig1, ax1 = plt.subplots(figsize=(12, 8), facecolor=bg_color)
ax1.set_facecolor(bg_color)

bars1 = ax1.barh(range(top_n), top_features['gain'].values, color=colors[0], edgecolor=colors[0], linewidth=1.5)
ax1.set_yticks(range(top_n))
ax1.set_yticklabels(top_features['feature'].values, fontsize=11, color=text_color)
ax1.set_xlabel('Gain (Average Split Improvement)', fontsize=12, color=text_color, fontweight='bold')
ax1.set_title(f'Top {top_n} Features by Gain\nClaim Prediction Model', 
              fontsize=14, color=text_color, fontweight='bold', pad=20)
ax1.tick_params(colors=text_color, labelsize=10)
ax1.invert_yaxis()

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, top_features['gain'].values)):
    ax1.text(val + 0.3, i, f'{val:.1f}', va='center', ha='left', color=text_color, fontsize=9)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color(secondary_text)
ax1.spines['left'].set_color(secondary_text)
ax1.grid(axis='x', alpha=0.2, color=secondary_text)

plt.tight_layout()
gain_chart = fig1
print("✓ Created Gain-based importance chart")

# Chart 2: Weight-based Feature Importance
fig2, ax2 = plt.subplots(figsize=(12, 8), facecolor=bg_color)
ax2.set_facecolor(bg_color)

# Sort by weight for this chart
top_features_weight = feature_importance_df.nlargest(top_n, 'weight').copy()

bars2 = ax2.barh(range(top_n), top_features_weight['weight'].values, color=colors[1], edgecolor=colors[1], linewidth=1.5)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels(top_features_weight['feature'].values, fontsize=11, color=text_color)
ax2.set_xlabel('Weight (Number of Times Used in Splits)', fontsize=12, color=text_color, fontweight='bold')
ax2.set_title(f'Top {top_n} Features by Weight\nClaim Prediction Model', 
              fontsize=14, color=text_color, fontweight='bold', pad=20)
ax2.tick_params(colors=text_color, labelsize=10)
ax2.invert_yaxis()

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, top_features_weight['weight'].values)):
    ax2.text(val + 50, i, f'{int(val)}', va='center', ha='left', color=text_color, fontsize=9)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color(secondary_text)
ax2.spines['left'].set_color(secondary_text)
ax2.grid(axis='x', alpha=0.2, color=secondary_text)

plt.tight_layout()
weight_chart = fig2
print("✓ Created Weight-based importance chart")

# Chart 3: Combined comparison scatter plot
fig3, ax3 = plt.subplots(figsize=(12, 8), facecolor=bg_color)
ax3.set_facecolor(bg_color)

# Normalize both metrics for comparison (0-100 scale)
top_features_combined = feature_importance_df.head(20).copy()
top_features_combined['gain_norm'] = (top_features_combined['gain'] / top_features_combined['gain'].max()) * 100
top_features_combined['weight_norm'] = (top_features_combined['weight'] / top_features_combined['weight'].max()) * 100

scatter = ax3.scatter(top_features_combined['gain_norm'], top_features_combined['weight_norm'], 
                     s=200, alpha=0.7, color=colors[2], edgecolors=colors[4], linewidth=2)

# Label each point with feature name
for idx, row in top_features_combined.iterrows():
    ax3.annotate(row['feature'], 
                (row['gain_norm'], row['weight_norm']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, color=text_color, alpha=0.9)

ax3.set_xlabel('Gain (Normalized, 0-100)', fontsize=12, color=text_color, fontweight='bold')
ax3.set_ylabel('Weight (Normalized, 0-100)', fontsize=12, color=text_color, fontweight='bold')
ax3.set_title('Feature Importance: Gain vs Weight Comparison\nTop 20 Features', 
              fontsize=14, color=text_color, fontweight='bold', pad=20)
ax3.tick_params(colors=text_color, labelsize=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_color(secondary_text)
ax3.spines['left'].set_color(secondary_text)
ax3.grid(alpha=0.2, color=secondary_text)

plt.tight_layout()
comparison_chart = fig3
print("✓ Created Gain vs Weight comparison chart")

print(f"\n📊 Feature Importance Analysis Complete")
print(f"   • {len(feature_importance_df)} features analyzed")
print(f"   • Top feature by gain: {feature_importance_df.iloc[0]['feature']} ({feature_importance_df.iloc[0]['gain']:.2f})")
print(f"   • Top feature by weight: {feature_importance_df.nlargest(1, 'weight').iloc[0]['feature']} ({int(feature_importance_df.nlargest(1, 'weight').iloc[0]['weight'])} splits)")
