
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


merged_data_path ="all_students_data"
final_fixation_threshold=483.70

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
FIXATION_THRESHOLD = final_fixation_threshold
RT_MIN, RT_MAX = 522, 5000

all_results = []

files = sorted([f for f in os.listdir(merged_data_path) if f.endswith('.csv')])

print(f" Analysis on {len(files)} files...")

for f in files:
    full_path = os.path.join(merged_data_path, f)
    country = "Portugal" if f.lower().startswith('pt_') else "Romania"

    try:
        # Load data
        df = pd.read_csv(full_path, sep=None, engine='python')
        df.columns = [str(col).strip() for col in df.columns]

        # 1. Gaze Patterns
        time_diff = df['Timestamp'].diff() / 1000.0
        time_diff = time_diff.where(time_diff > 0.001, np.nan)
        velocity = np.sqrt(df['EyeTracker-x'].diff() ** 2 + df['EyeTracker-y'].diff() ** 2) / time_diff
        velocity_clean = velocity.fillna(0).replace([np.inf, -np.inf], 0)

        fix_pct = (velocity_clean <= FIXATION_THRESHOLD).mean() * 100

        # 2. Performance Logic
        hit_rate, avg_rt, total_objs = 0, 0, 0
        if 'ObjectState' in df.columns:
            apps = df[df['ObjectState'] == 'Appear']
            clicks = df[df['ObjectName'] == 'Mouse Click']
            total_objs = len(apps)

            rts = []
            for _, app in apps.iterrows():
                valid = clicks[(clicks['Timestamp'] > app['Timestamp'] + RT_MIN) &
                               (clicks['Timestamp'] < app['Timestamp'] + RT_MAX)]
                if not valid.empty:
                    rts.append(valid.iloc[0]['Timestamp'] - app['Timestamp'])

            if total_objs > 0:
                hit_rate = (len(rts) / total_objs) * 100
                avg_rt = np.mean(rts) if rts else 0

        # 3. Spatial Usage
        x_range = df['EyeTracker-x'].max() - df['EyeTracker-x'].min()
        y_range = df['EyeTracker-y'].max() - df['EyeTracker-y'].min()
        usage = (x_range * y_range) / (SCREEN_WIDTH * SCREEN_HEIGHT) * 100

        all_results.append({
            'Filename': f,
            'Country': country,
            'Hits %': round(hit_rate, 1),
            'RT (ms)': round(avg_rt, 0),
            'Fixation %': round(fix_pct, 1),
            'Screen %': round(usage, 1),
            'Objects': f"{len(rts)}/{total_objs}"
        })

    except Exception:
        continue

results_df = pd.DataFrame(all_results)

plt.figure(figsize=(16, 5))

# Plot A: Hit Rate vs Fixation Stability
plt.subplot(1, 2, 1)
sns.scatterplot(data=results_df, x='Fixation %', y='Hits %', hue='Country', style='Country', s=100)
plt.title('Gaze Stability vs. Task Success')
plt.grid(True, alpha=0.2)

# Plot B: Reaction Time Distribution
plt.subplot(1, 2, 2)
sns.kdeplot(data=results_df[results_df['RT (ms)'] > 0], x='RT (ms)', hue='Country', fill=True)
plt.title('Reaction Time Density (Portugal vs Romania)')

plt.tight_layout()
plt.show()

# FULL DATA TABLE (Styled for direct viewing)
print("\n" + "FULL INDIVIDUAL FILE RESULTS".center(90))
print(f"{'Filename':<50} | {'Hits %':<8} | {'RT (ms)':<8} | {'Fix %':<8} | {'Usage %':<8} | {'Objects'}")

for _, row in results_df.iterrows():
    # Highlight low success files in simple text logic
    status = "|" if row['Hits %'] < 50 else " "
    print(
        f"{row['Filename'][:49]:<50} | {row['Hits %']:<8} | {row['RT (ms)']:<8} | {row['Fixation %']:<8} | {row['Screen %']:<8} | {row['Objects']} {status}")

print(f"Summary: Processed {len(results_df)} files.")