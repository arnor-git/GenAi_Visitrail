import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from matplotlib.patches import Rectangle
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Basic screen width, height and analysis settings
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
VELOCITY_THRESHOLD = 721
MIN_REACTION_TIME = 522
MAX_REACTION_TIME = 5000
AOI_WIDTH = 240
AOI_HEIGHT = 230


def load_and_process_data(filepath):
    df = pd.read_csv(filepath, sep=';')
    df[['EyeTracker_x', 'EyeTracker_y']] = df['EyeTracker'].str.strip("()").str.split(",", expand=True).astype(float)
    df[['obj_x', 'obj_y']] = df['GameObjectPos (Screen Coordinates)'].str.strip("()").str.split(",",
                                                                                                expand=True).astype(
        float)

    initial_row_count = len(df)
    clean_df = df.dropna(subset=['EyeTracker_x', 'EyeTracker_y', 'ObjectName', 'ObjectState', 'Label']).copy()
    clean_df = clean_df.drop(columns=['EyeTracker', 'GameObjectPos (Screen Coordinates)'])
    clean_df = clean_df.reset_index(drop=True)
    final_row_count = len(clean_df)
    rows_removed = initial_row_count - final_row_count

    tolerance = 50
    valid_coords = (
            (clean_df['EyeTracker_x'] >= -tolerance) &
            (clean_df['EyeTracker_x'] <= SCREEN_WIDTH + tolerance) &
            (clean_df['EyeTracker_y'] >= -tolerance) &
            (clean_df['EyeTracker_y'] <= SCREEN_HEIGHT + tolerance)
    )
    clean_df = clean_df[valid_coords].copy()

    print(f"  Loaded {initial_row_count} rows, kept {len(clean_df)} after cleaning ({rows_removed} removed)")
    return clean_df


def detect_fixations_ivt_pixel(df, velocity_threshold_px=VELOCITY_THRESHOLD):
    time_diff = df['Timestamp'].diff() / 1000.0
    time_diff = time_diff.where(time_diff > 0.001, np.nan)
    velocity_x = df['EyeTracker_x'].diff() / time_diff
    velocity_y = df['EyeTracker_y'].diff() / time_diff
    velocity_magnitude_px = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
    velocity_magnitude_px = velocity_magnitude_px.fillna(0)
    velocity_magnitude_px = velocity_magnitude_px.replace([np.inf, -np.inf], 0)
    df['velocity_px_per_sec'] = velocity_magnitude_px
    df['movement_type'] = np.where(velocity_magnitude_px <= velocity_threshold_px, 'fixation', 'saccade')
    return df


def analyze_object_timeline(df):
    events_df = df.dropna(subset=['ObjectName']).copy()
    if len(events_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    appear_events = events_df[events_df['ObjectState'] == 'Appear'].copy()
    disappear_events = events_df[events_df['ObjectState'] == 'Disappear'].copy()
    click_events = events_df[events_df['ObjectName'] == 'Mouse Click'].copy()

    timeline_data = []
    for _, event in appear_events.iterrows():
        timeline_data.append({
            'timestamp': event['Timestamp'],
            'event_type': 'Object Appear',
            'object_name': event['ObjectName'],
            'object_label': event.get('Label', 'Unknown'),
            'x_pos': event.get('obj_x', np.nan),
            'y_pos': event.get('obj_y', np.nan)
        })

    for _, event in disappear_events.iterrows():
        timeline_data.append({
            'timestamp': event['Timestamp'],
            'event_type': 'Object Disappear',
            'object_name': event['ObjectName'],
            'object_label': event.get('Label', 'Unknown'),
            'x_pos': event.get('obj_x', np.nan),
            'y_pos': event.get('obj_y', np.nan)
        })

    for _, event in click_events.iterrows():
        timeline_data.append({
            'timestamp': event['Timestamp'],
            'event_type': 'Click',
            'object_name': 'Mouse Click',
            'object_label': event.get('Label', 'Unknown'),
            'click_type': event['ObjectState'],
            'x_pos': event['EyeTracker_x'],
            'y_pos': event['EyeTracker_y']
        })

    timeline_df = pd.DataFrame(timeline_data)
    timeline_df = timeline_df.sort_values('timestamp').reset_index(drop=True)

    response_analysis = []
    for _, appear_event in appear_events.iterrows():
        obj_name = appear_event['ObjectName']
        appear_time = appear_event['Timestamp']
        obj_label = appear_event.get('Label', 'Unknown')

        subsequent_clicks = click_events[
            (click_events['Timestamp'] > appear_time) &
            (click_events['Timestamp'] < appear_time + 5000)
            ].copy()

        if len(subsequent_clicks) > 0:
            valid_time_clicks = subsequent_clicks[subsequent_clicks['Timestamp'] >= appear_time + 522]
            if len(valid_time_clicks) > 0:
                first_click = valid_time_clicks.iloc[0]
                reaction_time = first_click['Timestamp'] - appear_time
                response_analysis.append({
                    'object_name': obj_name,
                    'object_label': obj_label,
                    'appear_time': appear_time,
                    'click_time': first_click['Timestamp'],
                    'reaction_time': reaction_time,
                    'click_type': first_click['ObjectState'],
                    'responded': True
                })
            else:
                response_analysis.append({
                    'object_name': obj_name, 'object_label': obj_label, 'appear_time': appear_time,
                    'click_time': np.nan, 'reaction_time': np.nan, 'click_type': 'Too Fast Response', 'responded': False
                })
        else:
            response_analysis.append({
                'object_name': obj_name, 'object_label': obj_label, 'appear_time': appear_time,
                'click_time': np.nan, 'reaction_time': np.nan, 'click_type': 'No Response', 'responded': False
            })

    response_df = pd.DataFrame(response_analysis)
    return timeline_df, response_df


def analyze_click_performance_simple(df):
    if not all(col in df.columns for col in ['ObjectName', 'ObjectState', 'Label']):
        return pd.DataFrame(), {}

    events_df = df.dropna(subset=['ObjectName', 'ObjectState', 'Label']).copy()

    target_appears = events_df[
        (events_df['ObjectState'] == 'Appear') & (events_df['Label'].str.contains('Target', na=False))
        ].sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])

    target_clicks = events_df[
        (events_df['ObjectName'] == 'Mouse Click') & (events_df['ObjectState'] == 'Correct') &
        (events_df['Label'].str.contains('Target', na=False))
        ].sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])

    distractor_appears = events_df[
        (events_df['ObjectState'] == 'Appear') & (events_df['Label'].str.contains('Distractor', na=False))
        ].drop_duplicates(subset=['Timestamp'])

    incorrect_clicks = events_df[
        (events_df['ObjectName'] == 'Mouse Click') & (events_df['ObjectState'] == 'Incorrect') &
        (events_df['Label'].str.contains('Distractor', na=False))
        ].drop_duplicates(subset=['Timestamp'])

    matched_pairs = 0
    used_clicks = set()
    for _, target in target_appears.iterrows():
        target_time = target['Timestamp']
        available_clicks = target_clicks[
            (target_clicks['Timestamp'] > target_time) & (~target_clicks['Timestamp'].isin(used_clicks))
            ]
        if len(available_clicks) > 0:
            matched_pairs += 1
            used_clicks.add(available_clicks.iloc[0]['Timestamp'])

    total_targets = len(target_appears)
    total_distractors = len(distractor_appears)
    false_alarms = len(incorrect_clicks)

    metrics = {
        'target_hit_rate': (matched_pairs / total_targets * 100) if total_targets > 0 else 0,
        'false_alarm_rate': (false_alarms / total_distractors * 100) if total_distractors > 0 else 0,
        'avg_target_rt': 0,
        'avg_spatial_accuracy': 0
    }

    if matched_pairs > 0:
        reaction_times = []
        used_clicks = set()
        for _, target in target_appears.iterrows():
            target_time = target['Timestamp']
            available_clicks = target_clicks[
                (target_clicks['Timestamp'] > target_time) & (target_clicks['Timestamp'] < target_time + 5000) &
                (~target_clicks['Timestamp'].isin(used_clicks))
                ]
            if len(available_clicks) > 0:
                click = available_clicks.iloc[0]
                rt = click['Timestamp'] - target_time
                if 100 <= rt <= 5000:
                    reaction_times.append(rt)
                used_clicks.add(click['Timestamp'])
        if reaction_times:
            metrics['avg_target_rt'] = np.mean(reaction_times)

    performance_data = []
    used_clicks = set()
    for _, target in target_appears.iterrows():
        target_time = target['Timestamp']
        available_clicks = target_clicks[
            (target_clicks['Timestamp'] > target_time) & (target_clicks['Timestamp'] < target_time + 5000) &
            (~target_clicks['Timestamp'].isin(used_clicks))
            ]
        if len(available_clicks) > 0:
            click = available_clicks.iloc[0]
            performance_data.append({
                'object_name': target['ObjectName'], 'object_label': target['Label'], 'is_target': True,
                'appear_time': target_time, 'click_time': click['Timestamp'],
                'reaction_time': click['Timestamp'] - target_time, 'click_type': 'Correct', 'responded': True
            })
            used_clicks.add(click['Timestamp'])
        else:
            performance_data.append({
                'object_name': target['ObjectName'], 'object_label': target['Label'], 'is_target': True,
                'appear_time': target_time, 'click_time': np.nan, 'reaction_time': np.nan,
                'click_type': 'No Response', 'responded': False
            })

    return pd.DataFrame(performance_data), metrics


def calculate_spatial_metrics(df):
    if len(df) == 0:
        return {}
    x_min, x_max = df['EyeTracker_x'].min(), df['EyeTracker_x'].max()
    y_min, y_max = df['EyeTracker_y'].min(), df['EyeTracker_y'].max()
    x_coverage = x_max - x_min
    y_coverage = y_max - y_min
    gaze_distances = np.sqrt(df['EyeTracker_x'].diff() ** 2 + df['EyeTracker_y'].diff() ** 2)
    total_path_length = gaze_distances.sum()
    screen_utilization = (x_coverage * y_coverage) / (SCREEN_WIDTH * SCREEN_HEIGHT) * 100
    gaze_density = total_path_length / (x_coverage * y_coverage) if (x_coverage * y_coverage) > 0 else 0
    return {
        'x_coverage': x_coverage, 'y_coverage': y_coverage, 'total_path_length': total_path_length,
        'screen_utilization': screen_utilization, 'gaze_density': gaze_density
    }


def create_plots(df, timeline_df, response_df, performance_df, performance_metrics, spatial_metrics, file_name,
                 output_dir):
    """Create and save all analysis plots"""
    events_df = df.dropna(subset=['ObjectName']).copy()
    click_events_df = events_df[events_df['ObjectName'] == 'Mouse Click'].copy()
    duration_seconds = (df['Timestamp'].max() - df['Timestamp'].min()) / 1000.0

    # Plot 1: Timeline
    if len(timeline_df) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))
        start_time = timeline_df['timestamp'].min()
        timeline_df = timeline_df.copy()
        timeline_df['rel_time'] = (timeline_df['timestamp'] - start_time) / 1000.0
        appear_events = timeline_df[timeline_df['event_type'] == 'Object Appear']
        click_events = timeline_df[timeline_df['event_type'] == 'Click']
        colors = {'Target': '#2ecc71', 'Distractor': '#e74c3c', 'Unknown': '#95a5a6', 'Neutral': '#f39c12'}
        click_colors = {'Correct': '#2ecc71', 'Incorrect': '#e74c3c', 'Neutral': '#f39c12'}
        unique_objects = appear_events['object_name'].unique() if len(appear_events) > 0 else []
        y_positions = {obj: i for i, obj in enumerate(unique_objects)}

        for _, event in appear_events.iterrows():
            obj_name = event['object_name']
            obj_label = event['object_label']
            color = colors.get(obj_label, '#95a5a6')
            y_pos = y_positions.get(obj_name, 0)
            appear_time = event['rel_time']
            disappear_events_filtered = timeline_df[
                (timeline_df['event_type'] == 'Object Disappear') &
                (timeline_df['object_name'] == obj_name) & (timeline_df['rel_time'] > appear_time)
                ]
            duration = disappear_events_filtered.iloc[0]['rel_time'] - appear_time if len(
                disappear_events_filtered) > 0 else 2.0
            rect = Rectangle((appear_time, y_pos - 0.3), duration, 0.6, facecolor=color, alpha=0.7, edgecolor='black',
                             linewidth=1)
            ax.add_patch(rect)
            # Add duration text label
            duration_text = f"{obj_name}\n({obj_label})\n{duration:.1f}s"
            ax.text(appear_time + duration / 2, y_pos, duration_text,
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        click_y_max = len(unique_objects) + 0.5
        for _, event in click_events.iterrows():
            click_type = event.get('click_type', 'Unknown')
            color = click_colors.get(click_type, '#95a5a6')
            ax.axvline(x=event['rel_time'], color=color, linestyle='-', linewidth=4, alpha=0.8, ymin=0, ymax=1)
            # Add click star marker and label
            ax.scatter(event['rel_time'], click_y_max, s=200, c=color, marker='*', edgecolors='black', linewidth=2,
                       zorder=10)
            ax.text(event['rel_time'], click_y_max + 0.2, f"{click_type}\nClick",
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

        # Draw response lines with RT annotations
        if len(response_df) > 0:
            for _, resp in response_df.iterrows():
                if resp['responded']:
                    appear_time_resp = (resp['appear_time'] - start_time) / 1000.0
                    click_time_resp = (resp['click_time'] - start_time) / 1000.0
                    obj_name = resp['object_name']
                    y_pos = y_positions.get(obj_name, 0)
                    line_color = click_colors.get(resp['click_type'], '#95a5a6')
                    ax.plot([appear_time_resp + 1, click_time_resp], [y_pos, click_y_max],
                            color=line_color, linestyle='--', linewidth=2, alpha=0.6)
                    mid_time = (appear_time_resp + click_time_resp) / 2
                    mid_y = (y_pos + click_y_max) / 2
                    ax.annotate(f'RT: {resp["reaction_time"]:.0f}ms',
                                xy=(mid_time, mid_y), fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8),
                                ha='center')

        ax.set_xlim(-0.5, timeline_df['rel_time'].max() + 1)
        ax.set_ylim(-0.5, click_y_max + 1)
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Objects', fontsize=12, fontweight='bold')
        ax.set_title(f'{file_name} - Object Appearance and Click Analysis Timeline', fontsize=14, fontweight='bold')
        if len(unique_objects) > 0:
            ax.set_yticks(range(len(unique_objects)))
            ax.set_yticklabels(unique_objects)

        # Build legend
        legend_elements = []
        for label, color in colors.items():
            if any(appear_events['object_label'] == label):
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, label=f'{label} Object'))
        for click_type, color in click_colors.items():
            if any(click_events.get('click_type', pd.Series()) == click_type):
                legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=4, label=f'{click_type} Click'))
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{file_name}_timeline.png'), dpi=150)
        plt.close()

    # Plot 2: Eye Movement Patterns
    fig, ax = plt.subplots(figsize=(10, 6))
    fixation_mask = df['movement_type'] == 'fixation'
    saccade_mask = df['movement_type'] == 'saccade'
    ax.scatter(df.loc[fixation_mask, 'EyeTracker_x'], df.loc[fixation_mask, 'EyeTracker_y'],
               c='#2E8B57', alpha=0.6, s=10, label=f'Fixations ({fixation_mask.sum()})', edgecolors='none')
    ax.scatter(df.loc[saccade_mask, 'EyeTracker_x'], df.loc[saccade_mask, 'EyeTracker_y'],
               c='#DC143C', alpha=0.6, s=15, label=f'Saccades ({saccade_mask.sum()})', edgecolors='none')
    click_types = {'Correct': ('gold', '*', 250), 'Incorrect': ('red', 'X', 200), 'Neutral': ('orange', 'o', 150)}
    for click_type, (color, marker, size) in click_types.items():
        mask = click_events_df['ObjectState'] == click_type
        if mask.any():
            ax.scatter(click_events_df.loc[mask, 'EyeTracker_x'], click_events_df.loc[mask, 'EyeTracker_y'],
                       marker=marker, s=size, c=color, edgecolors='black', linewidth=2,
                       label=f'{click_type} Click ({mask.sum()})', zorder=10)
    ax.set_xlim(0, SCREEN_WIDTH)
    ax.set_ylim(0, SCREEN_HEIGHT)
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(f'{file_name} - Eye Movement Patterns')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{file_name}_eye_movements.png'), dpi=150)
    plt.close()

    # Plot 3: Velocity Profile
    fig, ax = plt.subplots(figsize=(10, 5))
    time_points = df['Timestamp']
    velocity_values = df['velocity_px_per_sec']
    ax.plot(time_points, velocity_values, color='#1f77b4', linewidth=1.5, alpha=0.8)
    ax.fill_between(time_points, 0, velocity_values, where=fixation_mask, color='#51cf66', alpha=0.4, label='Fixation',
                    interpolate=True)
    ax.fill_between(time_points, 0, velocity_values, where=saccade_mask, color='#ff6b6b', alpha=0.4, label='Saccade',
                    interpolate=True)
    ax.axhline(y=721.0, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Threshold (721 px/s)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (pixels/second)')
    ax.set_title(f'{file_name} - Velocity Profile')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{file_name}_velocity.png'), dpi=150)
    plt.close()

    # Plot 4: Performance Summary
    if len(performance_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        targets = performance_df[performance_df['is_target'] == True]
        if len(targets) > 0:
            hit_rate = max(0, performance_metrics['target_hit_rate'])
            miss_rate = max(0, 100 - hit_rate)
            if hit_rate > 0 or miss_rate > 0:
                axes[0, 0].pie([hit_rate, miss_rate], labels=['Hits', 'Misses'],
                               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title(f'Target Performance ({len(targets)} targets)')
        target_rts = targets[targets['click_type'] == 'Correct']['reaction_time'].dropna()
        if len(target_rts) > 0:
            axes[0, 1].hist(target_rts, bins=min(10, len(target_rts)), alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].axvline(target_rts.mean(), color='red', linestyle='--', linewidth=2,
                               label=f'Mean: {target_rts.mean():.0f}ms')
            axes[0, 1].set_xlabel('Reaction Time (ms)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Reaction Time Distribution')
            axes[0, 1].legend()
        movement_counts = df['movement_type'].value_counts()
        if len(movement_counts) > 0:
            axes[1, 0].pie(movement_counts.values, labels=movement_counts.index, colors=['#2E8B57', '#DC143C'],
                           autopct='%1.1f%%')
            axes[1, 0].set_title('Gaze Movement Distribution')
            axes[1, 1].bar(movement_counts.index, movement_counts.values, color=['#2E8B57', '#DC143C'], alpha=0.8)
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Movement Type Frequency')
        plt.suptitle(f'{file_name} - Performance Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{file_name}_performance.png'), dpi=150)
        plt.close()


def create_comparison_plots(level_results, output_dir):
    """Create comparison plots for multiple files"""
    if len(level_results) <= 1:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    level_names = list(level_results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(level_names)]

    hit_rates = [level_results[level]['performance_metrics'].get('target_hit_rate', 0) for level in level_names]
    axes[0, 0].bar(level_names, hit_rates, color=colors, alpha=0.8)
    axes[0, 0].set_ylabel('Target Success Rate (%)')
    axes[0, 0].set_title('(a) Success Rate Comparison')
    for i, v in enumerate(hit_rates):
        if v > 0:
            axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

    reaction_times = [level_results[level]['performance_metrics'].get('avg_target_rt', 0) for level in level_names]
    axes[0, 1].bar(level_names, reaction_times, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Average Response Time (ms)')
    axes[0, 1].set_title('(b) Response Time Comparison')
    for i, v in enumerate(reaction_times):
        if v > 0:
            axes[0, 1].text(i, v + 20, f'{v:.0f}ms', ha='center', fontweight='bold')

    screen_utils = [level_results[level]['spatial_metrics'].get('screen_utilization', 0) for level in level_names]
    axes[1, 0].bar(level_names, screen_utils, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('Screen Area Used (%)')
    axes[1, 0].set_title('(c) Screen Utilization')

    false_alarms = [level_results[level]['performance_metrics'].get('false_alarm_rate', 0) for level in level_names]
    axes[1, 1].bar(level_names, false_alarms, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('Mistake Rate (%)')
    axes[1, 1].set_title('(d) False Alarm Rate')

    plt.suptitle('Performance Comparison Across Files', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
    plt.close()


def print_summary(results):
    """Print summary report to console"""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    for file_name, result in results.items():
        df = result['df']
        metrics = result['performance_metrics']
        spatial = result['spatial_metrics']
        duration = (df['Timestamp'].max() - df['Timestamp'].min()) / 1000.0
        fixation_rate = (df['movement_type'] == 'fixation').mean() * 100

        print(f"\n--- {file_name} ---")
        print(f"  Data points: {len(df):,}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Fixation rate: {fixation_rate:.1f}%")
        print(f"  Screen utilization: {spatial.get('screen_utilization', 0):.1f}%")
        print(f"  Target hit rate: {metrics.get('target_hit_rate', 0):.1f}%")
        print(f"  Avg reaction time: {metrics.get('avg_target_rt', 0):.0f}ms")
        print(f"  False alarm rate: {metrics.get('false_alarm_rate', 0):.1f}%")


def main():
    # Hardcoded file paths
    files = [
        'one_student_data/Stimulus_Student8_Activity58.csv',
        'one_student_data/Stimulus_Student8_Activity59.csv',
        'one_student_data/Stimulus_Student8_Activity60.csv',
    ]

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for filepath in files:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        file_name = Path(filepath).stem
        print(f"\nProcessing: {file_name}")

        try:
            df = load_and_process_data(filepath)
            df = detect_fixations_ivt_pixel(df)
            timeline_df, response_df = analyze_object_timeline(df)
            performance_df, performance_metrics = analyze_click_performance_simple(df)
            spatial_metrics = calculate_spatial_metrics(df)

            results[file_name] = {
                'df': df,
                'timeline_df': timeline_df,
                'response_df': response_df,
                'performance_df': performance_df,
                'performance_metrics': performance_metrics,
                'spatial_metrics': spatial_metrics
            }

            # Save processed data
            df.to_csv(os.path.join(output_dir, f'{file_name}_processed.csv'), sep=';', index=False)

            create_plots(df, timeline_df, response_df, performance_df, performance_metrics, spatial_metrics, file_name,
                         output_dir)
            print(f"  Plots saved to {output_dir}/")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    if len(results) > 1:
        create_comparison_plots(results, output_dir)

    print_summary(results)
    print(f"\nOutput saved to: {output_dir}/")


if __name__ == "__main__":
    main()