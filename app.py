import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from matplotlib.patches import Rectangle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

from trustworthy_genai import render_trustworthy_recommendations_tab

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="VisiTrail",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B58;
        text-align: center;
        margin-bottom: 2.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Basic screen width, heigit and analysis settings
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
VELOCITY_THRESHOLD = 721  # tested value that works well
MIN_REACTION_TIME = 522
MAX_REACTION_TIME = 5000
AOI_WIDTH = 240
AOI_HEIGHT = 230


def load_and_process_data(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=';')

    # Parse coordinate strings into actual numbers and seprate columns
    df[['EyeTracker_x', 'EyeTracker_y']] = df['EyeTracker'].str.strip("()").str.split(",", expand=True).astype(float)
    df[['obj_x', 'obj_y']] = df['GameObjectPos (Screen Coordinates)'].str.strip("()").str.split(",",
                                                                                                expand=True).astype(
        float)

    # Clean the uselsee or nan data
    initial_row_count = len(df)
    clean_df = df.dropna(subset=['EyeTracker_x', 'EyeTracker_y', 'ObjectName', 'ObjectState', 'Label']).copy()
    clean_df = clean_df.drop(columns=['EyeTracker', 'GameObjectPos (Screen Coordinates)'])
    clean_df = clean_df.reset_index(drop=True)

    final_row_count = len(clean_df)
    rows_removed = initial_row_count - final_row_count

    # Filter impossible coordinates
    tolerance = 50
    valid_coords = (
            (clean_df['EyeTracker_x'] >= -tolerance) &
            (clean_df['EyeTracker_x'] <= SCREEN_WIDTH + tolerance) &
            (clean_df['EyeTracker_y'] >= -tolerance) &
            (clean_df['EyeTracker_y'] <= SCREEN_HEIGHT + tolerance)
    )
    clean_df = clean_df[valid_coords].copy()

    # Create download link for processed data
    csv_buffer = io.StringIO()
    clean_df.to_csv(csv_buffer, sep=';', index=False)
    csv_string = csv_buffer.getvalue()

    st.download_button(
        label=f"Download processed_{uploaded_file.name}",
        data=csv_string,
        file_name=f"processed_{uploaded_file.name}",
        mime='text/csv'
    )

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
            valid_time_clicks = subsequent_clicks[
                subsequent_clicks['Timestamp'] >= appear_time + 522
                ]

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
                    'object_name': obj_name,
                    'object_label': obj_label,
                    'appear_time': appear_time,
                    'click_time': np.nan,
                    'reaction_time': np.nan,
                    'click_type': 'Too Fast Response',
                    'responded': False
                })
        else:
            response_analysis.append({
                'object_name': obj_name,
                'object_label': obj_label,
                'appear_time': appear_time,
                'click_time': np.nan,
                'reaction_time': np.nan,
                'click_type': 'No Response',
                'responded': False
            })

    response_df = pd.DataFrame(response_analysis)

    return timeline_df, response_df


def analyze_click_performance_simple(df):
    """Match targets with clicks properly"""
    if not all(col in df.columns for col in ['ObjectName', 'ObjectState', 'Label']):
        return pd.DataFrame(), {}

    events_df = df.dropna(subset=['ObjectName', 'ObjectState', 'Label']).copy()

    # Get targets and clicks
    target_appears = events_df[
        (events_df['ObjectState'] == 'Appear') &
        (events_df['Label'].str.contains('Target', na=False))
        ].sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])

    target_clicks = events_df[
        (events_df['ObjectName'] == 'Mouse Click') &
        (events_df['ObjectState'] == 'Correct') &
        (events_df['Label'].str.contains('Target', na=False))
        ].sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])

    distractor_appears = events_df[
        (events_df['ObjectState'] == 'Appear') &
        (events_df['Label'].str.contains('Distractor', na=False))
        ].drop_duplicates(subset=['Timestamp'])

    incorrect_clicks = events_df[
        (events_df['ObjectName'] == 'Mouse Click') &
        (events_df['ObjectState'] == 'Incorrect') &
        (events_df['Label'].str.contains('Distractor', na=False))
        ].drop_duplicates(subset=['Timestamp'])

    # Match clicks to targets (1:1 pairing)
    matched_pairs = 0
    used_clicks = set()

    for _, target in target_appears.iterrows():
        target_time = target['Timestamp']

        available_clicks = target_clicks[
            (target_clicks['Timestamp'] > target_time) &
            (~target_clicks['Timestamp'].isin(used_clicks))
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

    # Calculate reaction times
    if matched_pairs > 0:
        reaction_times = []
        used_clicks = set()

        for _, target in target_appears.iterrows():
            target_time = target['Timestamp']

            available_clicks = target_clicks[
                (target_clicks['Timestamp'] > target_time) &
                (target_clicks['Timestamp'] < target_time + 5000) &
                (~target_clicks['Timestamp'].isin(used_clicks))
                ]

            if len(available_clicks) > 0:
                click = available_clicks.iloc[0]
                rt = click['Timestamp'] - target_time
                if 100 <= rt <= 5000:  # reasonable reaction time
                    reaction_times.append(rt)
                used_clicks.add(click['Timestamp'])

        if reaction_times:
            metrics['avg_target_rt'] = np.mean(reaction_times)

    # Build performance dataframe
    performance_data = []
    used_clicks = set()

    for _, target in target_appears.iterrows():
        target_time = target['Timestamp']

        available_clicks = target_clicks[
            (target_clicks['Timestamp'] > target_time) &
            (target_clicks['Timestamp'] < target_time + 5000) &
            (~target_clicks['Timestamp'].isin(used_clicks))
            ]

        if len(available_clicks) > 0:
            click = available_clicks.iloc[0]
            performance_data.append({
                'object_name': target['ObjectName'],
                'object_label': target['Label'],
                'is_target': True,
                'appear_time': target_time,
                'click_time': click['Timestamp'],
                'reaction_time': click['Timestamp'] - target_time,
                'click_type': 'Correct',
                'responded': True
            })
            used_clicks.add(click['Timestamp'])
        else:
            performance_data.append({
                'object_name': target['ObjectName'],
                'object_label': target['Label'],
                'is_target': True,
                'appear_time': target_time,
                'click_time': np.nan,
                'reaction_time': np.nan,
                'click_type': 'No Response',
                'responded': False
            })

    performance_df = pd.DataFrame(performance_data)
    return performance_df, metrics


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
        'x_coverage': x_coverage,
        'y_coverage': y_coverage,
        'total_path_length': total_path_length,
        'screen_utilization': screen_utilization,
        'gaze_density': gaze_density
    }


def create_timeline_subplot(timeline_df, response_df, ax):
    """Helper function to create timeline plot in subplot"""
    start_time = timeline_df['timestamp'].min()
    timeline_df = timeline_df.copy()
    timeline_df['rel_time'] = (timeline_df['timestamp'] - start_time) / 1000.0

    appear_events = timeline_df[timeline_df['event_type'] == 'Object Appear']
    click_events = timeline_df[timeline_df['event_type'] == 'Click']

    colors = {'Target': '#2ecc71', 'Distractor': '#e74c3c', 'Unknown': '#95a5a6', 'Neutral': '#f39c12'}
    click_colors = {'Correct': '#2ecc71', 'Incorrect': '#e74c3c', 'Neutral': '#f39c12'}

    unique_objects = appear_events['object_name'].unique() if len(appear_events) > 0 else []
    y_positions = {obj: i for i, obj in enumerate(unique_objects)}

    # Draw object rectangles
    for _, event in appear_events.iterrows():
        obj_name = event['object_name']
        obj_label = event['object_label']
        color = colors.get(obj_label, '#95a5a6')
        y_pos = y_positions.get(obj_name, 0)
        appear_time = event['rel_time']

        duration = 2.0  # default duration
        rect = Rectangle((appear_time, y_pos - 0.3), duration, 0.6,
                         facecolor=color, alpha=0.7, edgecolor='black', linewidth=1)
        ax.add_patch(rect)

    # Draw click lines
    click_y_max = len(unique_objects) + 0.5
    for _, event in click_events.iterrows():
        click_type = event.get('click_type', 'Unknown')
        color = click_colors.get(click_type, '#95a5a6')
        ax.axvline(x=event['rel_time'], color=color, linestyle='-', linewidth=3, alpha=0.8)

    ax.set_xlim(-0.5, timeline_df['rel_time'].max() + 1)
    ax.set_ylim(-0.5, click_y_max + 1)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Objects')

    if len(unique_objects) > 0:
        ax.set_yticks(range(len(unique_objects)))
        ax.set_yticklabels(unique_objects, fontsize=8)


def create_exact_plots(df, timeline_df, response_df, performance_df, performance_metrics, spatial_metrics, file_name):
    st.subheader(f"Analysis Results for {file_name}")

    duration_seconds = (df['Timestamp'].max() - df['Timestamp'].min()) / 1000.0
    events_df = df.dropna(subset=['ObjectName']).copy()
    click_events_df = events_df[events_df['ObjectName'] == 'Mouse Click'].copy()

    # Plot 1: Object Timeline (most important first)
    if len(timeline_df) > 0:
        st.subheader("1. Object Appearance and Click Analysis Timeline")
        create_object_click_timeline_plot(timeline_df, response_df)

    # Plot 2 & 3: Side by side eye movement stuff
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2. Eye Movement Patterns")

        fig, ax = plt.subplots(figsize=(8, 5))

        fixation_mask = df['movement_type'] == 'fixation'
        saccade_mask = df['movement_type'] == 'saccade'

        # Plot fixations and saccades with different colors
        ax.scatter(df.loc[fixation_mask, 'EyeTracker_x'], df.loc[fixation_mask, 'EyeTracker_y'],
                   c='#2E8B57', alpha=0.6, s=10, label=f'Fixations ({fixation_mask.sum()})', edgecolors='none')
        ax.scatter(df.loc[saccade_mask, 'EyeTracker_x'], df.loc[saccade_mask, 'EyeTracker_y'],
                   c='#DC143C', alpha=0.6, s=15, label=f'Saccades ({saccade_mask.sum()})', edgecolors='none')

        # Add click markers
        click_types = {'Correct': ('gold', '*', 250), 'Incorrect': ('red', 'X', 200), 'Neutral': ('orange', 'o', 150)}
        for click_type, (color, marker, size) in click_types.items():
            mask = click_events_df['ObjectState'] == click_type
            if mask.any():
                ax.scatter(click_events_df.loc[mask, 'EyeTracker_x'], click_events_df.loc[mask, 'EyeTracker_y'],
                           marker=marker, s=size, c=color, edgecolors='black', linewidth=2,
                           label=f'{click_type} Click ({mask.sum()})', zorder=10)

        ax.set_xlim(0, SCREEN_WIDTH)
        ax.set_ylim(0, SCREEN_HEIGHT)
        ax.set_xlabel('X Position (pixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (pixels)', fontsize=12, fontweight='bold')
        ax.set_title('Gaze Trajectory', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)

        # Add some basic stats
        quality_text = f"Data Quality:\n• Duration: {duration_seconds:.1f}s\n• Samples: {len(df)}\n• Screen Use: {spatial_metrics.get('screen_utilization', 0):.1f}%"
        ax.text(0.02, 0.98, quality_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("3. Velocity Profile")

        fig, ax = plt.subplots(figsize=(8, 5))

        time_points = df['Timestamp']
        velocity_values = df['velocity_px_per_sec']

        ax.plot(time_points, velocity_values, color='#1f77b4', linewidth=1.5, alpha=0.8, label='Gaze Velocity')

        fixation_mask = df['movement_type'] == 'fixation'
        saccade_mask = df['movement_type'] == 'saccade'

        ax.fill_between(time_points, 0, velocity_values,
                        where=fixation_mask, color='#51cf66', alpha=0.4,
                        label='Fixation Periods', interpolate=True)
        ax.fill_between(time_points, 0, velocity_values,
                        where=saccade_mask, color='#ff6b6b', alpha=0.4,
                        label='Saccadic Movements', interpolate=True)

        for _, event in click_events_df.iterrows():
            click_type = event['ObjectState']
            color = {'Correct': '#ffd43b', 'Incorrect': '#ff6b6b', 'Neutral': '#ff922b'}.get(click_type, '#ff922b')
            ax.axvline(x=event['Timestamp'], color=color, linestyle='--', alpha=0.8, linewidth=2)

        ax.axhline(y=721.0, color='red', linestyle=':', linewidth=2, alpha=0.7,
                   label='Fixation/Saccade Threshold (721 px/s - Validated)')

        ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Velocity (pixels/second)', fontsize=12, fontweight='bold')
        ax.set_title('Eye Speed Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        stats_text = f'Peak: {velocity_values.max():.0f} px/s\nMean: {velocity_values.mean():.1f} px/s\nFixation Rate: {(fixation_mask.sum() / len(df) * 100):.1f}%'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        st.pyplot(fig)

    # Plot 4: Performance Summary (full width)
    if len(performance_df) > 0:
        st.subheader("4. Performance Summary")
        create_performance_summary_plot(performance_df, performance_metrics, df)


def create_object_click_timeline_plot(timeline_df, response_df):
    if len(timeline_df) == 0:
        st.write("No timeline data available")
        return

    fig, ax = plt.subplots(figsize=(16, 8))

    start_time = timeline_df['timestamp'].min()
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

        disappear_events = timeline_df[
            (timeline_df['event_type'] == 'Object Disappear') &
            (timeline_df['object_name'] == obj_name) &
            (timeline_df['rel_time'] > appear_time)
            ]

        if len(disappear_events) > 0:
            disappear_time = disappear_events.iloc[0]['rel_time']
            duration = disappear_time - appear_time
        else:
            duration = 2.0

        rect = Rectangle((appear_time, y_pos - 0.3), duration, 0.6,
                         facecolor=color, alpha=0.7, edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        duration_text = f"{obj_name}\n({obj_label})\n{duration:.1f}s"
        ax.text(appear_time + duration / 2, y_pos, duration_text,
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    click_y_max = len(unique_objects) + 0.5

    for _, event in click_events.iterrows():
        click_type = event.get('click_type', 'Unknown')
        color = click_colors.get(click_type, '#95a5a6')

        ax.axvline(x=event['rel_time'], color=color, linestyle='-',
                   linewidth=4, alpha=0.8, ymin=0, ymax=1)

        ax.scatter(event['rel_time'], click_y_max, s=200, c=color,
                   marker='*', edgecolors='black', linewidth=2, zorder=10)

        ax.text(event['rel_time'], click_y_max + 0.2, f"{click_type}\nClick",
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

    if len(response_df) > 0:
        for _, resp in response_df.iterrows():
            if resp['responded']:
                appear_time = (resp['appear_time'] - start_time) / 1000.0
                click_time = (resp['click_time'] - start_time) / 1000.0
                obj_name = resp['object_name']
                y_pos = y_positions.get(obj_name, 0)

                line_color = click_colors.get(resp['click_type'], '#95a5a6')
                ax.plot([appear_time + 1, click_time], [y_pos, click_y_max],
                        color=line_color, linestyle='--', linewidth=2, alpha=0.6)

                mid_time = (appear_time + click_time) / 2
                mid_y = (y_pos + click_y_max) / 2
                ax.annotate(f'RT: {resp["reaction_time"]:.0f}ms',
                            xy=(mid_time, mid_y), fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8),
                            ha='center')

    ax.set_xlim(-0.5, timeline_df['rel_time'].max() + 1)
    ax.set_ylim(-0.5, click_y_max + 1)
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Objects', fontsize=12, fontweight='bold')
    ax.set_title('Object Appearance and Click Analysis Timeline', fontsize=14, fontweight='bold')

    if len(unique_objects) > 0:
        ax.set_yticks(range(len(unique_objects)))
        ax.set_yticklabels(unique_objects)

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
    st.pyplot(fig)


def create_performance_summary_plot(performance_df, performance_metrics, df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    targets = performance_df[performance_df['is_target'] == True]

    if len(targets) > 0:
        hit_rate = max(0, performance_metrics['target_hit_rate'])
        miss_rate = max(0, 100 - hit_rate)

        if hit_rate > 0 or miss_rate > 0:
            axes[0, 0].pie([hit_rate, miss_rate], labels=['Hits', 'Misses'],
                           colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title(f'Target Performance\n({len(targets)} targets)')
        else:
            axes[0, 0].text(0.5, 0.5, 'No performance data', ha='center', va='center')
            axes[0, 0].set_title('Target Performance')
    else:
        axes[0, 0].text(0.5, 0.5, 'No targets found', ha='center', va='center')
        axes[0, 0].set_title('Target Performance')

    target_rts = targets[targets['click_type'] == 'Correct']['reaction_time'].dropna()
    if len(target_rts) > 0:
        axes[0, 1].hist(target_rts, bins=min(10, len(target_rts)), alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(target_rts.mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {target_rts.mean():.0f}ms')
        axes[0, 1].set_xlabel('Reaction Time (ms)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Reaction Time Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No reaction time data', ha='center', va='center')
        axes[0, 1].set_title('Reaction Time Distribution')

    # Movement distribution plots
    movement_counts = df['movement_type'].value_counts()
    if len(movement_counts) > 0:
        colors = ['#2E8B57', '#DC143C']

        # Plot 1: Movement distribution pie chart
        axes[1, 0].pie(movement_counts.values, labels=movement_counts.index,
                       colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Gaze Movement Distribution')

        # Plot 2: Movement distribution bar chart
        axes[1, 1].bar(movement_counts.index, movement_counts.values, color=colors, alpha=0.8)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Movement Type Frequency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 0].text(0.5, 0.5, 'No movement data', ha='center', va='center')
        axes[1, 0].set_title('Gaze Movement Distribution')
        axes[1, 1].text(0.5, 0.5, 'No movement data', ha='center', va='center')
        axes[1, 1].set_title('Movement Type Frequency')

    plt.suptitle('Performance Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)


def create_level_comparison_plots(level_results):
    if len(level_results) <= 1:
        return

    st.header("Multi-Level Performance Comparison")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    level_names = list(level_results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(level_names)]

    # --- (a) Success Rate Comparison ---
    hit_rates = [level_results[level]['performance_metrics'].get('target_hit_rate', 0) for level in level_names]
    max_hit_rate = max(hit_rates)
    axes[0, 0].bar(level_names, hit_rates, color=colors, alpha=0.8)
    axes[0, 0].set_ylabel('Target Success Rate (%)')
    axes[0, 0].set_title('(a) Success Rate Comparison')
    axes[0, 0].set_ylim(0, max_hit_rate * 1.2 if max_hit_rate > 0 else 10)
    for i, v in enumerate(hit_rates):
        if v > 0:
            axes[0, 0].text(i, v + max_hit_rate * 0.05, f'{v:.1f}%', ha='center', fontweight='bold')

    # --- (b) Response Time Comparison ---
    reaction_times = [level_results[level]['performance_metrics'].get('avg_target_rt', 0) for level in level_names]
    max_rt = max(reaction_times)
    axes[0, 1].bar(level_names, reaction_times, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Average Response Time (ms)')
    axes[0, 1].set_title('(b) Response Time Comparison')
    axes[0, 1].set_ylim(0, max_rt * 1.2 if max_rt > 0 else 100)
    for i, v in enumerate(reaction_times):
        if v > 0:
            axes[0, 1].text(i, v + max_rt * 0.05, f'{v:.0f}ms', ha='center', fontweight='bold')

    # --- (c) Screen Utilization ---
    screen_utils = [level_results[level]['spatial_metrics'].get('screen_utilization', 0) for level in level_names]
    max_util = max(screen_utils)
    axes[1, 0].bar(level_names, screen_utils, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('Screen Area Used (%)')
    axes[1, 0].set_title('(c) Screen Utilization')
    axes[1, 0].set_ylim(0, max_util * 1.2 if max_util > 0 else 10)
    for i, v in enumerate(screen_utils):
        axes[1, 0].text(i, v + max_util * 0.05, f'{v:.1f}%', ha='center', fontweight='bold')

    # --- (d) False Alarm Rate ---
    false_alarms = [level_results[level]['performance_metrics'].get('false_alarm_rate', 0) for level in level_names]
    max_fa = max(false_alarms)
    axes[1, 1].bar(level_names, false_alarms, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('Mistake Rate (%)')
    axes[1, 1].set_title('(d) False Alarm Rate')
    axes[1, 1].set_ylim(0, max_fa * 1.2 if max_fa > 0 else 10)
    for i, v in enumerate(false_alarms):
        axes[1, 1].text(i, v + max_fa * 0.05, f'{v:.1f}%', ha='center', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # --- Final Touch ---
    plt.suptitle('Performance Comparison Across Game Levels', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

    # --- Summary Table ---
    st.subheader("Performance Summary Table")
    table_data = []
    for level in level_names:
        metrics = level_results[level]
        hit_rate = metrics['performance_metrics'].get('target_hit_rate', 0)
        rt = metrics['performance_metrics'].get('avg_target_rt', 0)
        screen_use = metrics['spatial_metrics'].get('screen_utilization', 0)
        false_alarm = metrics['performance_metrics'].get('false_alarm_rate', 0)

        table_data.append({
            'Level': level,
            'Success Rate (%)': f"{hit_rate:.1f}",
            'Response Time (ms)': f"{rt:.0f}",
            'Screen Use (%)': f"{screen_use:.1f}",
            'Mistakes (%)': f"{false_alarm:.1f}"
        })

    st.table(pd.DataFrame(table_data))


def generate_recommendations(results):
    all_hit_rates = [r['performance_metrics'].get('target_hit_rate', 0) for r in results.values()]
    all_rts = [r['performance_metrics'].get('avg_target_rt', 0) for r in results.values() if
               r['performance_metrics'].get('avg_target_rt', 0) > 0]
    all_screen_usage = [r['spatial_metrics'].get('screen_utilization', 0) for r in results.values()]
    all_fixation_rates = [(r['df']['movement_type'] == 'fixation').mean() * 100 for r in results.values()]

    avg_hit_rate = np.mean(all_hit_rates) if all_hit_rates else 0
    avg_rt = np.mean(all_rts) if all_rts else 0
    avg_screen_usage = np.mean(all_screen_usage) if all_screen_usage else 0
    avg_fixation_rate = np.mean(all_fixation_rates) if all_fixation_rates else 0

    recommendations = []

    if avg_hit_rate < 70:
        recommendations.append({
            'category': 'Teaching Strategy',
            'issue': f'Low success rate ({avg_hit_rate:.1f}%)',
            'recommendation': 'Break down tasks into smaller steps. Use guided practice with immediate feedback. Consider visual attention training exercises.',
            'priority': 'High',
            'audience': 'Teacher'
        })

    if avg_rt > 1500:
        recommendations.append({
            'category': 'Cognitive Processing',
            'issue': f'Slow response time ({avg_rt:.0f}ms)',
            'recommendation': 'Allow additional processing time. Use verbal prompts to guide attention. Reduce time pressure during assessments.',
            'priority': 'High',
            'audience': 'Teacher'
        })

    if avg_screen_usage < 30:
        recommendations.append({
            'category': 'Visual Search Skills',
            'issue': f'Limited exploration ({avg_screen_usage:.1f}%)',
            'recommendation': 'Teach systematic scanning strategies. Use highlighting or pointing to guide visual search. Practice visual tracking exercises.',
            'priority': 'Medium',
            'audience': 'Teacher'
        })

    if avg_fixation_rate < 60:
        recommendations.append({
            'category': 'Attention Regulation',
            'issue': f'High visual activity ({100 - avg_fixation_rate:.1f}% movement)',
            'recommendation': 'Assess for attention difficulties (ADHD). Consider attention training interventions. Evaluate visual processing abilities.',
            'priority': 'High',
            'audience': 'Psychologist'
        })

    if avg_hit_rate < 50:
        recommendations.append({
            'category': 'Visual Processing Assessment',
            'issue': f'Very low performance ({avg_hit_rate:.1f}%)',
            'recommendation': 'Conduct comprehensive visual-perceptual assessment. Screen for visual field defects. Evaluate executive function abilities.',
            'priority': 'High',
            'audience': 'Psychologist'
        })

    if len(results) > 1:
        hit_rates_trend = np.diff(all_hit_rates)
        if np.mean(hit_rates_trend) < -5:
            recommendations.append({
                'category': 'Performance Monitoring',
                'issue': 'Declining performance across sessions',
                'recommendation': 'Monitor for fatigue effects or motivation issues. Consider shorter testing sessions. Evaluate learning differences.',
                'priority': 'Medium',
                'audience': 'Both'
            })

    if not recommendations:
        recommendations.append({
            'category': 'Overall Assessment',
            'issue': 'Performance within normal ranges',
            'recommendation': 'Continue current instructional approaches. Monitor progress over time. Consider enrichment activities for advanced learners.',
            'priority': 'Low',
            'audience': 'Teacher'
        })

    return recommendations


def main():
    st.markdown('<h1 class="main-header">VisiTrail Dashboard</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Upload CSV Files")

        uploaded_files = st.file_uploader(
            "Choose CSV files",
            accept_multiple_files=True,
            type=['csv'],
            help="Upload eye-tracking CSV files (semicolon separated (Currently))"
        )

        st.header("Parameter Settings")
        st.write("**Validated Parameters:**")
        st.write(f"• Velocity Threshold: {VELOCITY_THRESHOLD} px/s")
        st.write(f"• Min Reaction Time: {MIN_REACTION_TIME} ms")
        st.write(f"• Max Reaction Time: {MAX_REACTION_TIME} ms")
        st.write(f"• Screen: {SCREEN_WIDTH}×{SCREEN_HEIGHT} pixels")
        st.write(f"• AOI Size: {AOI_WIDTH}×{AOI_HEIGHT} pixels")

        analyze_button = st.button("Run Analysis", type="primary")

    if not uploaded_files:
        st.info("Please upload CSV files to begin analysis")

        st.subheader("Expected CSV Format")
        sample_data = pd.DataFrame({
            'Timestamp': [1000, 1016, 1033],
            'EyeTracker': ['(910,510)', '(915,515)', '(971,510)'],
            'GameObjectPos (Screen Coordinates)': ['(110,210)', '(110,210)', '(110,210)'],
            'ObjectName': ['Target1', 'Mouse Click', 'Target1'],
            'ObjectState': ['Appear', 'Correct', 'Disappear'],
            'Label': ['Target', 'Target', 'Target']
        })
        st.dataframe(sample_data)
        return

    if analyze_button:
        results = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name.replace('.csv', '')
            status_text.text(f"Processing {file_name}...")

            try:
                df = load_and_process_data(uploaded_file)
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
                    'spatial_metrics': spatial_metrics,
                    'data_points': len(df)
                }

                progress_bar.progress((i + 1) / len(uploaded_files))

            except Exception as e:
                st.error(f"Error processing {file_name}: {str(e)}")
                continue

        status_text.text("Analysis Finished, Data Saved!")
        st.session_state['visitrail_results'] = results

    results = st.session_state.get('visitrail_results')

    if results:
        # st.tabs() has no memory of which tab was selected — any rerun (the Generate
        # button, the Approve/Reject radios inside AI Interpretation, anything) resets
        # it to the first tab. This is a known Streamlit limitation, not a bug in this
        # app: https://github.com/streamlit/streamlit/issues/6146 and widely reported
        # elsewhere. A session-state-backed radio (or selectbox) is the standard
        # workaround — it remembers its value across reruns the way st.tabs cannot.
        # As a side benefit, only the selected section is actually built below, instead
        # of st.tabs's default of rendering every tab's content on every single rerun.
        section_names = list(results.keys())
        if len(results) > 1:
            section_names.extend(["Overall Comparison", "Recommendations", "AI Interpretation"])
        else:
            section_names.extend(["Recommendations", "AI Interpretation"])

        selected_section = st.radio("View", section_names, key="visitrail_selected_section",
                                     horizontal=True, label_visibility="collapsed")
        st.divider()

        if selected_section in results:
            file_name = selected_section
            result = results[file_name]

            create_exact_plots(
                result['df'],
                result['timeline_df'],
                result['response_df'],
                result['performance_df'],
                result['performance_metrics'],
                result['spatial_metrics'],
                file_name
            )

            df = result['df']
            performance_metrics = result['performance_metrics']
            spatial_metrics = result['spatial_metrics']
            response_df = result['response_df']
            performance_df = result['performance_df']

            duration_seconds = (df['Timestamp'].max() - df['Timestamp'].min()) / 1000.0
            fixation_rate = (df['movement_type'] == 'fixation').mean() * 100
            avg_velocity = df['velocity_px_per_sec'].mean()
            peak_velocity = df['velocity_px_per_sec'].max()

            st.subheader(f"Summary Report")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Data Quality:**")
                st.write(f"• Total eye tracking points: {len(df):,}")
                st.write(f"• Session length: {duration_seconds:.1f} seconds")
                st.write(f"• Data collection rate: {len(df) / duration_seconds:.1f} points/sec")
                st.write(f"• Screen area used: {spatial_metrics.get('screen_utilization', 0):.1f}%")

                st.write("**Eye Movement Behavior:**")
                st.write(f"• Time looking steadily: {fixation_rate:.1f}%")
                st.write(f"• Average eye speed: {avg_velocity:.1f} px/s")
                st.write(f"• Fastest eye movement: {peak_velocity:.0f} px/s")

            with col2:
                st.write("**Performance Results:**")
                if performance_metrics['target_hit_rate'] > 0:
                    st.write(f"• Target success rate: {performance_metrics['target_hit_rate']:.1f}%")
                if performance_metrics['avg_target_rt'] > 0:
                    st.write(f"• Average response time: {performance_metrics['avg_target_rt']:.0f} ms")
                st.write(f"• Mistake rate: {performance_metrics['false_alarm_rate']:.1f}%")

                if len(response_df) > 0:
                    st.write("**Response Patterns:**")
                    responded_count = response_df['responded'].sum()
                    st.write(f"• Objects clicked: {responded_count}/{len(response_df)}")

        elif selected_section == "Overall Comparison":
            create_level_comparison_plots(results)

            st.subheader("Analysis Parameters")
            col1, col2 = st.columns(2)
            with col1:
                st.write("• Reaction time window: 522-5000 ms")
                st.write("• Eye movement threshold: 721 px/s")
                st.write("• Data quality tolerance: 50 pixels")
            with col2:
                st.write("• Object area: 240×230 pixels")
                st.write("• Target detection: labels containing 'Target'")
                st.write("• Analysis method: Label-based matching")

        elif selected_section == "Recommendations":
            st.header("Professional Recommendations")

            recommendations = generate_recommendations(results)

            teacher_recs = [r for r in recommendations if r['audience'] in ['Teacher', 'Both']]
            psych_recs = [r for r in recommendations if r['audience'] in ['Psychologist', 'Both']]

            if teacher_recs:
                st.subheader("For Teachers and Educators")
                for rec in teacher_recs:
                    if rec['priority'] == 'High':
                        st.error(f"**{rec['category']}**: {rec['issue']}")
                    elif rec['priority'] == 'Medium':
                        st.warning(f"**{rec['category']}**: {rec['issue']}")
                    else:
                        st.info(f"**{rec['category']}**: {rec['issue']}")
                    st.write(f"**Recommendation**: {rec['recommendation']}")
                    st.write("")

            if psych_recs:
                st.subheader("For Psychologists and Specialists")
                for rec in psych_recs:
                    if rec['priority'] == 'High':
                        st.error(f"**{rec['category']}**: {rec['issue']}")
                    elif rec['priority'] == 'Medium':
                        st.warning(f"**{rec['category']}**: {rec['issue']}")
                    else:
                        st.info(f"**{rec['category']}**: {rec['issue']}")
                    st.write(f"**Recommendation**: {rec['recommendation']}")
                    st.write("")

        elif selected_section == "AI Interpretation":
            st.header("AI-Assisted Interpretation")
            st.caption("Experimental, and separate from the rule-based Recommendations tab on purpose — "
                       "nothing here is generated until you click the button per session, and every claim "
                       "is checked against the same numbers shown above before you see it.")
            for file_name, result in results.items():
                render_trustworthy_recommendations_tab(file_name, result)
                st.divider()


if __name__ == "__main__":
    main()