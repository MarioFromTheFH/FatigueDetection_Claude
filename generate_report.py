import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
import os

def generate_mental_fatigue_report_pdf(output_filename="mental_fatigue_report.pdf", data_csv_path=None):
    """
    Generates a PDF report for mental fatigue detection results, including text
    and graphs.

    Args:
        output_filename (str): The name of the PDF file to be created.
        data_csv_path (str, optional): Path to the CSV file containing the data.
                                       If None, simulated data will be used.
    """
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # --- 1. Report Title ---
    story.append(Paragraph("<b>Mental Fatigue Detection Report: Key Insights and Visualizations</b>", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # --- 2. Load or Simulate Data ---
    df = pd.DataFrame() # Initialize an empty DataFrame
    if data_csv_path and os.path.exists(data_csv_path):
        try:
            # Load data from CSV, assuming no header in the provided format
            df = pd.read_csv(data_csv_path, header=None, names=[
                'Recording Time', 'Amount of Recorded Faces', 'Average Eye Openness',
                'Average Mental Fatigue Level', 'Amount of Glowing Objects',
                'Average Total Blinking Rate', 'Dominant Emotion', 'Head Pose', 'Frame Number'
            ])
            # Convert relevant columns to numeric, coercing errors to NaN
            df['Recording Time'] = pd.to_numeric(df['Recording Time'], errors='coerce')
            df['Amount of Recorded Faces'] = pd.to_numeric(df['Amount of Recorded Faces'], errors='coerce')
            df['Average Eye Openness'] = pd.to_numeric(df['Average Eye Openness'], errors='coerce')
            df['Average Mental Fatigue Level'] = pd.to_numeric(df['Average Mental Fatigue Level'], errors='coerce')
            df['Amount of Glowing Objects'] = pd.to_numeric(df['Amount of Glowing Objects'], errors='coerce')
            df['Average Total Blinking Rate'] = pd.to_numeric(df['Average Total Blinking Rate'], errors='coerce')

            # Drop rows with NaN values in critical columns for plotting
            df.dropna(subset=[
                'Recording Time', 'Amount of Recorded Faces', 'Average Eye Openness',
                'Average Mental Fatigue Level', 'Amount of Glowing Objects',
                'Average Total Blinking Rate'
            ], inplace=True)

            # Sort by Recording Time to ensure correct line plots
            df.sort_values(by='Recording Time', inplace=True)

            print(f"Loaded data from {data_csv_path}. Shape: {df.shape}")
            if df.empty:
                print("Warning: Loaded CSV is empty after cleaning. Using simulated data.")
                df = simulate_data() # Fallback to simulated data if CSV is empty
        except Exception as e:
            print(f"Error loading CSV from {data_csv_path}: {e}. Using simulated data instead.")
            df = simulate_data()
    else:
        print("No valid CSV path provided or file not found. Using simulated data.")
        df = simulate_data()

    # Ensure 'Dominant Emotion' and 'Head Pose' columns exist for plotting
    if 'Dominant Emotion' not in df.columns:
        df['Dominant Emotion'] = 'Unknown' # Default if not present
    if 'Head Pose' not in df.columns:
        df['Head Pose'] = 'Unknown' # Default if not present


    # --- 3. Core Mental Fatigue Metrics Section ---
    story.append(Paragraph("<b>1. Core Mental Fatigue Metrics</b>", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))

    # Graph 1: Average Mental Fatigue Level over Recording Time
    if not df.empty and 'Average Mental Fatigue Level' in df.columns and 'Recording Time' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['Recording Time'], df['Average Mental Fatigue Level'], marker='o', linestyle='-')
        plt.title('Average Mental Fatigue Level Over Time')
        plt.xlabel('Recording Time')
        plt.ylabel('Average Mental Fatigue Level')
        plt.grid(True)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img = Image(buf, width=7 * inch, height=3.5 * inch)
        story.append(img)
        story.append(Spacer(1, 0.1 * inch))

    # --- 4. Physiological Indicators of Fatigue Section ---
    story.append(Paragraph("<b>2. Physiological Indicators of Fatigue</b>", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))

    # Graph 2: Average Eye Openness vs. Average Mental Fatigue Level (Scatter)
    if not df.empty and 'Average Mental Fatigue Level' in df.columns and 'Average Eye Openness' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.scatter(df['Average Mental Fatigue Level'], df['Average Eye Openness'], alpha=0.7)
        plt.title('Average Eye Openness vs. Average Mental Fatigue Level')
        plt.xlabel('Average Mental Fatigue Level')
        plt.ylabel('Average Eye Openness')
        plt.grid(True)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img = Image(buf, width=7 * inch, height=3.5 * inch)
        story.append(img)
        story.append(Spacer(1, 0.1 * inch))

    # Graph 3: Average Total Blinking Rate Over Time
    if not df.empty and 'Average Total Blinking Rate' in df.columns and 'Recording Time' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['Recording Time'], df['Average Total Blinking Rate'], marker='o', linestyle='-', color='orange')
        plt.title('Average Total Blinking Rate Over Time')
        plt.xlabel('Recording Time')
        plt.ylabel('Average Total Blinking Rate')
        plt.grid(True)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img = Image(buf, width=7 * inch, height=3.5 * inch)
        story.append(img)
        story.append(Spacer(1, 0.1 * inch))

    # --- 5. Behavioral and Emotional Insights Section ---
    story.append(Paragraph("<b>3. Behavioral and Emotional Insights</b>", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))

    # Graph 4: Dominant Emotion Distribution (Bar Chart)
    if not df.empty and 'Dominant Emotion' in df.columns and len(df['Dominant Emotion'].unique()) > 1:
        emotion_counts = df['Dominant Emotion'].value_counts()
        plt.figure(figsize=(10, 5))
        emotion_counts.plot(kind='bar', color='skyblue')
        plt.title('Dominant Emotion Distribution')
        plt.xlabel('Dominant Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img = Image(buf, width=7 * inch, height=3.5 * inch)
        story.append(img)
        story.append(Spacer(1, 0.1 * inch))

    # Graph 5: Head Pose Distribution (Bar Chart)
    if not df.empty and 'Head Pose' in df.columns and len(df['Head Pose'].unique()) > 1:
        head_pose_counts = df['Head Pose'].value_counts()
        plt.figure(figsize=(10, 5))
        head_pose_counts.plot(kind='bar', color='lightgreen')
        plt.title('Head Pose Distribution')
        plt.xlabel('Head Pose')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img = Image(buf, width=7 * inch, height=3.5 * inch)
        story.append(img)
        story.append(Spacer(1, 0.1 * inch))

    # --- 6. Crowd Dynamics and Environmental Factors Section ---
    story.append(Paragraph("<b>4. Crowd Dynamics and Environmental Factors</b>", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))

    # Graph 6: Amount of Recorded Faces Over Time (Crowd Density)
    if not df.empty and 'Amount of Recorded Faces' in df.columns and 'Recording Time' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['Recording Time'], df['Amount of Recorded Faces'], marker='o', linestyle='-', color='purple')
        plt.title('Amount of Recorded Faces Over Time (Crowd Density)')
        plt.xlabel('Recording Time')
        plt.ylabel('Amount of Recorded Faces')
        plt.grid(True)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img = Image(buf, width=7 * inch, height=3.5 * inch)
        story.append(img)
        story.append(Spacer(1, 0.1 * inch))

    # Graph 7: Amount of Glowing Objects Over Time
    if not df.empty and 'Amount of Glowing Objects' in df.columns and 'Recording Time' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['Recording Time'], df['Amount of Glowing Objects'], marker='o', linestyle='-', color='red')
        plt.title('Amount of Glowing Objects Over Time')
        plt.xlabel('Recording Time')
        plt.ylabel('Amount of Glowing Objects')
        plt.grid(True)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img = Image(buf, width=7 * inch, height=3.5 * inch)
        story.append(img)
        story.append(Spacer(1, 0.1 * inch))


    # --- 7. Temporal Analysis and Trends Section (No new graphs, covered by others) ---
    # This section is more about analysis approach rather than direct data points for new graphs.
    # The time-series plots (Fatigue, Blinking Rate, Recorded Faces, Glowing Objects) already cover this.
    story.append(Paragraph("<b>5. Temporal Analysis and Trends</b>", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))


    # --- 8. Conclusion ---
    story.append(Paragraph("<b>Conclusion</b>", styles['h2']))
    story.append(Spacer(1, 0.2 * inch))

    # Build the PDF
    try:
        doc.build(story)
        print(f"PDF report '{output_filename}' generated successfully!")
    except Exception as e:
        print(f"Error building PDF: {e}")

def simulate_data():
    """
    Simulates sample data for demonstration purposes.
    """
    data = {
        'Recording Time': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        'Amount of Recorded Faces': [10, 12, 11, 15, 14, 13, 16, 17, 18, 20, 19, 22, 21, 25, 24, 23, 26, 27, 28, 30],
        'Average Eye Openness': [0.9, 0.88, 0.85, 0.8, 0.75, 0.7, 0.68, 0.65, 0.6, 0.55, 0.5, 0.48, 0.45, 0.4, 0.38, 0.35, 0.32, 0.3, 0.28, 0.25],
        'Average Mental Fatigue Level': [5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5, 30.0, 31.5, 33.0],
        'Amount of Glowing Objects': [0, 0, 1, 0, 0, 2, 1, 0, 0, 3, 2, 0, 0, 4, 3, 0, 0, 5, 4, 0],
        'Average Total Blinking Rate': [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0],
        'Dominant Emotion': ['Neutral', 'Neutral', 'Happy', 'Neutral', 'Sad', 'Neutral', 'Happy', 'Neutral', 'Unknown', 'Neutral', 'Sad', 'Neutral', 'Happy', 'Neutral', 'Sad', 'Neutral', 'Unknown', 'Neutral', 'Happy', 'Neutral'],
        'Head Pose': ['Forward', 'Forward', 'Looking Up', 'Forward', 'Looking Down', 'Forward', 'Looking Up', 'Forward', 'Looking Up', 'Forward', 'Looking Down', 'Forward', 'Looking Up', 'Forward', 'Looking Down', 'Forward', 'Looking Up', 'Forward', 'Looking Up', 'Forward'],
        'Frame Number': range(20)
    }
    return pd.DataFrame(data)

# --- How to use the script ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("No argument provided. Running with simulated Data")
        generate_mental_fatigue_report_pdf()
    else:
        csvlogfile=sys.argv[1]
        name_without_suffix = os.path.splitext(csvlogfile)[0]
        generate_mental_fatigue_report_pdf(output_filename=name_without_suffix+".pdf", data_csv_path=csvlogfile)

