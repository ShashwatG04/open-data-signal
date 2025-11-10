import os
import subprocess
import time

steps = [
    ("🧱 Building dataset", "python data_pipeline.py"),
    ("🧩 Generating features", "python features.py"),
    ("🤖 Training models", "python train.py"),
    ("📊 Running backtest", "python backtest.py"),
    ("🌐 Launching Streamlit app", "streamlit run app.py")
]

def run_command(name, command):
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ {name} failed. Stopping pipeline.")
        exit(result.returncode)
    print(f"✅ {name} completed successfully.")
    time.sleep(1)

if __name__ == "__main__":
    print("🚀 Starting full Open-Data Signals workflow...")
    for step_name, cmd in steps[:-1]:
        run_command(step_name, cmd)
    print("\n🎯 All preprocessing and training done! Launching dashboard...\n")
    run_command("🌐 Streamlit app", steps[-1][1])
