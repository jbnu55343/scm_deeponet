import subprocess
from pathlib import Path

sumo_exe = r"D:\software_luo\SUMO\bin\sumo.exe"
root_dir = Path.cwd()
scenarios_dir = Path("scenarios")

# 找所有场景配置
configs = sorted(scenarios_dir.glob("S*/sumo.sumocfg"))

for config in configs:
    scenario_name = config.parent.name
    scenario_dir = config.parent
    
    print(f"\n[RUN] {scenario_name}")
    
    # 在场景目录中运行 SUMO
    result = subprocess.run(
        [sumo_exe, "-c", "sumo.sumocfg"],
        cwd=str(scenario_dir),
        capture_output=True,
        text=True,
        timeout=600
    )
    
    if result.returncode == 0:
        print(f"[OK] {scenario_name} completed successfully")
    else:
        print(f"[ERROR] {scenario_name} failed with code {result.returncode}")
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}")

print("\n[DONE] All simulations completed")
