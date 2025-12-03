from pathlib import Path

template = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

  <input>
    <net-file value="../../net/shanghai_5km.net.xml"/>
    <route-files value="routes.rou.alt.xml"/>
    <additional-files value="../../scenarios/common.types.xml,edgedump.add.xml"/>
  </input>

  <output>
    <summary-output value="summary.xml"/>
    <tripinfo-output value="tripinfo.xml"/>
    <vehroute-output value="vehroute.xml"/>
    <edgedata-output value="edgedata.xml"/>
    <statistic-output value="statistics.xml"/>
  </output>

  <processing>
    <route-steps value="0"/>
  </processing>

  <report>
    <duration-log.statistics value="true"/>
    <no-step-log value="true"/>
    <log value="sumo.runlog"/>
  </report>

</configuration>
"""

scenarios_dir = Path("scenarios")
for sdir in sorted(scenarios_dir.glob("S*")):
    if not sdir.is_dir():
        continue
    config_path = sdir / "sumo.sumocfg"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(template)
    print(f"Updated: {config_path}")
