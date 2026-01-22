import pandas as pd
import json 

#file path, change based on test data
file_name = "evals_test_2.json"

with open(f'/Users/garettwilson/Documents/Coding Folder/CAP6318/Agentic-AI-with-Game-Theory/Performance_Testing/{file_name}', 'r') as file:
    data = json.load(file)

rows = []
for results_name, entries in data.items():
    for entry in entries:
        cases = entry.get("results", {}).get("cases", [])
        
        for case in cases:
            # Extract assertion details (e.g., Contains)
            assertions = case.get("assertions", {})
            assertion_name = list(assertions.keys())[0] if assertions else None
            assertion_data = assertions.get(assertion_name, {}) if assertion_name else {}
            
            row = {
                "results_name": results_name,
                "timestamp": entry["timestamp"],
                "story": entry["story"],
                "case_name": case.get("name"),
                "output": case.get("output"),
                "expected_output": case.get("expected_output"),
                "assertion_type": assertion_name,
                "passed": assertion_data.get("value"),
                "reason": assertion_data.get("reason"),
                "task_duration": case.get("task_duration"),
                "total_duration": case.get("total_duration"),
            }
            rows.append(row)

df = pd.DataFrame(rows)
df

grouped = df.groupby(by = ["case_name","assertion_type"])
grouped = grouped.agg({"passed":"sum","results_name": "count"})
grouped["Pass Rate %"] = round((grouped["passed"] / grouped["results_name"]) *100,4)
print(grouped)