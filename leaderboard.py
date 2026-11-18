import glob
import os
import sys
from subprocess import Popen, PIPE
from tabulate import tabulate

bench_type = sys.argv[1]
group_patterns = sys.argv[2:]
project_root = os.path.dirname(os.path.abspath(__file__))

results = []
group_folders = []
for pattern in group_patterns:
    absolute_pattern = pattern if os.path.isabs(pattern) else os.path.join(project_root, pattern)
    for match in glob.glob(absolute_pattern):
        if os.path.isdir(match):
            group_folders.append(os.path.relpath(match, project_root).replace(os.sep, '/'))

for folder in dict.fromkeys(group_folders):
    file = f"{bench_type}.py"
    file_path = os.path.join(project_root, folder, file)

    if not os.path.exists(file_path):
        continue

    print("Evaluating:", os.path.join(folder, file))

    process = Popen(
        [sys.executable, file_path],
        cwd=project_root,
        stdout=PIPE,
        stderr=PIPE,
    )
    output, err = process.communicate()

    text = output.decode("utf-8", errors="replace")
    stderr_text = err.decode("utf-8", errors="replace")

    lines = [line for line in text.splitlines() if line.startswith("Result:")]

    if len(lines) != 1:
        print("Invalid output format:", text)
        if stderr_text:
            print("Error:", stderr_text)
        results.append((float("inf"), folder))
        continue

    result_line = lines[0]
    print(result_line)

    try:
        value = float(result_line.split()[1])
    except (IndexError, ValueError):
        value = float("inf")

    results.append((value, folder))

results.sort(key=lambda x: x[0])

formatted_results = [
    [idx + 1, folder, str(value)]
    for idx, (value, folder) in enumerate(results)
]

print(tabulate(formatted_results, headers=["Rank", "Group", "Time [ms]"]))