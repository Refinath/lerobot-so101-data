import pandas as pd

pd.set_option("display.max_colwidth", None)

task_instruction_file = "dataset/move-the-black-bowl-from-the-top-of-the-drawer-to-on-top-of-the-cookie/meta/tasks.parquet"

df = pd.read_parquet(task_instruction_file)
print(df.to_string())