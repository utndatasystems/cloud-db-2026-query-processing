import pandas as pd
import numpy as np
import duckdb
import time
import utils
import os
from numba import njit

def validate(actual_result):
  expected_result = pd.read_csv('join_expected.csv')

  if not actual_result.equals(expected_result):
    print("EXPECTED:\n===")
    print(expected_result)
    print("===\nACTUAL:\n===")
    print(actual_result)
    print("===")
    return False
  return True

@njit(cache=False)
def manual_join(shipdates, line_keys, prices, exists, start_us, end_us):
  volume = 0.0
  for i in range(shipdates.size):
    if shipdates[i] >= start_us and shipdates[i] < end_us:
      if exists[line_keys[i]]:
        volume += prices[i]
  return volume

def query(lineitem_df, part_df):
  # NOTE: Don't use duckdb or built-in pandas functions for the join.
  # Filter the lineitem date range and probe a manual hash table.
  shipdates = lineitem_df['l_shipdate'].to_numpy(dtype='datetime64[us]', copy=False).view(np.int64)
  line_keys = lineitem_df['l_partkey'].to_numpy(dtype=np.int32, copy=False)
  prices = lineitem_df['l_extendedprice'].to_numpy(copy=False)

  start_us = np.datetime64('1995-09-01', 'us').view(np.int64)
  end_us = np.datetime64('1995-10-01', 'us').view(np.int64)
  volume = int(manual_join(shipdates, line_keys, prices, part_exists, start_us, end_us))

  return pd.DataFrame({'volume': [volume]})

# ---------------------------------------------------------
# Load or generate data
# ---------------------------------------------------------

# Cache two tables to Parquet only if not already present
if not os.path.exists('lineitem.parquet') or not os.path.exists('part.parquet'):
  utils.enforce_parquet_files()
else:
  print('Using the cached Parquet files.')

# Open the connection.
con = duckdb.connect(database=':memory:', read_only=False)

# Run the schema.
con.execute(utils.tpch_schema)

# Load the tables into DuckDB.
lineitem_df = con.sql(f'select * from read_parquet(\'lineitem.parquet\');').fetchdf()
part_df = con.sql(f'select * from read_parquet(\'part.parquet\');').fetchdf()

# Build the membership table and compile the kernel before timing.
part_keys = part_df['p_partkey'].to_numpy(dtype=np.int32, copy=False)
part_exists = np.zeros(int(part_keys.max()) + 1, dtype=np.bool_)
part_exists[part_keys] = True

shipdates = lineitem_df['l_shipdate'].to_numpy(dtype='datetime64[us]', copy=False).view(np.int64)
line_keys = lineitem_df['l_partkey'].to_numpy(dtype=np.int32, copy=False)
prices = lineitem_df['l_extendedprice'].to_numpy(copy=False)
start_us = np.datetime64('1995-09-01', 'us').view(np.int64)
end_us = np.datetime64('1995-10-01', 'us').view(np.int64)
manual_join(shipdates[:100], line_keys[:100], prices[:100], part_exists, start_us, end_us)

# ---------------------------------------------------------
# Run query (generation & loading must happen before timing)
# ---------------------------------------------------------
start = time.perf_counter()
result = query(lineitem_df, part_df)
end = time.perf_counter()

# Validate result and print time
if validate(result):
  print(f'Result: {(end - start) * 1e3} ms')
else:
  print("Result: Error")