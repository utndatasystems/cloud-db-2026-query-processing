import pandas as pd
import numpy as np
import time
from numba import njit


def validate(actual_result):
    expected_result = pd.read_csv('dmv_fuel_type_passengers_expected.csv')
    if not actual_result.equals(expected_result):
        print("EXPECTED:\n===")
        print(expected_result)
        print("===\nACTUAL:\n===")
        print(actual_result)
        print("===")
        return False
    return True


@njit(cache=False)
def aggregate(fuel_codes, passengers):
    counts = np.zeros(9, dtype=np.int64)
    passenger_counts = np.zeros(9, dtype=np.int64)
    passenger_sums = np.zeros(9, dtype=np.float64)

    for i in range(fuel_codes.size):
        code = fuel_codes[i]
        if code < 0:
            code = 8
        counts[code] += 1

        value = passengers[i]
        if not np.isnan(value):
            passenger_counts[code] += 1
            passenger_sums[code] += value

    return counts, passenger_counts, passenger_sums


fuel_names = np.array([
    'COMP N/G', 'DIESEL', 'ELECTRIC', 'FLEX', 'GAS',
    'NONE', 'OTHER', 'PROPANE', ''
])
fuel_codes_by_name = {
    name: code for code, name in enumerate(fuel_names[:-1])
}


def query(df):
    counts, passenger_counts, passenger_sums = aggregate(prepared_fuel_codes, prepared_passengers)

    averages = np.full(9, np.nan, dtype=np.float64)
    for code in range(9):
        if passenger_counts[code]:
            averages[code] = round(passenger_sums[code] / passenger_counts[code], 1)

    names = list(fuel_names[:-1]) + [np.nan]
    return pd.DataFrame({
        'fuel_type': names,
        'vehicle_count': counts,
        'avg_passengers': averages,
    })


# Read data
df = pd.read_csv('dmv_fuel_type_passengers.csv')

# Prepare encoded arrays and warm the JIT outside the measured query.
raw_codes = df['fuel_type'].map(fuel_codes_by_name).to_numpy(dtype=np.float64, copy=False)
prepared_fuel_codes = np.full(len(raw_codes), -1, dtype=np.int8)
nonmissing = ~np.isnan(raw_codes)
prepared_fuel_codes[nonmissing] = raw_codes[nonmissing].astype(np.int8)
prepared_passengers = df['passengers'].to_numpy(dtype=np.float64, copy=False)
aggregate(prepared_fuel_codes[:100], prepared_passengers[:100])

# Run query (data is loaded before, everything else needs to be timed)
start = time.perf_counter()
result = query(df)
end = time.perf_counter()

if validate(result):
    print(f'Result: {(end - start) * 1e3} ms')
else:
    print("Result: Error")