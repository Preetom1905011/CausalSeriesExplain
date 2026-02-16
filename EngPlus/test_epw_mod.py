import pandas as pd
from io import StringIO
import os
import numpy as np

from test import run_sim
# Column names in EPW data block order (34 fields)
EPW_COLS = [
    "Year", "Month", "Day", "Hour", "Minute", "DataSource_UncertaintyFlags",
    "DryBulb_C", "DewPoint_C", "RelHumidity_pct", "AtmosPressure_Pa",
    "ExtraterrestrialHorizontalRadiation_Wh_m2",
    "ExtraterrestrialDirectNormalRadiation_Wh_m2",
    "HorizontalIRIntensity_Wh_m2",
    "GlobalHorizontalRadiation_Wh_m2", "DirectNormalRadiation_Wh_m2",
    "DiffuseHorizontalRadiation_Wh_m2",
    "GlobalHorizontalIlluminance_lux", "DirectNormalIlluminance_lux",
    "DiffuseHorizontalIlluminance_lux", "ZenithLuminance_Cd_m2",
    "WindDirection_deg", "WindSpeed_m_s", "TotalSkyCover_tenths",
    "OpaqueSkyCover_tenths", "Visibility_km", "CeilingHeight_m",
    "PresentWeatherObservation", "PresentWeatherCodes",
    "PrecipitableWater_mm", "AerosolOpticalDepth_thousandths",
    "SnowDepth_cm", "DaysSinceLastSnowfall", "Albedo", 
    "LiquidPrecipitationDepth_mm", "LiquidPrecipitationQuantity_hr"
]

# Human-readable header labels (for CSV column header)
CSV_HEADER_LABELS = [
    "Year", "Month", "Day", "Hour", "Minute", "Data Source & Uncertainty Flags",
    "Dry Bulb Temperature (C)", "Dew Point Temperature (C)", "Relative Humidity (%)",
    "Atmospheric Station Pressure (Pa)", "Extraterrestrial Horizontal Radiation (Wh/m2)",
    "Extraterrestrial Direct Normal Radiation (Wh/m2)", "Horizontal IR Intensity (Wh/m2)",
    "Global Horizontal Radiation (Wh/m2)", "Direct Normal Radiation (Wh/m2)",
    "Diffuse Horizontal Radiation (Wh/m2)", "Global Horizontal Illuminance (lux)",
    "Direct Normal Illuminance (lux)", "Diffuse Horizontal Illuminance (lux)",
    "Zenith Luminance (Cd/m2)", "Wind Direction (deg)", "Wind Speed (m/s)",
    "Total Sky Cover (tenths)", "Opaque Sky Cover (tenths)", "Visibility (km)",
    "Ceiling Height (m)", "Present Weather Observation", "Present Weather Codes",
    "Precipitable Water (mm)", "Aerosol Optical Depth (thousandths)",
    "Snow Depth (cm)", "Days Since Last Snowfall", "Albedo",
    "Liquid Precipitation Depth (mm)", "Liquid Precipitation Quantity (hr)"
]


def read_and_modify_epw(epw_path, modifications=None):
    """
    Read EPW file, parse header lines and data table into DataFrame with EPW_COLS.
    Apply modifications dict if provided: {col_name: scalar_or_callable}.
    Returns: (header_lines_list, df)
    """
    if modifications is None:
        modifications = {}

    with open(epw_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    # Heuristic: find first data row (first line that starts with a digit and contains commas)
    data_start = None
    for i, ln in enumerate(lines):
        if ln.strip() and ln[0].isdigit() and ',' in ln:
            data_start = i
            break
    if data_start is None:
        raise ValueError("Could not find data start in EPW file.")

    header_lines = lines[:data_start]
    data_lines = lines[data_start:]

    # Read data into DataFrame
    df = pd.read_csv(StringIO(''.join(data_lines)), header=None, names=EPW_COLS, dtype=str)

    # Convert numeric columns where possible, but keep originals for safe reserializing missing codes
    # We'll create numeric copies for convenient modifications
    num_cols = []
    for c in EPW_COLS:
        # try convert to float where sensible
        try:
            df[c + "_num"] = pd.to_numeric(df[c], errors='coerce')
            num_cols.append(c + "_num")
        except Exception:
            df[c + "_num"] = pd.NA

    # Apply modifications: allow scalar add/replace or callable taking Series (numeric)
    for col, op in modifications.items():
        if col not in EPW_COLS:
            raise KeyError(f"Column '{col}' not in EPW columns")
        num_col = col + "_num"
        if callable(op):
            df[num_col] = op(df[num_col])
        else:
            # interpret scalar as add if numeric, else replace
            if isinstance(op, (int, float)):
                df[num_col] = df[num_col] + op
            else:
                # replace with scalar value for all rows
                df[num_col] = float(op)

    # After modifications, reflect numeric changes back into string columns with reasonable formatting.
    # For integer-like fields, preserve integer formatting when possible.
    def fmt_val(orig_str, num_val):
        # If original missing sentinel (like 9999) and num_val is NaN, return orig_str
        if pd.isna(num_val):
            return orig_str
        # choose formatting: integer if original had no decimal point and is integer-like
        if isinstance(orig_str, str) and '.' not in orig_str:
            # if num_val is integral within tolerance, format without decimals
            if abs(num_val - int(num_val)) < 1e-6:
                return str(int(round(num_val)))
        # otherwise format with 3 significant digits after decimal where appropriate
        # keep 1 decimal for temperatures, 3 for pressure? We'll use general formatter with up to 3 decimals
        return f"{num_val:.3f}".rstrip('0').rstrip('.')

    # Update string columns from numeric columns where numeric is not NaN
    for c in EPW_COLS:import pandas as pd
from io import StringIO
import os

# 34 EPW data block column names (in order)
EPW_COLS = [
    "Year", "Month", "Day", "Hour", "Minute", "DataSource_UncertaintyFlags",
    "DryBulb_C", "DewPoint_C", "RelHumidity_pct", "AtmosPressure_Pa",
    "ExtraterrestrialHorizontalRadiation_Wh_m2",
    "ExtraterrestrialDirectNormalRadiation_Wh_m2",
    "HorizontalIRIntensity_Wh_m2",
    "GlobalHorizontalRadiation_Wh_m2", "DirectNormalRadiation_Wh_m2",
    "DiffuseHorizontalRadiation_Wh_m2",
    "GlobalHorizontalIlluminance_lux", "DirectNormalIlluminance_lux",
    "DiffuseHorizontalIlluminance_lux", "ZenithLuminance_Cd_m2",
    "WindDirection_deg", "WindSpeed_m_s", "TotalSkyCover_tenths",
    "OpaqueSkyCover_tenths", "Visibility_km", "CeilingHeight_m",
    "PresentWeatherObservation", "PresentWeatherCodes",
    "PrecipitableWater_mm", "AerosolOpticalDepth_thousandths",
    "SnowDepth_cm", "DaysSinceLastSnowfall", "Albedo",
    "LiquidPrecipitationDepth_mm", "LiquidPrecipitationQuantity_hr"
]

# Human-friendly CSV headers (same order)
CSV_HEADER_LABELS = [
    "Year", "Month", "Day", "Hour", "Minute", "Data Source & Uncertainty Flags",
    "Dry Bulb Temperature (C)", "Dew Point Temperature (C)", "Relative Humidity (%)",
    "Atmospheric Station Pressure (Pa)", "Extraterrestrial Horizontal Radiation (Wh/m2)",
    "Extraterrestrial Direct Normal Radiation (Wh/m2)", "Horizontal IR Intensity (Wh/m2)",
    "Global Horizontal Radiation (Wh/m2)", "Direct Normal Radiation (Wh/m2)",
    "Diffuse Horizontal Radiation (Wh/m2)", "Global Horizontal Illuminance (lux)",
    "Direct Normal Illuminance (lux)", "Diffuse Horizontal Illuminance (lux)",
    "Zenith Luminance (Cd/m2)", "Wind Direction (deg)", "Wind Speed (m/s)",
    "Total Sky Cover (tenths)", "Opaque Sky Cover (tenths)", "Visibility (km)",
    "Ceiling Height (m)", "Present Weather Observation", "Present Weather Codes",
    "Precipitable Water (mm)", "Aerosol Optical Depth (thousandths)",
    "Snow Depth (cm)", "Days Since Last Snowfall", "Albedo",
    "Liquid Precipitation Depth (mm)", "Liquid Precipitation Quantity (hr)"
]

def read_epw_to_df(epw_path):
    """
    Read EPW file and map data block to DataFrame with EPW_COLS.
    Returns: header_lines (list of strings), df (DataFrame with columns EPW_COLS)
    - Keeps original string field values in df[EPW_COLS].
    - Also creates numeric conversion columns named '<col>_num' for convenient numeric ops.
    """
    with open(epw_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    # find first data row (first line that starts with digit and contains commas)
    data_start = None
    for i, ln in enumerate(lines):
        if ln.strip() and ln[0].isdigit() and ',' in ln:
            data_start = i
            break
    if data_start is None:
        raise ValueError("Could not locate EPW data block start.")

    header_lines = lines[:data_start]
    data_lines = lines[data_start:]

    # Read data block into DataFrame (strings)
    df = pd.read_csv(StringIO(''.join(data_lines)), header=None, names=EPW_COLS, dtype=str)

    print(df.columns, len(df.columns))
    # Add numeric columns where possible for convenience (col + "_num")
    for c in EPW_COLS:
        df[c + "_num"] = pd.to_numeric(df[c], errors='coerce')
    
    print(df.columns, len(df.columns))

    return header_lines, df


def save_df_to_csv_and_epw(df, header_lines, csv_out, epw_out, include_header=False):
    """
    Save DataFrame to:
      - csv_out: CSV with human-readable CSV_HEADER_LABELS (columns from EPW_COLS)
      - epw_out: EPW-style file with data rows from df[EPW_COLS]
    Parameters:
      include_header (bool): if True, write header_lines before data rows; if False, write only data rows.
    """
    # CSV: use EPW_COLS values, map to CSV_HEADER_LABELS
    csv_df = df[EPW_COLS].copy()
    csv_df.columns = CSV_HEADER_LABELS
    csv_df.to_csv(csv_out, index=False)
    print(">>>>", len(csv_df.columns))

    # EPW: write header_lines (optional) then data rows exactly from df[EPW_COLS]
    with open(epw_out, 'w', encoding='utf8') as f:
        if include_header and header_lines:
            for ln in header_lines:
                f.write(ln)
        # write data rows from df[EPW_COLS] using their string representations
        for idx, row in df[EPW_COLS].iterrows():
            # ensure no NaN -> write empty field if NaN
            fields = ["" if pd.isna(val) else str(val) for val in row.values]
            f.write(','.join(fields) + '\n')

    print(f"Saved CSV -> {csv_out}")
    print(f"Saved EPW -> {epw_out} (include_header={include_header})")

def num_to_str(orig_str, num):
    if pd.isna(num):
        return "" if pd.isna(orig_str) else str(orig_str)
    if abs(num - int(round(num))) < 1e-6:
        return str(int(round(num)))
    return f"{num:.3f}".rstrip('0').rstrip('.')

# If the raw signals are not "eventful" enough, synthesize events:
# Example: add a Gaussian peak in the middle and a plateau later
def add_events(signal, peak_center_rel=0.3, peak_amp=5.0, peak_width=5, plateau_start_rel=0.6, plateau_value_increase=10.0):
    n = len(signal)
    i = np.arange(n)
    center = int(n * peak_center_rel)
    # gaussian bump
    bump = peak_amp * np.exp(-0.5 * ((i - center) / peak_width)**2)
    signal = signal + bump
    # plateau
    pstart = int(n * plateau_start_rel)
    signal[pstart:pstart + 10] = signal[pstart:pstart + 10] + plateau_value_increase
    return signal


# --------------------
# Example usage:
# --------------------
if __name__ == "__main__":
    epw_in = "C:/EnergyPlusV25-2-0/WeatherData/USA_VA_Sterling-Washington.Dulles.Intl.AP.724030_TMY3.epw"
    csv_out = "C:/EnergyPlusV25-2-0/TestResults/washington_dulles_modified.csv"
    epw_out = "C:/EnergyPlusV25-2-0/TestResults/washington_dulles_modified.epw"

    header, df = read_epw_to_df(epw_in)

    # Example: modify a column using the numeric column and reflect back into string column
    # (User asked to "update a particular column value" — here's a simple pattern.)
    # e.g., add +1.0 C to Dry Bulb numeric column, then write back
    df["DryBulb_C_num"] = df["DryBulb_C_num"] + 1.0
    # copy numeric formatted back to string column with minimal formatting (preserve integer when integer)
    
    df["DryBulb_C"] = [num_to_str(orig, nv) for orig, nv in zip(df["DryBulb_C"].astype(str), df["DryBulb_C_num"])]

    # Save CSV and EPW (no header lines before data)

    save_df_to_csv_and_epw(df, header, csv_out, epw_out, include_header=True)

