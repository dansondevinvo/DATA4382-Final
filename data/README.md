# Data

## Source

**Montgomery County Crash Reporting — Drivers Data**  
https://data.montgomerycountymd.gov/Public-Safety/Crash-Reporting-Drivers-Data/mmzv-x632

This dataset is publicly available through the Montgomery County Open Data portal.

## Download Instructions

1. Visit the link above
2. Click **Export → CSV**
3. Save the file as: `Crash_Reporting_-_Drivers_Data.csv`
4. Place it in this `data/` folder

## Dataset Summary

| Property | Value |
|----------|-------|
| Total Records | 201,147 |
| Original Features | 39 |
| Features Used in Model | 29 |
| Target Variable | Injury Severity (binary: injury vs. no injury) |
| Class Distribution | 82% No Injury / 18% Injury |
| Date Range | 2018 – 2024 |

## Key Features

- `Weather` — Weather conditions at time of crash
- `Surface Condition` — Road surface condition
- `Light` — Lighting conditions
- `Traffic Control` — Type of traffic control device present
- `Collision Type` — Type of collision
- `Speed Limit` — Posted speed limit at crash location
- `Driver Substance Abuse` — Whether substance abuse was involved
- `Driver At Fault` — Whether the driver was determined to be at fault
- `Vehicle Damage Extent` — Severity of vehicle damage (strongest predictor)
- `Vehicle Body Type` — Type of vehicle
- `Latitude / Longitude` — Geographic coordinates of the crash
- `Crash Date/Time` — Used to engineer Hour, DayOfWeek, IsNight, IsWeekend

## Note on Raw Data

The raw CSV is **not included** in this repository due to file size constraints. Please download it directly from the Montgomery County Open Data portal using the link above.
