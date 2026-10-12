# Zanco 13X packed-bed breakthrough data

Source: S. E. Zanco, M. Ambrosetti, G. Groppi, E. Tronconi and M. Mazzotti,
*Heat transfer intensification with packed open-cell foams in TSA processes for CO2 capture*,
**Chemical Engineering Journal 430 (2022), 131000**.
[DOI: 10.1016/j.cej.2021.131000](https://doi.org/10.1016/j.cej.2021.131000).
The digitised data are distributed under **[Creative Commons Attribution 4.0 International
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**, with attribution to the article's
authors. These numerical derivatives are graph readings prepared for pyDAC, not the authors'
original instrument records. Digitisation and resampling are modifications; this packaging adds
column headers and changes the CSV separator to a comma, without changing any numeric value.

| File | Figure of the cited paper | Quantity and location |
|---|---|---|
| `co2.csv` | Digitised from Figure 8b, packed-bed trace | Outlet CO2 mole fraction, dimensionless |
| `center.csv` | Digitised from Figure 8a, dashed packed-bed trace | Bed temperature at r = 0.0 mm, z = 120 mm |
| `half_radius.csv` | Digitised from Figure 8a, dashed packed-bed trace | Bed temperature at r = 21.0 mm, z = 120 mm |
| `wall.csv` | Digitised from Figure 8a, dashed packed-bed trace | Bed temperature at r = 37.4 mm, z = 120 mm; a bed probe, not metal-wall temperature |

The first column, `time_min`, is elapsed time in minutes from the adsorption-switch origin;
no fitted shift or stretch is applied. Convert it to seconds with `time_s = 60 * time_min`.
`y_CO2` is a mole fraction, not percent; `T_K` is absolute temperature in kelvin.
For plotting in Celsius use `T_C = T_K - 273.15`. CO2 samples span 0.7–60.0 min (594 rows);
each bed-temperature series spans 0–45 min at 0.05 min intervals (901 rows).

Digitisation method and reading uncertainty:

- Temperature axes were calibrated from image gridlines (about 0.137 K/pixel); connected components and a slope-limited tracker separated dashed experimental traces from solid curves.
- The temperature traces were interpolated onto a 0.05 min grid; T(0) = 298.15 K was imposed, and the first approximately 0.4 min interpolate to the first readable pixels.
- Estimated temperature reading bands are ±2 K for t < 2.9 min and ±0.5 K for t > 4.6 min; on the steep ramp, use a time uncertainty of ±0.05 min.
- These estimates describe graph reading, not instrument or solver uncertainty.
- Beyond the last visible temperature pixels (44.98 / 44.43 / 44.13 min for centre / half radius / near wall), the respective values are held constant to 45 min.
- CO2 was also digitised from the graph; its original extraction software and reading uncertainty are not documented. Stored decimals reflect interpolation precision, not measurement accuracy.

Some late temperature readings lie below the nominal 298.15 K jacket temperature. They are
kept as read, not offset or clipped.

Only the four traces compared in the pyDAC paper are included. The source
`n2.csv` is exactly `1 - y_CO2` at identical sample times, so a nitrogen overlay can be derived
without treating it as a second independent measurement. An outlet-temperature trace
was also digitised but is not part of that comparison and is not included.

Load the files directly with NumPy; pyDAC has no experimental-data loader:

```python
import numpy as np

time_min, y_co2 = np.loadtxt("co2.csv", delimiter=",", skiprows=1).T
time_s = 60 * time_min
y_n2 = 1 - y_co2  # derived, not an independent measurement
```
