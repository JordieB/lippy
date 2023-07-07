#Personal/Education/PikesPeak/WQM1019 #WaterQuality/SupplimentalReading
# Resources
[Link](https://blog.orendatech.com/langelier-saturation-index)
[EnergyPurse](https://www.energypurse.com/langelier-saturation-index-lsi-and-its-importance-in-water-chemistry/)

# Notes
- Unbiased measure of water balance. Used by most water treatment plants
## Measures 
- Difference between actual pH and saturation pH
- How saturated water is with calcium carbonate (CaCO$_3$)
- Corrosivity
- Water's ability to dissolve or deposit calcium carbonate

## Saturation
- Perfect saturation is 0.00 LSI
- Acceptable range is $\pm$ 0.30
- < -0.31 - Undersaturated. Will dissolve solid CaCO$_3$. Has a tendency to remove existing calcium carbonate protective coatings in pipelines and equipment.
- > 0.31 - Oversaturated. Will deposit CaCO$_3$, potentially leading to blockages
| LSI        | Description                             |
| ---------- | --------------------------------------- |
| 2 to 0.5   | Scale forming but non corrosive         |
| 0.5 to 0   | Slightly scale forming and corrosive    | 
| 0.0        | Balanced but pitting corrosion possible |
| -0.5 to 0  | Slight corrosion but non-scale forming  |
| -2 to -0.5 | Serious corrosion                       |

## Calculating
$$
\begin{align}
LSI &= pH (measured) - pH_s \\
\\
Where: \\
  pH_s &= (9.3 + A + B) - (C + D) \\
\\
  A &= \frac{Log_{10}(TDS) - 1}{10},\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; \;\;\;\;\;\;\;\;  \text{TDS = Total dissovled solids as (mg/l of} \; CaCO_3 ) \\
  B &= -13.12 \times Log_{10}(^\circ C + 273) + 34.55, \;\;\;\;\;\;\;\; ^{\circ}\text{C = Temperature\;of\;the water in Celsius} \\
  C &= Log_{10}(Ca^{2+}) - 0.4 \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; Ca^{2+} = \text{Calcium Hardness as (mg/l of} \; CaCO_3 ) \\
  D &= Log_{10}(Alkalinity)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; \text{Alkalinity as (mg/l of} \; CaCO_3 )

\end{align}
$$
