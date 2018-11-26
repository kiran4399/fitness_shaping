\* The Miracle Worker *\
Maximize
Health_restored;_to_be_maximized: 25 Medicine_1_units + 20 Medicine_2_units
Subject To
Herb_A_constraint: 3 Medicine_1_units + 4 Medicine_2_units <= 25
Herb_B_constraint: 2 Medicine_1_units + Medicine_2_units <= 10
Bounds
0 <= Medicine_1_units
0 <= Medicine_2_units
Generals
Medicine_1_units
Medicine_2_units
End
