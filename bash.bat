python deforestation_s1_change.py before.tiff after.tiff --out-prefix results/s1_vh --method threshold --decrease-db -1.5
python deforestation_s1_change.py before.tiff after.tiff --out-prefix results/s1_vh --method otsu

python deforestation_s1_change.py before.tiff after.tiff --units linear

python deforestation_s1_change.py before.tiff after.tiff --smooth 3

python deforestation_s1_change.py before.tiff after.tiff --urban-mask urban_mask.tiff
