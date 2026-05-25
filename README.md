## Stolen Model Detection — Assignment 2

## CMS Assigments Team ID - Team #70
Sarika 
Student ID: 7084305
Ananya
Student ID: 7086628
Leaderboard: team_LX

## How to Reproduce Best Result

Install dependencies:
pip install torch torchvision safetensors pandas numpy requests
Run the script:
python submission.py

## Output

The script generates:
submission.csv

Format:
id,score

* id: model index (0 to 359)
* score: stealing confidence (higher means more likely stolen)

## What the script does

* Loads target model weights
* Downloads 360 suspect models
* Computes weight similarity between target and suspect models
* Uses cosine similarity, exact match ratio, near-exact match ratio, parameter match ratio, and normalized L2 distance
* Combines all metrics into a final score
* Normalizes scores to [0, 1]
* Saves submission.csv

