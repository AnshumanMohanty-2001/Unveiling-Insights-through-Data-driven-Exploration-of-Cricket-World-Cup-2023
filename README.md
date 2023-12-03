[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/h_LXMCrc)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=12837995&assignment_repo_type=AssignmentRepo)
# DSCI 510 Final Project

## Name of the Project: Unveiling Insights through Data-driven Exploration of Cricket World

## Team Members (Name and Student IDs): Anshuman Mohanty (4257570790) and Kaustubh Sharma (1765749035)
## Project Structure:
Unveiling Insights through Data-driven Exploration of Cricket World
 ```sh
├── data/
│ ├── processed/
│ └── raw/
├── results/
│ └── Plots/
│   └──analysis_plots/
│   └──maps/
│   └──with_null_value/
│   └──without_null_value/
│ └── final_report.pdf
├── src/
│ ├──utils/
│   ├──utils.py
│ ├──clean_data.py
│ ├──get_data.py
│ ├──run_analysis.py
│ ├──visualize_results.py
├──.gitignore
├──main.py
├──project_proposal
├──README.md
├──requirements.txt
  ```
## Instructions to create a conda environment: 
After navigating to the project directory, we need to type in the code: 
  ```sh
  conda create -n venv
  ```
## Instructions on how to install the required libraries: 
We need to type in code: 
  ```sh
  pip install -r requirements.txt
  ```

## Instructions on how to download the data
We need to type in code: 
  ```sh
  python main.py -get
  ```

## Instructions on how to clean the data
We need to type in code: 
  ```sh
  python main.py -clean
  ```
## Instrucions on how to run analysis code
We need to type in code: 
  ```sh
  python main.py -analyze
  ```

## Instructions on how to create visualizations
We need to type in code: 
  ```sh
  python main.py -visualize
  ```
