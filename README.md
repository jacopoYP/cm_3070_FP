# Financial Advisor Bot (Reinforcement Learning + GA)

This project implements a multi-agent Reinforcement Learning trading system
with optional Genetic Algorithm optimisation and a web-based interface.

Developed with Python 3.10.4

## Project Overview

Pipeline flow:

<ol>
  <li>Build technical + sentiment features</li>
  <li>Train Buy and Sell DDQN agents</li>
  <li>Optimise hyperparameters via Genetic Algorithm</li>
  <li>Deploy best models</li>
  <li>Run Web API interface</li>
</ol>

## Environment Setup

### Python Version
Python 3.10.4

Check version: python --version

Create Virtual Environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

## Build Features

Generates:
- data/features.npy
- data/prices.npy
- data/row_meta.parquet


python scripts/build\_features.py \ <br>
  --config config.yaml \ <br>
  --out data <br>

## Train pipeline

python scripts/run\_pipeline.py \ <br>
--config config.yaml \ <br>
--features data/features.npy \ <br>
--prices data/prices.npy \ <br>
--out\_root runs

### Output
runs/pipeline\_YYYYMMDD\_HHMMSS/ <br>
    buy\_agent.pt <br/>
    sell\_agent.pt

## Genetic algorithm for optimization
python scripts/run\_ga.py \ <br>
  --config config.yaml \ <br>
  --features data/features.npy \ <br>
  --prices data/prices.npy \ <br>
  --buy\_model runs/pipeline\_xxx/buy\_agent.pt \ <br>
  --sell\_model runs/pipeline\_xxx/sell\_agent.pt \ <br>
  --out\_root runs\_ga \ <br>
  --seed 42 \ <br>
  --pop 16 \ <br>
  --gens 10 <br>

### Output
runs\_ga/ga\_YYYYMMDD\_HHMMSS/ <br>
    best\_config.yaml

## Deploy models

Copy the best trained models (buy\_agent.pt and sell\_agent.pt) into the <strong>models/</strong> folder

## Web interface

Run it using <strong>uvicorn api.app:app --reload</strong>

## Project structure

* scripts/  <br>
    build\_features.py <br>
    run\_pipeline.py <br>
    run\_ga.py <br>
* api/
* agents/
* core/
* data/
* features/
* ga/
* logs/
* models/
* nlp/
* runs/
* runs_ga/
* sentiment/
* static/
* trade/
* web/

* config.yaml
* requirements.txt