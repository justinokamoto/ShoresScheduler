# Shores Scheduler

I'm now in charge of scheduling volunteers and paid workers for kids ministry within Shores of Grace. After realizing how tedious it was to do (and how well it translated to a classic optimization problem) I decided to make a little weekend project
out of it. Most of this code was written in a few hours, thanks to a very healthy dose of Cline AI. So I take no responsibility for any silliness within the code 😁.

## Summary

This repository contains a complete implementation of a fair personnel scheduling system formulated as an Integer Linear Program (ILP). The system optimizes shift assignments while ensuring fairness, respecting availability constraints, preventing consecutive day assignments, and maintaining gender and language diversity requirements.

## Files Overview

- **`scheduling_ilp_model.py`** - Main implementation of the PersonnelSchedulingILP class
- **`mathematical_model_documentation.md`** - Formal mathematical description of the ILP model
- **`example_database.json`** - Example input data with personnel and events
- **`README.md`** - This file

## Key Features

### 🎯 **Fairness Optimization**
- Minimizes variance in normalized assignment rates across all personnel
- Normalizes by available days to avoid penalizing unavailable personnel
- Uses capacity factors for personalized weighting (e.g., part-time workers)

### 🚫 **Constraint Handling**
- **Availability**: Personnel cannot be scheduled when marked unavailable
- **Coverage**: Exactly two people per shift
- **Gender Diversity**: At least one female per shift
- **Language Requirements**: At least one Portuguese-fluent person per shift
- **Consecutive Days**: Prevents burnout by ensuring no consecutive day assignments

### 📊 **Mathematical Formulation**
- **Objective**: Minimize Σ(z⁺ᵢ + z⁻ᵢ) - sum of absolute deviations from fair share
- **Variables**: Binary assignment variables, assignment counts, and deviation variables
- **Constraints**: Coverage, availability, consecutive days, and fairness constraints

## Quick Start

### Prerequisites
```bash
pip install pulp
```

### Run Program

To add a shift for the volunteers, run:

```
python main.py [YYYY-MM-DD] --save
```

To add a shift for the workers, run:

```
python main.py [YYYY-MM-DD] --database ../ShoresSchedulerData/db_workers.json --save --min-days 14
```

### Basic Library Usage
```python
from scheduling_ilp_model import PersonnelSchedulingILP

# Initialize with your data
scheduler = PersonnelSchedulingILP('example_database.json')

# Build and solve the model
scheduler.build_model()
status = scheduler.solve()

# Get the solution
solution = scheduler.get_solution()
print(f"Status: {solution['status']}")
print(f"Assignments: {solution['assignments']}")
```

### Run Examples
```bash
# Basic example with provided data
python scheduling_ilp_model.py

# Extended examples with analysis
python example_usage.py
```

## Input Data Format

The model expects JSON input with the following structure:

```json
{
  "people": [
    {
      "id": 1,
      "name": "Person Name",
      "male": true,
      "fluent_pt": false,
      "capacity_factor": 1.0,
      "unavailable": [
        {
          "start": "2025-01-01",
          "end": "2025-01-03"
        }
      ]
    }
  ],
  "events": [
    "2025-01-01",
    "2025-02-01",
    "2025-03-01"
  ]
}
```

### Data Fields

- **`people`**: Array of personnel objects
  - **`id`**: Unique identifier
  - **`name`**: Person's name
  - **`male`**: Boolean indicating if person is male (true) or female (false)
  - **`fluent_pt`**: Boolean indicating if person is fluent in Portuguese
  - **`capacity_factor`**: Weighting factor (1.0 = full-time, 0.5 = part-time)
  - **`unavailable`**: Array of unavailable periods with start/end dates
- **`events`**: Array of event dates (ISO format)

## Model Components

### Decision Variables
- **xᵢⱼ**: Binary variable = 1 if person i is assigned to event j
- **yᵢ**: Integer variable = total assignments for person i
- **z⁺ᵢ, z⁻ᵢ**: Continuous variables for positive/negative deviations

### Objective Function
```
Minimize: Σᵢ (z⁺ᵢ + z⁻ᵢ)
```

### Key Constraints
1. **Coverage**: Σᵢ xᵢⱼ = 2 ∀j (exactly two people per shift)
2. **Gender Diversity**: Σᵢ:female xᵢⱼ ≥ 1 ∀j (at least one female per shift)
3. **Language Requirements**: Σᵢ:Portuguese xᵢⱼ ≥ 1 ∀j (at least one Portuguese speaker per shift)
4. **Consecutive Days**: xᵢⱼ₁ + xᵢⱼ₂ ≤ 1 for consecutive events
5. **Assignment Count**: yᵢ = Σⱼ xᵢⱼ ∀i
6. **Fairness**: z⁺ᵢ - z⁻ᵢ = (yᵢ/dᵢ) - fᵢ ∀i

Where:
- **dᵢ**: Available days for person i
- **fᵢ**: Normalized fair share for person i

## Example Output

```
=== Optimal Schedule ===
2025-01-02T00:00:00: Justin Okamoto, Ana Silva (ID: [1, 5])
2025-01-04T00:00:00: Larissa Garcia, Carlos Santos (ID: [3, 4])
2025-01-06T00:00:00: Larissa Garcia, Carlos Santos (ID: [3, 4])
2025-01-08T00:00:00: Thabata Okamoto, Michael Johnson (ID: [2, 6])
2025-01-10T00:00:00: Justin Okamoto, Thabata Okamoto (ID: [1, 2])

=== Assignment Counts ===
Justin Okamoto: 2.0 assignments
Thabata Okamoto: 2.0 assignments
Larissa Garcia: 2.0 assignments
Carlos Santos: 2.0 assignments
Ana Silva: 1.0 assignments
Michael Johnson: 1.0 assignments

=== Fairness Metrics ===
Justin Okamoto:
  - Assignments: 2.0
  - Available Days: 5
  - Normalized Assignment Rate: 0.4000
  - Deviation: +0.0183, -0.0000
```

## Advanced Features

### Capacity Factor Weighting
```python
# Part-time worker gets 50% weighting
{
  "id": 2,
  "name": "Part-time Worker",
  "capacity_factor": 0.5,
  "unavailable": []
}
```

### Complex Availability Patterns
```python
# Multiple unavailable periods
"unavailable": [
  {"start": null, "end": "2025-01-01"},      # Unavailable until Jan 1
  {"start": "2025-01-15", "end": "2025-01-20"}  # Unavailable Jan 15-20
]
```

## Performance Characteristics

- **Time Complexity**: Polynomial for LP relaxation, exponential for ILP (NP-hard)
- **Practical Scale**: Handles dozens of personnel and hundreds of shifts efficiently
- **Solver**: Uses PuLP with CBC/GLPK backend solvers

## Mathematical Properties

1. **Fairness**: L1 norm minimization approximates variance reduction
2. **Optimality**: Guarantees globally optimal solution when feasible
3. **Scalability**: Linear in number of personnel and events for constraints
4. **Flexibility**: Easy to add new constraints or modify objective function

## License

This implementation is provided as-is for educational and practical use in personnel scheduling applications.
