# Personnel Scheduling Integer Linear Program (ILP) - Mathematical Formulation

## Problem Description

The personnel scheduling problem aims to assign personnel to shifts while ensuring fairness, respecting availability constraints, and preventing consecutive day assignments. The system requires exactly two people per shift, with at least one female and one Portuguese-fluent person per shift. The primary objective is to minimize the variance in normalized assignment rates across all personnel.

## Mathematical Model

### Sets and Indices

- **P**: Set of personnel (people), indexed by *i*
- **E**: Set of events/shifts, indexed by *j*
- **T**: Set of time periods (days), indexed by *t*

### Parameters

- **a_{i,j}**: Availability indicator = 1 if person *i* is available for event *j*, 0 otherwise
- **c_i**: Capacity factor for person *i* (e.g., 1.0 for full-time, 0.5 for part-time)
- **d_i**: Number of available days for person *i* = ∑_{j∈E} a_{i,j}
- **consecutive_{j1,j2}**: Binary indicator = 1 if events *j1* and *j2* are on consecutive days
- **male_i**: Binary indicator = 1 if person *i* is male, 0 if female
- **pt_i**: Binary indicator = 1 if person *i* is fluent in Portuguese, 0 otherwise
- **n**: Total number of events = |E|

### Decision Variables

- **x_{i,j}**: Binary variable = 1 if person *i* is assigned to event *j*, 0 otherwise
- **y_i**: Integer variable representing total assignments for person *i*
- **z^+_i**: Non-negative continuous variable representing positive deviation from fair share for person *i*
- **z^-_i**: Non-negative continuous variable representing negative deviation from fair share for person *i*

### Objective Function

**Minimize:** ∑_{i∈P} (z^+_i + z^-_i)

The objective minimizes the sum of absolute deviations from fair normalized assignment rates, which approximates variance minimization using the L1 norm.

### Constraints

#### 1. Coverage Constraint (Exactly two people per shift)
For each event *j* ∈ E:
```
∑_{i∈P : a_{i,j}=1} x_{i,j} = 2
```

#### 2. Female Representation Constraint (At least one female per shift)
For each event *j* ∈ E:
```
∑_{i∈P : a_{i,j}=1, male_i=0} x_{i,j} ≥ 1
```

#### 3. Portuguese Fluency Constraint (At least one Portuguese speaker per shift)
For each event *j* ∈ E:
```
∑_{i∈P : a_{i,j}=1, pt_i=1} x_{i,j} ≥ 1
```

#### 4. Availability Constraint (Implicitly enforced)
Personnel can only be assigned to shifts when available:
```
x_{i,j} = 0  ∀ i∈P, j∈E : a_{i,j} = 0
```
This is enforced by only creating decision variables x_{i,j} where a_{i,j} = 1.

#### 5. No Consecutive Days Constraint
For each person *i* ∈ P and all pairs of events *j1*, *j2* ∈ E where consecutive_{j1,j2} = 1:
```
x_{i,j1} + x_{i,j2} ≤ 1
```

#### 6. Assignment Count Constraint
For each person *i* ∈ P:
```
y_i = ∑_{j∈E : a_{i,j}=1} x_{i,j}
```

#### 7. Fairness Deviation Constraint
For each person *i* ∈ P:
```
z^+_i - z^-_i = (y_i / d_i) - f_i
```

Where the normalized fair share f_i is calculated as:
```
f_i = (2n × c_i × d_i) / (∑_{k∈P} c_k × d_k) / d_i
```

Simplifying:
```
f_i = (2n × c_i) / (∑_{k∈P} c_k × d_k)
```

Note: The factor of 2 accounts for having two people per shift, creating 2n total position slots.

### Variable Bounds and Types

- x_{i,j} ∈ {0, 1} ∀ i ∈ P, j ∈ E : a_{i,j} = 1
- y_i ≥ 0, integer ∀ i ∈ P
- z^+_i ≥ 0 ∀ i ∈ P
- z^-_i ≥ 0 ∀ i ∈ P

## Key Features of the Model

### 1. Fairness Normalization
The model normalizes fairness by dividing assignments by available days, ensuring that personnel are not penalized for days they were unavailable.

### 2. Capacity-Weighted Fairness
The capacity factor c_i allows for personalized fairness weights to account for part-time workers or different workload expectations.

### 3. Consecutive Day Prevention
The model prevents burnout by ensuring no person works consecutive days.

### 4. Absolute Deviation Minimization
Using z^+ and z^- variables, the model minimizes the L1 norm of deviations, which provides a good approximation to variance minimization while maintaining linear programming formulation.

## Example Application

Given the updated sample data:
- 6 personnel: Justin (ID=1, male, no Portuguese), Thabata (ID=2, female, Portuguese), Larissa (ID=3, female, Portuguese), Carlos (ID=4, male, Portuguese), Ana (ID=5, female, Portuguese), Michael (ID=6, male, no Portuguese)
- 5 events: 2025-01-02, 2025-01-04, 2025-01-06, 2025-01-08, 2025-01-10
- All have capacity_factor = 1.0 (except Michael with 0.8)

### Personnel Attributes:
- Person 1 (Justin): Male, no Portuguese, available for events 1,2,3,4,5 (5 days)
- Person 2 (Thabata): Female, Portuguese, available for events 1,2,3,4,5 (5 days)
- Person 3 (Larissa): Female, Portuguese, available for events 1,2,3,4,5 (5 days)
- Person 4 (Carlos): Male, Portuguese, available for events 1,2,3,4,5 (5 days)
- Person 5 (Ana): Female, Portuguese, available for events 1,2,4,5 (4 days - unavailable Jan 4-6)
- Person 6 (Michael): Male, no Portuguese, available for events 2,3,4,5 (4 days - unavailable Jan 2)

### Fair Share Calculation:
Total weighted availability = 1.0×5 + 1.0×5 + 1.0×5 + 1.0×5 + 1.0×4 + 0.8×4 = 27.2
Total positions = 5 events × 2 people = 10 positions

Fair share examples:
- Justin: (10 × 1.0 × 5) / 27.2 = 1.84 assignments
- Ana: (10 × 1.0 × 4) / 27.2 = 1.47 assignments
- Michael: (10 × 0.8 × 4) / 27.2 = 1.18 assignments

The model will find an assignment that:
1. Assigns exactly 2 people per shift
2. Ensures at least 1 female per shift
3. Ensures at least 1 Portuguese speaker per shift
4. Minimizes deviations from fair normalized assignment rates
5. Prevents consecutive day assignments

## Computational Complexity

- **Variables**: O(|P| × |E|) binary variables + O(|P|) continuous variables
- **Constraints**: O(|E|) coverage + O(|E|) gender + O(|E|) language + O(|P| × |E|²) consecutive + O(|P|) counting + O(|P|) fairness
- **Problem Class**: Integer Linear Program (NP-hard in general, but practical for moderate sizes)

The model scales reasonably well for typical workforce scheduling problems with dozens of personnel and hundreds of shifts.
