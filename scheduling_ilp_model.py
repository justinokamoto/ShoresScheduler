"""
Personnel Scheduling Integer Linear Program (ILP)
================================================

This module implements a fair personnel scheduling system as an ILP that:
1. Minimizes variance in assignments (normalized by availability and capacity)
2. Respects availability constraints
3. Enforces minimum days between shifts for the same person (default: 7 days)
4. Ensures exactly two people per shift
5. Ensures at least one female per shift
6. Ensures at least one Portuguese-fluent person per shift
"""

import pulp
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set
import numpy as np

from models import Person, Shift, ScheduleData


class PersonnelSchedulingILP:
    def __init__(self, data_file: str, min_days_between_shifts: int = 7):
        """Initialize the scheduling ILP with data from JSON file.
        
        Args:
            data_file: Path to JSON file containing people and shift data
            min_days_between_shifts: Minimum number of days required between shifts for the same person
        """
        # Use the new data model
        self.data = ScheduleData(data_file)
        
        # Configuration parameters
        self.min_days_between_shifts = min_days_between_shifts
        
        # Create availability matrix using new data model
        self.availability_matrix = self._build_availability_matrix()
        
        # Initialize the optimization problem
        self.prob = None
        self.x_vars = {}  # Decision variables
        self.y_vars = {}  # Assignment count variables
        self.z_plus_vars = {}  # Positive deviation variables for fairness
        self.z_minus_vars = {}  # Negative deviation variables for fairness
        
    def _build_availability_matrix(self) -> Dict[Tuple[int, int], bool]:
        """Build availability matrix using Person objects: (person_id, shift_index) -> is_available"""
        availability = {}
        
        for person_id, person in self.data.people.items():
            for shift_idx, shift in enumerate(self.data.shifts):
                availability[(person_id, shift_idx)] = person.is_available_on_date(shift.date)
        
        return availability
    
    def _calculate_available_days_count(self) -> Dict[int, int]:
        """Calculate number of available days for each person using new data model."""
        available_days = {}
        for person_id in self.data.person_ids:
            count = sum(1 for shift_idx in self.data.shift_indices 
                       if self.availability_matrix.get((person_id, shift_idx), False))
            available_days[person_id] = max(count, 1)  # Avoid division by zero
        return available_days
    
    def _get_capacity_factor(self, person_id: int) -> float:
        """Get capacity factor using Person object."""
        return self.data.get_person(person_id).capacity_factor
    
    def _are_shifts_too_close(self, shift_idx1: int, shift_idx2: int) -> bool:
        """Check if shifts are too close using Shift objects."""
        if shift_idx1 == shift_idx2:
            return False
        shift1 = self.data.get_shift(shift_idx1)
        shift2 = self.data.get_shift(shift_idx2)
        days_apart = abs((shift1.date - shift2.date).days)
        return days_apart < self.min_days_between_shifts
    
    def _is_male(self, person_id: int) -> bool:
        """Check if person is male using Person object."""
        return self.data.get_person(person_id).male
    
    def _is_fluent_pt(self, person_id: int) -> bool:
        """Check if person is fluent in Portuguese using Person object."""
        return self.data.get_person(person_id).fluent_pt
    
    def build_model(self):
        """Build the complete ILP model using new data model."""
        self.prob = pulp.LpProblem("Personnel_Scheduling", pulp.LpMinimize)
        
        # Decision Variables
        # x[i,j] = 1 if person i is assigned to shift j, 0 otherwise
        self.x_vars = {}
        for person_id in self.data.person_ids:
            for shift_idx in self.data.shift_indices:
                if self.availability_matrix.get((person_id, shift_idx), False):
                    self.x_vars[(person_id, shift_idx)] = pulp.LpVariable(
                        f"x_{person_id}_{shift_idx}", cat='Binary'
                    )
        
        # y[i] = total number of assignments for person i
        self.y_vars = {}
        for person_id in self.data.person_ids:
            self.y_vars[person_id] = pulp.LpVariable(f"y_{person_id}", lowBound=0, cat='Integer')
        
        # z+[i] and z-[i] = positive and negative deviations from fair share for person i
        self.z_plus_vars = {}
        self.z_minus_vars = {}
        for person_id in self.data.person_ids:
            self.z_plus_vars[person_id] = pulp.LpVariable(f"z_plus_{person_id}", lowBound=0, cat='Continuous')
            self.z_minus_vars[person_id] = pulp.LpVariable(f"z_minus_{person_id}", lowBound=0, cat='Continuous')
        
        # Add constraints
        self._add_constraints()
        
        # Set objective
        self._set_objective()
    
    def _add_constraints(self):
        """Add all constraints to the model using new data model."""
        
        # Constraint 1: Exactly two people per shift
        for shift_idx in self.data.shift_indices:
            available_persons = [
                self.x_vars[(person_id, shift_idx)] 
                for person_id in self.data.person_ids 
                if (person_id, shift_idx) in self.x_vars
            ]
            if available_persons:
                self.prob += pulp.lpSum(available_persons) == 2, f"TwoPeoplePerShift_{shift_idx}"
        
        # Constraint 2: At least one female per shift
        for shift_idx in self.data.shift_indices:
            female_persons = [
                self.x_vars[(person_id, shift_idx)]
                for person_id in self.data.person_ids
                if (person_id, shift_idx) in self.x_vars and not self._is_male(person_id)
            ]
            if female_persons:
                self.prob += pulp.lpSum(female_persons) >= 1, f"AtLeastOneFemalePerShift_{shift_idx}"
        
        # Constraint 3: At least one Portuguese speaker per shift
        for shift_idx in self.data.shift_indices:
            pt_speakers = [
                self.x_vars[(person_id, shift_idx)]
                for person_id in self.data.person_ids
                if (person_id, shift_idx) in self.x_vars and self._is_fluent_pt(person_id)
            ]
            if pt_speakers:
                self.prob += pulp.lpSum(pt_speakers) >= 1, f"AtLeastOnePtSpeakerPerShift_{shift_idx}"
        
        # Constraint 4: Minimum days between shifts for the same person
        for person_id in self.data.person_ids:
            for shift_idx1 in self.data.shift_indices:
                for shift_idx2 in self.data.shift_indices:
                    if (self._are_shifts_too_close(shift_idx1, shift_idx2) and
                        (person_id, shift_idx1) in self.x_vars and
                        (person_id, shift_idx2) in self.x_vars):
                        self.prob += (
                            self.x_vars[(person_id, shift_idx1)] + 
                            self.x_vars[(person_id, shift_idx2)] <= 1
                        ), f"MinDaysBetweenShifts_{person_id}_{shift_idx1}_{shift_idx2}"
        
        # Constraint 5: Count total assignments for each person
        for person_id in self.data.person_ids:
            assignments = [
                self.x_vars[(person_id, shift_idx)]
                for shift_idx in self.data.shift_indices
                if (person_id, shift_idx) in self.x_vars
            ]
            if assignments:
                self.prob += (
                    self.y_vars[person_id] == pulp.lpSum(assignments)
                ), f"CountAssignments_{person_id}"
            else:
                self.prob += self.y_vars[person_id] == 0, f"CountAssignments_{person_id}"
        
        # Constraint 6: Define fairness deviations
        # Calculate expected fair share for each person (accounting for 2 people per shift)
        available_days = self._calculate_available_days_count()
        total_shifts = len(self.data.shift_indices)
        total_positions = total_shifts * 2  # 2 people per shift
        
        for person_id in self.data.person_ids:
            capacity_factor = self._get_capacity_factor(person_id)
            available_days_count = available_days[person_id]
            
            # Fair share = (total_positions * capacity_factor * available_days) / 
            #              sum(capacity_factor * available_days for all persons)
            total_weighted_availability = sum(
                self._get_capacity_factor(pid) * available_days[pid] 
                for pid in self.data.person_ids
            )
            
            if total_weighted_availability > 0:
                fair_share = (total_positions * capacity_factor * available_days_count) / total_weighted_availability
            else:
                fair_share = 0
            
            # Normalized assignment ratio = actual_assignments / available_days
            # Normalized fair share = fair_share / available_days
            normalized_fair_share = fair_share / available_days_count if available_days_count > 0 else 0
            
            # z+[i] - z-[i] = (y[i] / available_days[i]) - normalized_fair_share
            self.prob += (
                self.z_plus_vars[person_id] - self.z_minus_vars[person_id] ==
                self.y_vars[person_id] / available_days_count - normalized_fair_share
            ), f"FairnessDeviation_{person_id}"
    
    def _set_objective(self):
        """Set the objective function to minimize variance in normalized assignments using new data model."""
        # Minimize sum of absolute deviations (L1 norm approximation of variance)
        objective_terms = []
        for person_id in self.data.person_ids:
            objective_terms.extend([self.z_plus_vars[person_id], self.z_minus_vars[person_id]])
        
        if objective_terms:
            self.prob += pulp.lpSum(objective_terms), "MinimizeFairnessVariance"
    
    def solve(self, solver=None):
        """Solve the ILP model."""
        if self.prob is None:
            self.build_model()
        
        if solver is None:
            # Try different solvers in order of preference
            solvers = [pulp.PULP_CBC_CMD, pulp.GLPK_CMD]
            for solver_class in solvers:
                try:
                    solver = solver_class(msg=0)
                    break
                except:
                    continue
        
        self.prob.solve(solver)
        return pulp.LpStatus[self.prob.status]
    
    def get_solution(self) -> Dict:
        """Get the solution from the solved model using new data model."""
        if self.prob is None or self.prob.status != pulp.LpStatusOptimal:
            return {"status": "No optimal solution found"}
        
        solution = {
            "status": "Optimal",
            "objective_value": pulp.value(self.prob.objective),
            "assignments": {},
            "assignment_counts": {},
            "fairness_metrics": {}
        }
        
        # Extract assignments using Shift objects
        for (person_id, shift_idx), var in self.x_vars.items():
            if pulp.value(var) == 1:
                shift = self.data.get_shift(shift_idx)
                shift_date = shift.date_str
                if shift_date not in solution["assignments"]:
                    solution["assignments"][shift_date] = []
                solution["assignments"][shift_date].append(person_id)
        
        # Extract assignment counts
        for person_id, var in self.y_vars.items():
            solution["assignment_counts"][person_id] = pulp.value(var)
        
        # Extract fairness metrics
        available_days = self._calculate_available_days_count()
        for person_id in self.data.person_ids:
            solution["fairness_metrics"][person_id] = {
                "assignments": pulp.value(self.y_vars[person_id]),
                "available_days": available_days[person_id],
                "normalized_assignments": pulp.value(self.y_vars[person_id]) / available_days[person_id] if available_days[person_id] > 0 else 0,
                "positive_deviation": pulp.value(self.z_plus_vars[person_id]),
                "negative_deviation": pulp.value(self.z_minus_vars[person_id])
            }
        
        return solution
    
    def print_model_summary(self):
        """Print a summary of the model structure using new data model."""
        print("=== Personnel Scheduling ILP Model Summary ===")
        print(f"Number of people: {len(self.data.person_ids)}")
        print(f"Number of shifts: {len(self.data.shift_indices)}")
        print(f"Number of decision variables: {len(self.x_vars)}")
        print(f"Minimum days between shifts: {self.min_days_between_shifts}")
        
        print("\nShift Dates:")
        for shift_idx, shift in enumerate(self.data.shifts):
            print(f"  Shift {shift_idx}: {shift.date_str}")
        
        print("\nDetailed Availability Matrix:")
        # Dynamic column headers based on actual number of shifts
        header = "Person ID"
        for shift_idx in self.data.shift_indices:
            header += f" | Shift {shift_idx}"
        print(header)
        print("-" * len(header))
        
        for person_id in self.data.person_ids:
            availability_row = [f"    {person_id:2d}    "]
            for shift_idx in self.data.shift_indices:
                is_available = self.availability_matrix.get((person_id, shift_idx), False)
                availability_row.append(f"   {'✓' if is_available else '✗'}     ")
            print("|".join(availability_row))
        
        print("\nDecision Variables Created:")
        for (person_id, shift_idx) in sorted(self.x_vars.keys()):
            shift = self.data.get_shift(shift_idx)
            print(f"  x_{person_id}_{shift_idx} (Person {person_id}, {shift.date_str})")
        
        print("\nCapacity Factors:")
        for person_id, person in self.data.people.items():
            print(f"Person {person_id} ({person.name}): {person.capacity_factor}")
        
        available_days = self._calculate_available_days_count()
        print("\nAvailable Days Count:")
        for person_id, count in available_days.items():
            print(f"Person {person_id}: {count} days")


def main():
    """Example usage of the PersonnelSchedulingILP class."""
    # Initialize the model with 1 day minimum between shifts (due to limited example data)
    scheduler = PersonnelSchedulingILP('database.json', min_days_between_shifts=1)
    
    # Build the model first
    scheduler.build_model()
    
    # Print model summary
    scheduler.print_model_summary()
    
    # Solve the model
    print("\n=== Solving Model ===")
    status = scheduler.solve()
    print(f"Solution Status: {status}")
    
    # Get and display solution
    solution = scheduler.get_solution()
    if solution["status"] == "Optimal":
        print(f"\nObjective Value (Total Fairness Deviation): {solution['objective_value']:.4f}")
        
        print("\n=== Optimal Schedule ===")
        for shift_date, assigned_persons in sorted(solution["assignments"].items()):
            person_names = []
            for person_id in assigned_persons:
                person = scheduler.data.people.get(person_id)
                if person:
                    person_names.append(person.name)
            print(f"{shift_date}: {', '.join(person_names)} (ID: {assigned_persons})")
        
        print("\n=== Assignment Counts ===")
        for person_id, count in solution["assignment_counts"].items():
            person = scheduler.data.people.get(person_id)
            person_name = person.name if person else f"Person {person_id}"
            print(f"{person_name}: {count} assignments")
        
        print("\n=== Fairness Metrics ===")
        for person_id, metrics in solution["fairness_metrics"].items():
            person = scheduler.data.people.get(person_id)
            person_name = person.name if person else f"Person {person_id}"
            print(f"{person_name}:")
            print(f"  - Assignments: {metrics['assignments']}")
            print(f"  - Available Days: {metrics['available_days']}")
            print(f"  - Normalized Assignment Rate: {metrics['normalized_assignments']:.4f}")
            print(f"  - Deviation: +{metrics['positive_deviation']:.4f}, -{metrics['negative_deviation']:.4f}")
    else:
        print("No optimal solution found!")
        print(solution)


if __name__ == "__main__":
    main()
