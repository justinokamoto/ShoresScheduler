"""
Incremental Personnel Scheduling Integer Linear Program (ILP)
============================================================

This module implements an incremental personnel scheduling system as an ILP that:
1. Takes an existing schedule as fixed constraints
2. Optimally assigns personnel to ONE new shift
3. Minimizes variance in total assignments (existing + new)
4. Respects availability constraints
5. Enforces minimum days between shifts for the same person (default: 7 days)
6. Ensures the specified number of people per new shift (configurable, default: 2)
7. Ensures at least one female per new shift
8. Ensures at least one Portuguese-fluent person per new shift
"""

import pulp
from datetime import date, timedelta
from typing import Dict, List, Tuple, Set
import numpy as np

from models import Person, Shift, ScheduleData


class IncrementalPersonnelScheduler:
    def __init__(self, data_file: str, new_shift_date: str, min_days_between_shifts: int = 7, people_needed: int = 2):
        """Initialize the incremental scheduling ILP with data from JSON file.
        
        Args:
            data_file: Path to JSON file containing people and existing shift data
            new_shift_date: ISO date string for the new shift to be scheduled (e.g., "2025-01-12")
            min_days_between_shifts: Minimum number of days required between shifts for the same person
            people_needed: Number of people needed for the shift
        """
        # Use the existing data model
        self.data = ScheduleData(data_file)
        
        # The new shift we're scheduling for
        self.new_shift_date = date.fromisoformat(new_shift_date)
        self.new_shift = Shift(date=self.new_shift_date)
        
        # Configuration parameters
        self.min_days_between_shifts = min_days_between_shifts
        self.people_needed = people_needed
        
        # Parse existing assignments to understand current state
        self.existing_assignment_counts = self._calculate_existing_assignment_counts()
        
        # Initialize the optimization problem
        self.prob = None
        self.x_vars = {}  # Decision variables (only for new shift)
        self.z_plus_vars = {}  # Positive deviation variables for fairness
        self.z_minus_vars = {}  # Negative deviation variables for fairness
        
    def _calculate_existing_assignment_counts(self) -> Dict[int, int]:
        """Calculate how many shifts each person is already assigned to."""
        assignment_counts = {person_id: 0 for person_id in self.data.person_ids}
        
        for shift_assignments in self.data.existing_assignments:
            for person_id in shift_assignments:
                if person_id in assignment_counts:
                    assignment_counts[person_id] += 1
        
        return assignment_counts
    
    def _is_person_available_for_new_shift(self, person_id: int) -> bool:
        """Check if person is available for the new shift date."""
        person = self.data.get_person(person_id)
        return person.is_available_on_date(self.new_shift_date)
    
    def _violates_minimum_days_constraint(self, person_id: int) -> bool:
        """Check if assigning person to new shift would violate minimum days constraint."""
        for shift_idx, shift_assignments in enumerate(self.data.existing_assignments):
            if person_id in shift_assignments:
                existing_shift = self.data.get_shift(shift_idx)
                days_apart = abs((self.new_shift_date - existing_shift.date).days)
                if days_apart < self.min_days_between_shifts:
                    return True
        return False
        
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
            
    def build_model(self):
        """Build the incremental ILP model for the new shift only."""
        self.prob = pulp.LpProblem("Incremental_Personnel_Scheduling", pulp.LpMinimize)
        
        # Decision Variables - ONLY for the new shift
        # x[i] = 1 if person i is assigned to the new shift, 0 otherwise
        self.x_vars = {}
        for person_id in self.data.person_ids:
            # Check if person is available and doesn't violate minimum days constraint
            if (self._is_person_available_for_new_shift(person_id) and 
                not self._violates_minimum_days_constraint(person_id)):
                self.x_vars[person_id] = pulp.LpVariable(
                    f"x_{person_id}_new_shift", cat='Binary'
                )
        
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
        """Add constraints for the incremental scheduling of the new shift."""
        
        # Constraint 1: Required number of people for the new shift
        available_persons = [
            self.x_vars[person_id] 
            for person_id in self.data.person_ids 
            if person_id in self.x_vars
        ]
        if available_persons:
            self.prob += pulp.lpSum(available_persons) == self.people_needed, "RequiredPeoplePerNewShift"
        
        # Constraint 2: At least one female for the new shift
        female_persons = [
            self.x_vars[person_id]
            for person_id in self.data.person_ids
            if person_id in self.x_vars and not self.data.get_person(person_id).male
        ]
        if female_persons:
            self.prob += pulp.lpSum(female_persons) >= 1, "AtLeastOneFemalePerNewShift"
        
        # Constraint 3: At least one Portuguese speaker for the new shift
        pt_speakers = [
            self.x_vars[person_id]
            for person_id in self.data.person_ids
            if person_id in self.x_vars and self.data.get_person(person_id).fluent_pt
        ]
        if pt_speakers:
            self.prob += pulp.lpSum(pt_speakers) >= 1, "AtLeastOnePtSpeakerPerNewShift"
        
        # Constraint 4: Define fairness deviations based on total assignments (existing + new)
        # Calculate expected fair share including the new shift
        total_existing_shifts = len(self.data.shifts)
        total_shifts_including_new = total_existing_shifts + 1
        # Calculate total positions assuming existing shifts had 2 people, new shift has people_needed
        total_positions = total_existing_shifts * 2 + self.people_needed
        
        # Calculate total weighted availability for fair share calculation
        total_weighted_availability = 0
        for person_id in self.data.person_ids:
            capacity_factor = self.data.get_person(person_id).capacity_factor
            # For fairness calculation, consider if person is available for new shift
            is_available_for_new = (self._is_person_available_for_new_shift(person_id) and 
                                   not self._violates_minimum_days_constraint(person_id))
            availability_weight = (total_existing_shifts + (1 if is_available_for_new else 0))
            total_weighted_availability += capacity_factor * availability_weight
        
        for person_id in self.data.person_ids:
            capacity_factor = self.data.get_person(person_id).capacity_factor
            
            # Calculate person's availability weight
            is_available_for_new = (self._is_person_available_for_new_shift(person_id) and 
                                   not self._violates_minimum_days_constraint(person_id))
            availability_weight = (total_existing_shifts + (1 if is_available_for_new else 0))
            
            # Fair share calculation
            if total_weighted_availability > 0:
                fair_share = (total_positions * capacity_factor * availability_weight) / total_weighted_availability
            else:
                fair_share = 0
            
            # Normalized fair share
            normalized_fair_share = fair_share / availability_weight if availability_weight > 0 else 0
            
            # Total assignments = existing assignments + new assignment (if any)
            existing_assignments = self.existing_assignment_counts[person_id]
            new_assignment = self.x_vars.get(person_id, 0)  # 0 if person not available for new shift
            total_assignments = existing_assignments + new_assignment
            
            # Normalized total assignments
            normalized_total_assignments = total_assignments / availability_weight if availability_weight > 0 else 0
            
            # z+[i] - z-[i] = normalized_total_assignments - normalized_fair_share
            self.prob += (
                self.z_plus_vars[person_id] - self.z_minus_vars[person_id] ==
                normalized_total_assignments - normalized_fair_share
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
        return self.prob.status
    
    def solution(self) -> List[int]:
        """Return the new shift assignment as an array of person IDs."""
        assert self.prob, "Model has not been built. Call solve() first."
        assert self.prob.status == pulp.LpStatusOptimal, f"No optimal solution found. Problem status: {pulp.LpStatus[self.prob.status]}"

        return sorted([person_id for person_id, var in self.x_vars.items() if var.varValue == 1])
