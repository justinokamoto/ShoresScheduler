"""
Statistics and debugging utilities for the IncrementalPersonnelScheduler.

This module contains functions for analyzing and displaying information about
the scheduling model and solutions.
"""

from typing import Dict
import pulp


def get_solution(scheduler) -> Dict:
    """Get the solution from the solved model for incremental scheduling."""
    if scheduler.prob is None or scheduler.prob.status != pulp.LpStatusOptimal:
        return {"status": "No optimal solution found"}
    
    solution = {
        "status": "Optimal",
        "objective_value": pulp.value(scheduler.prob.objective),
        "new_shift_assignment": [],
        "existing_assignment_counts": scheduler.existing_assignment_counts.copy(),
        "total_assignment_counts": {},
        "fairness_metrics": {}
    }
    
    # Extract assignments for the new shift
    for person_id, var in scheduler.x_vars.items():
        if pulp.value(var) == 1:
            solution["new_shift_assignment"].append(person_id)
    
    # Calculate total assignment counts (existing + new)
    for person_id in scheduler.data.person_ids:
        existing_count = scheduler.existing_assignment_counts[person_id]
        new_count = 1 if person_id in solution["new_shift_assignment"] else 0
        solution["total_assignment_counts"][person_id] = existing_count + new_count
    
    # Extract fairness metrics
    for person_id in scheduler.data.person_ids:
        total_existing_shifts = len(scheduler.data.shifts)
        is_available_for_new = (scheduler._is_person_available_for_new_shift(person_id) and 
                               not scheduler._violates_minimum_days_constraint(person_id))
        availability_weight = (total_existing_shifts + (1 if is_available_for_new else 0))
        
        solution["fairness_metrics"][person_id] = {
            "existing_assignments": scheduler.existing_assignment_counts[person_id],
            "new_assignment": 1 if person_id in solution["new_shift_assignment"] else 0,
            "total_assignments": solution["total_assignment_counts"][person_id],
            "availability_weight": availability_weight,
            "normalized_total_assignments": solution["total_assignment_counts"][person_id] / availability_weight if availability_weight > 0 else 0,
            "positive_deviation": pulp.value(scheduler.z_plus_vars[person_id]),
            "negative_deviation": pulp.value(scheduler.z_minus_vars[person_id])
        }
    
    return solution


def print_model_summary(scheduler):
    """Print a summary of the incremental scheduling model."""
    print("=== Incremental Personnel Scheduling Model Summary ===")
    print(f"Number of people: {len(scheduler.data.person_ids)}")
    print(f"Number of existing shifts: {len(scheduler.data.shifts)}")
    print(f"New shift date: {scheduler.new_shift.date_str}")
    print(f"People needed for new shift: {scheduler.people_needed}")
    print(f"Number of decision variables: {len(scheduler.x_vars)}")
    print(f"Minimum days between shifts: {scheduler.min_days_between_shifts}")
    
    print("\nExisting Shift Dates:")
    for shift_idx, shift in enumerate(scheduler.data.shifts):
        print(f"  Shift {shift_idx}: {shift.date_str}")
    
    print("\nExisting Assignment Counts:")
    for person_id, count in scheduler.existing_assignment_counts.items():
        person = scheduler.data.people.get(person_id)
        person_name = person.name if person else f"Person {person_id}"
        print(f"  {person_name}: {count} assignments")
    
    print(f"\nEligibility for New Shift ({scheduler.new_shift.date_str}):")
    print("Person ID | Name                | Available | Min Days OK | Eligible")
    print("-" * 70)
    
    for person_id in scheduler.data.person_ids:
        person = scheduler.data.people.get(person_id)
        person_name = person.name if person else f"Person {person_id}"
        
        is_available = scheduler._is_person_available_for_new_shift(person_id)
        min_days_ok = not scheduler._violates_minimum_days_constraint(person_id)
        is_eligible = person_id in scheduler.x_vars
        
        print(f"    {person_id:2d}    | {person_name:19s} | {'✓' if is_available else '✗':9s} | {'✓' if min_days_ok else '✗':11s} | {'✓' if is_eligible else '✗'}")
    
    print("\nDecision Variables Created:")
    for person_id in sorted(scheduler.x_vars.keys()):
        person = scheduler.data.people.get(person_id)
        person_name = person.name if person else f"Person {person_id}"
        print(f"  x_{person_id}_new_shift ({person_name})")
    
    print("\nCapacity Factors:")
    for person_id, person in scheduler.data.people.items():
        print(f"  {person.name}: {person.capacity_factor}")
