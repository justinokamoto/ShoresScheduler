#!/usr/bin/env python3
"""
Main entry point for the Incremental Personnel Scheduling System.

This script provides a command-line interface to schedule personnel for a new shift
using the Integer Linear Program (ILP) model defined in scheduling_ilp_model.py.
"""

import argparse
from datetime import datetime
from scheduling_ilp_model import IncrementalPersonnelScheduler


def main():
    """Main function with argument parsing for the personnel scheduling system."""
    parser = argparse.ArgumentParser(
        description='Schedule personnel for a new shift using Integer Linear Programming.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 2025-01-15
  %(prog)s 2025-01-15 --database custom_database.json
  %(prog)s 2025-01-15 --database custom_database.json --min-days 14
        """
    )
    
    # Required parameter
    parser.add_argument(
        'shift_date',
        help='Date for the new shift to be scheduled (YYYY-MM-DD format)'
    )
    
    # Optional parameters
    parser.add_argument(
        '--database',
        default='database.json',
        help='Path to the JSON database file containing personnel and existing shift data. Default: database.json'
    )
    
    parser.add_argument(
        '--min-days',
        type=int,
        default=30,
        help='Minimum number of days required between shifts for the same person. Default: 30'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate shift date format
    try:
        datetime.fromisoformat(args.shift_date)
    except ValueError:
        parser.error(f"Invalid shift date format: {args.shift_date}. Use YYYY-MM-DD format.")
    
    # Initialize the scheduler with parsed arguments
    print(f"Initializing scheduler with:")
    print(f"  Database file: {args.database}")
    print(f"  New shift date: {args.shift_date}")
    print(f"  Minimum days between shifts: {args.min_days}")
    print()
    
    try:
        scheduler = IncrementalPersonnelScheduler(
            args.database, 
            args.shift_date, 
            min_days_between_shifts=args.min_days
        )
        
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
            
            print(f"\n=== Optimal Assignment for New Shift ({args.shift_date}) ===")
            if solution["new_shift_assignment"]:
                person_names = []
                for person_id in solution["new_shift_assignment"]:
                    person = scheduler.data.people.get(person_id)
                    if person:
                        person_names.append(person.name)
                print(f"{args.shift_date}: {', '.join(person_names)} (ID: {solution['new_shift_assignment']})")
            else:
                print(f"{args.shift_date}: No assignment found")
            
            print("\n=== Total Assignment Counts (Existing + New) ===")
            for person_id, count in solution["total_assignment_counts"].items():
                person = scheduler.data.people.get(person_id)
                person_name = person.name if person else f"Person {person_id}"
                existing = solution["existing_assignment_counts"][person_id]
                new = 1 if person_id in solution["new_shift_assignment"] else 0
                print(f"{person_name}: {existing} + {new} = {count} assignments")
            
            print("\n=== Fairness Metrics ===")
            for person_id, metrics in solution["fairness_metrics"].items():
                person = scheduler.data.people.get(person_id)
                person_name = person.name if person else f"Person {person_id}"
                print(f"{person_name}:")
                print(f"  - Existing Assignments: {metrics['existing_assignments']}")
                print(f"  - New Assignment: {metrics['new_assignment']}")
                print(f"  - Total Assignments: {metrics['total_assignments']}")
                print(f"  - Availability Weight: {metrics['availability_weight']}")
                print(f"  - Normalized Total: {metrics['normalized_total_assignments']:.4f}")
                print(f"  - Deviation: +{metrics['positive_deviation']:.4f}, -{metrics['negative_deviation']:.4f}")
        else:
            print("No optimal solution found!")
            print(solution)
            
    except FileNotFoundError:
        print(f"Error: Database file '{args.database}' not found.")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
