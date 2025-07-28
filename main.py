#!/usr/bin/env python3
"""
Main entry point for the Incremental Personnel Scheduling System.

This script provides a command-line interface to schedule personnel for a new shift
using the Integer Linear Program (ILP) model defined in scheduling_ilp_model.py.
"""

import argparse
from datetime import date
from scheduling_ilp_model import IncrementalPersonnelScheduler
import stats
import pulp


def main():
    """Main function with argument parsing for the personnel scheduling system."""
    parser = argparse.ArgumentParser(
        description='Schedule personnel for a new shift using Integer Linear Programming.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 2025-01-15
  %(prog)s 2025-01-15 --people-needed 3
  %(prog)s 2025-01-15 --database custom_database.json --people-needed 4
  %(prog)s 2025-01-15 --database custom_database.json --min-days 14 --people-needed 2
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
    
    parser.add_argument(
        '--people-needed',
        type=int,
        default=2,
        help='Number of people needed for the shift. Default: 2'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate shift date format
    try:
        date.fromisoformat(args.shift_date)
    except ValueError:
        parser.error(f"Invalid shift date format: {args.shift_date}. Use YYYY-MM-DD format.")
    
    # Validate people_needed parameter
    if args.people_needed < 1:
        parser.error(f"Number of people needed must be at least 1, got: {args.people_needed}")
    
    # Initialize the scheduler with parsed arguments
    print(f"Initializing scheduler with:")
    print(f"  Database file: {args.database}")
    print(f"  New shift date: {args.shift_date}")
    print(f"  Minimum days between shifts: {args.min_days}")
    print(f"  People needed for shift: {args.people_needed}")
    print()
    
    try:
        scheduler = IncrementalPersonnelScheduler(
            args.database, 
            args.shift_date, 
            min_days_between_shifts=args.min_days,
            people_needed=args.people_needed
        )
        
        # Build the model first
        scheduler.build_model()
        
        # Print model summary
        stats.print_model_summary(scheduler)
        
        # Solve the model
        print("\n=== Solving Model ===")
        status = scheduler.solve()
        print(f"Solution Status: {status}")

        if status == pulp.LpStatusOptimal:
            assigned_ids = scheduler.solution()
            
            # TODO: Assign shiz

            # Get and display solution
            stats.print_solution(scheduler)
        else:
            print("No optimal solution found!")
            
    except FileNotFoundError:
        print(f"Error: Database file '{args.database}' not found.")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
