"""
Data Models for Personnel Scheduling System
==========================================

This module contains the data models that represent the core entities
in the personnel scheduling system: Person, Shift, and ScheduleData.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict


@dataclass
class Person:
    """Represents a person who can be assigned to shifts."""
    id: int
    name: str
    male: bool = True
    fluent_pt: bool = False
    capacity_factor: float = 1.0
    unavailable_periods: List[Dict[str, str]] = field(default_factory=list)
    
    def is_available_on_date(self, date: datetime) -> bool:
        """Check if person is available on a specific date."""
        for period in self.unavailable_periods:
            start_date = datetime.fromisoformat(period['start']) if period.get('start') else datetime.min
            end_date = datetime.fromisoformat(period['end']) if period.get('end') else datetime.max
            if start_date <= date <= end_date:
                return False
        return True
    
    @property
    def is_female(self) -> bool:
        """Check if person is female."""
        return not self.male
    
    @classmethod
    def from_dict(cls, person_id: int, data: Dict) -> 'Person':
        """Create Person from JSON data."""
        return cls(
            id=person_id,
            name=data.get('name', f'Person {person_id}'),
            male=data.get('male', True),
            fluent_pt=data.get('fluent_pt', False),
            capacity_factor=data.get('capacity_factor', 1.0),
            unavailable_periods=data.get('unavailable', [])
        )
    
    def block_availability(self, start_date: datetime, end_date: datetime):
        """Block this person's availability for a given date range."""
        if start_date > end_date:
            raise ValueError("Start date cannot be after end date")
        
        # Add the unavailable period
        unavailable_period = {
            'start': start_date.isoformat(),
            'end': end_date.isoformat()
        }
        
        self.unavailable_periods.append(unavailable_period)
    
    def clear_availability(self, start_date: datetime, end_date: datetime):
        """Clear this person's unavailable periods within a given date range."""
        if start_date > end_date:
            raise ValueError("Start date cannot be after end date")
        
        updated_periods = []
        
        for period in self.unavailable_periods:
            period_start = datetime.fromisoformat(period['start']) if period.get('start') else datetime.min
            period_end = datetime.fromisoformat(period['end']) if period.get('end') else datetime.max
            
            # Check if period overlaps with the clearing range
            if period_end < start_date or period_start > end_date:
                # No overlap, keep the period as is
                updated_periods.append(period)
            else:
                # There is overlap, we need to handle partial overlap
                if period_start < start_date:
                    # Keep the part before the clearing range
                    updated_periods.append({
                        'start': period['start'],
                        'end': (start_date - timedelta(microseconds=1)).isoformat()
                    })
                
                if period_end > end_date:
                    # Keep the part after the clearing range
                    updated_periods.append({
                        'start': (end_date + timedelta(microseconds=1)).isoformat(),
                        'end': period['end']
                    })
                # The overlapping part is removed (not added to updated_periods)
        
        self.unavailable_periods = updated_periods


@dataclass
class Shift:
    """Represents a shift that needs to be staffed."""
    date: datetime
    
    @property
    def date_str(self) -> str:
        """Get ISO format date string."""
        return self.date.isoformat()
    
    @classmethod
    def from_date_string(cls, date_str: str) -> 'Shift':
        """Create Shift from ISO date string."""
        return cls(date=datetime.fromisoformat(date_str))


class ScheduleData:
    """Contains all the data needed for personnel scheduling."""
    
    def __init__(self, data_file: str):
        """Initialize from JSON data file."""
        self.people: Dict[int, Person] = {}
        self.shifts: List[Shift] = []
        self.existing_assignments: List[List[int]] = []
        self._load_data(data_file)
    
    def _load_data(self, data_file: str):
        """Load and parse data from JSON file."""
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Load people
        for person_id_str, person_data in data['people'].items():
            person_id = int(person_id_str)
            self.people[person_id] = Person.from_dict(person_id, person_data)
        
        # Load shifts
        for shift_date_str in data['shifts']:
            self.shifts.append(Shift.from_date_string(shift_date_str))
        
        # Load existing assignments
        self.existing_assignments = data.get('scheduled', [])
    
    @property
    def person_ids(self) -> List[int]:
        """Get list of all person IDs."""
        return list(self.people.keys())
    
    @property
    def shift_indices(self) -> List[int]:
        """Get list of all shift indices."""
        return list(range(len(self.shifts)))
    
    def get_person(self, person_id: int) -> Person:
        """Get person by ID."""
        return self.people[person_id]
    
    def get_shift(self, shift_index: int) -> Shift:
        """Get shift by index."""
        return self.shifts[shift_index]
    
    def get_shift_by_date(self, date: datetime) -> Shift:
        """Get shift by date."""
        for shift in self.shifts:
            if shift.date == date:
                return shift
        raise ValueError(f"No shift found for date {date}")
    
    def add_scheduled_shift(self, shift_date: datetime, person_ids: List[int]):
        """Add a new scheduled shift with assigned persons."""
        # Validate person IDs exist
        for person_id in person_ids:
            if person_id not in self.people:
                raise ValueError(f"Person ID {person_id} does not exist")
        
        # Create and add the new shift
        new_shift = Shift(date=shift_date)
        self.shifts.append(new_shift)
        
        # Add the assignment for this new shift
        self.existing_assignments.append(person_ids.copy())
    
    def save_data(self, data_file: str):
        """Save current schedule data to JSON file."""
        # Convert people to dict format
        people_data = {}
        for person_id, person in self.people.items():
            people_data[str(person_id)] = {
                'name': person.name,
                'male': person.male,
                'fluent_pt': person.fluent_pt,
                'capacity_factor': person.capacity_factor,
                'unavailable': person.unavailable_periods
            }
        
        # Convert shifts to list of date strings
        shifts_data = [shift.date_str for shift in self.shifts]
        
        # Prepare the full data structure
        data = {
            'people': people_data,
            'shifts': shifts_data,
            'scheduled': self.existing_assignments
        }
        
        # Write to file
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
