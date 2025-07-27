"""
Data Models for Personnel Scheduling System
==========================================

This module contains the data models that represent the core entities
in the personnel scheduling system: Person, Shift, and ScheduleData.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
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
