"""Data validation module for Chicago 311 data."""
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate and clean Chicago 311 data."""
    
    def __init__(self):
        """Initialize validator with expected schema."""
        self.required_fields = [
            'sr_number', 'sr_type', 'creation_date', 'status'
        ]
        
        self.optional_fields = [
            'sr_short_code', 'owner_department', 'completion_date',
            'due_date', 'street_address', 'city', 'state', 'zip_code',
            'ward', 'police_district', 'community_area', 'ssa',
            'latitude', 'longitude', 'source', 'duplicate',
            'legacy_record', 'legacy_sr_number', 'parent_sr_number'
        ]
        
        self.valid_statuses = [
            'Open', 'Completed', 'Duplicate', 'Cancelled', 'Closed'
        ]
    
    def validate_record(self, record: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a single record."""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in record or record[field] is None or record[field] == '':
                errors.append(f"Missing required field: {field}")
        
        # Validate SR number format
        if 'sr_number' in record and record['sr_number']:
            if not self._validate_sr_number(record['sr_number']):
                errors.append(f"Invalid SR number format: {record['sr_number']}")
        
        # Validate dates
        date_fields = ['creation_date', 'completion_date', 'due_date']
        for field in date_fields:
            if field in record and record[field]:
                if not self._validate_date(record[field]):
                    errors.append(f"Invalid date format in {field}: {record[field]}")
        
        # Validate status
        if 'status' in record and record['status']:
            if record['status'] not in self.valid_statuses:
                errors.append(f"Invalid status: {record['status']}")
        
        # Validate coordinates
        if 'latitude' in record and record['latitude']:
            if not self._validate_latitude(record['latitude']):
                errors.append(f"Invalid latitude: {record['latitude']}")
        
        if 'longitude' in record and record['longitude']:
            if not self._validate_longitude(record['longitude']):
                errors.append(f"Invalid longitude: {record['longitude']}")
        
        # Validate ward (Chicago has 50 wards)
        if 'ward' in record and record['ward']:
            try:
                ward_num = int(record['ward'])
                if ward_num < 1 or ward_num > 50:
                    errors.append(f"Invalid ward number: {ward_num}")
            except (ValueError, TypeError):
                errors.append(f"Ward must be a number: {record['ward']}")
        
        # Validate zip code
        if 'zip_code' in record and record['zip_code']:
            if not self._validate_zip_code(record['zip_code']):
                errors.append(f"Invalid zip code: {record['zip_code']}")
        
        return len(errors) == 0, errors
    
    def validate_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a batch of records."""
        validation_results = {
            'total_records': len(records),
            'valid_records': 0,
            'invalid_records': 0,
            'errors': [],
            'valid_data': [],
            'error_summary': {}
        }
        
        for i, record in enumerate(records):
            is_valid, errors = self.validate_record(record)
            
            if is_valid:
                validation_results['valid_records'] += 1
                validation_results['valid_data'].append(record)
            else:
                validation_results['invalid_records'] += 1
                validation_results['errors'].append({
                    'record_index': i,
                    'sr_number': record.get('sr_number', 'Unknown'),
                    'errors': errors
                })
                
                # Count error types
                for error in errors:
                    error_type = error.split(':')[0]
                    validation_results['error_summary'][error_type] = (
                        validation_results['error_summary'].get(error_type, 0) + 1
                    )
        
        # Calculate validation rate
        validation_results['validation_rate'] = (
            validation_results['valid_records'] / validation_results['total_records']
            if validation_results['total_records'] > 0 else 0
        )
        
        return validation_results
    
    def clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize a record."""
        cleaned = record.copy()
        
        # Clean string fields
        string_fields = ['sr_type', 'owner_department', 'street_address', 'city', 'state']
        for field in string_fields:
            if field in cleaned and cleaned[field]:
                cleaned[field] = str(cleaned[field]).strip()
        
        # Standardize status
        if 'status' in cleaned and cleaned[field]:
            cleaned['status'] = cleaned['status'].title()
        
        # Clean and validate numeric fields
        numeric_fields = ['ward', 'community_area']
        for field in numeric_fields:
            if field in cleaned and cleaned[field]:
                try:
                    cleaned[field] = int(cleaned[field])
                except (ValueError, TypeError):
                    cleaned[field] = None
        
        # Clean coordinates
        for coord_field in ['latitude', 'longitude']:
            if coord_field in cleaned and cleaned[coord_field]:
                try:
                    cleaned[coord_field] = float(cleaned[coord_field])
                except (ValueError, TypeError):
                    cleaned[coord_field] = None
        
        # Parse dates
        date_fields = ['creation_date', 'completion_date', 'due_date']
        for field in date_fields:
            if field in cleaned and cleaned[field]:
                cleaned[field] = self._parse_datetime(cleaned[field])
        
        # Clean zip code
        if 'zip_code' in cleaned and cleaned['zip_code']:
            cleaned['zip_code'] = self._clean_zip_code(cleaned['zip_code'])
        
        # Handle boolean fields
        boolean_fields = ['duplicate', 'legacy_record']
        for field in boolean_fields:
            if field in cleaned and cleaned[field] is not None:
                cleaned[field] = self._parse_boolean(cleaned[field])
        
        return cleaned
    
    def _validate_sr_number(self, sr_number: str) -> bool:
        """Validate service request number format."""
        # Chicago SR numbers are typically 8-10 digits
        pattern = r'^\d{8,10}$'
        return bool(re.match(pattern, str(sr_number)))
    
    def _validate_date(self, date_str: str) -> bool:
        """Validate date string."""
        try:
            pd.to_datetime(date_str)
            return True
        except:
            return False
    
    def _validate_latitude(self, lat: Any) -> bool:
        """Validate latitude value."""
        try:
            lat_float = float(lat)
            # Chicago latitude range approximately
            return 41.6 <= lat_float <= 42.1
        except:
            return False
    
    def _validate_longitude(self, lon: Any) -> bool:
        """Validate longitude value."""
        try:
            lon_float = float(lon)
            # Chicago longitude range approximately
            return -87.9 <= lon_float <= -87.5
        except:
            return False
    
    def _validate_zip_code(self, zip_code: str) -> bool:
        """Validate Chicago zip code."""
        # Chicago zip codes start with 606xx
        pattern = r'^606\d{2}(-\d{4})?$'
        return bool(re.match(pattern, str(zip_code)))
    
    def _parse_datetime(self, date_str: Any) -> Optional[datetime]:
        """Parse datetime string."""
        try:
            return pd.to_datetime(date_str).to_pydatetime()
        except:
            return None
    
    def _clean_zip_code(self, zip_code: str) -> str:
        """Clean zip code format."""
        # Extract just the 5-digit zip
        zip_str = str(zip_code)
        match = re.search(r'(\d{5})', zip_str)
        return match.group(1) if match else zip_str
    
    def _parse_boolean(self, value: Any) -> Optional[bool]:
        """Parse boolean value."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'y']
        return None

def get_data_quality_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive data quality report."""
    validator = DataValidator()
    validation_results = validator.validate_batch(records)
    
    # Additional quality metrics
    df = pd.DataFrame(records)
    
    quality_report = {
        'validation_results': validation_results,
        'completeness': {},
        'uniqueness': {},
        'distributions': {}
    }
    
    # Completeness analysis
    for column in df.columns:
        null_count = df[column].isnull().sum()
        empty_count = (df[column] == '').sum() if df[column].dtype == 'object' else 0
        total_missing = null_count + empty_count
        
        quality_report['completeness'][column] = {
            'missing_count': int(total_missing),
            'missing_percentage': float(total_missing / len(df) * 100),
            'completeness_score': float((len(df) - total_missing) / len(df) * 100)
        }
    
    # Uniqueness analysis for key fields
    key_fields = ['sr_number', 'sr_type', 'status', 'ward']
    for field in key_fields:
        if field in df.columns:
            total_count = len(df[field].dropna())
            unique_count = df[field].nunique()
            quality_report['uniqueness'][field] = {
                'total': total_count,
                'unique': unique_count,
                'duplicate_rate': float((total_count - unique_count) / total_count * 100) if total_count > 0 else 0
            }
    
    return quality_report

if __name__ == "__main__":
    # Test the validator
    test_record = {
        'sr_number': '12345678',
        'sr_type': 'Pothole in Street',
        'creation_date': '2023-01-01T12:00:00',
        'status': 'Open',
        'latitude': '41.8781',
        'longitude': '-87.6298',
        'ward': '1',
        'zip_code': '60601'
    }
    
    validator = DataValidator()
    is_valid, errors = validator.validate_record(test_record)
    print(f"Record valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    cleaned_record = validator.clean_record(test_record)
    print(f"Cleaned record: {cleaned_record}")