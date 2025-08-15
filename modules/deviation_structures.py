
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import json

# Enhanced Data Structures for Pharmaceutical Deviation Reports
class DeviationType(Enum):
    EQUIPMENT = "Equipment"
    PROCESS = "Process"
    PROCEDURE = "Procedure"
    DOCUMENTATION = "Documentation"
    PERSONNEL = "Personnel"
    MATERIAL = "Material"
    FACILITY = "Facility"
    FORMULATION = "Formulation"

class Priority(Enum):
    CRITICAL = "Critical"
    MAJOR = "Major"
    MINOR = "Minor"

class Status(Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    CLOSED_DONE = "Closed - Done"
    CLOSED_CANCELLED = "Closed - Cancelled"
    INCIDENT_CLOSED = "Incident Closed - Done"  # Added based on examples

class Department(Enum):
    MANUFACTURING = "Manufacturing"
    QA = "QA"
    QC = "QC"
    ENGINEERING = "Engineering"
    WAREHOUSE = "Warehouse"
    REGULATORY = "Regulatory"

# New enums for enhanced functionality
class HumanErrorCategory(Enum):
    SKILL_BASED = "Skill-based Error"
    RULE_BASED = "Rule-based Error"
    KNOWLEDGE_BASED = "Knowledge-based Error"
    PROCEDURE_NOT_FOLLOWED = "Procedure Not Followed"
    INADEQUATE_TRAINING = "Inadequate Training"
    COMMUNICATION_ERROR = "Communication Error"
    DOCUMENTATION_ERROR = "Documentation Error"

class RootCauseCategory(Enum):
    HUMAN_ERROR = "Human Error"
    EQUIPMENT_FAILURE = "Equipment Failure"
    PROCEDURE_INADEQUACY = "Procedure Inadequacy"
    TRAINING_DEFICIENCY = "Training Deficiency"
    SYSTEM_FAILURE = "System Failure"
    ENVIRONMENTAL = "Environmental"
    MATERIAL_DEFECT = "Material Defect"
    DESIGN_DEFICIENCY = "Design Deficiency"

@dataclass
class RootCauseAnalysis:
    """Enhanced root cause analysis structure"""
    primary_cause: Optional[RootCauseCategory] = None
    human_error_category: Optional[HumanErrorCategory] = None
    detailed_analysis: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    why_analysis: List[str] = field(default_factory=list)  # 5 Why analysis

@dataclass
class CorrectiveAction:
    """Structure for corrective and preventive actions"""
    action_id: str = ""
    description: str = ""
    responsible_person: str = ""
    target_completion_date: Optional[datetime] = None
    actual_completion_date: Optional[datetime] = None
    status: str = "Open"
    action_type: str = "Corrective"  # Corrective or Preventive

@dataclass
class EffectivenessCheck:
    """Structure for effectiveness monitoring"""
    check_description: str = ""
    check_method: str = ""
    success_criteria: str = ""
    planned_check_date: Optional[datetime] = None
    actual_check_date: Optional[datetime] = None
    results: str = ""
    is_effective: Optional[bool] = None

@dataclass
class DeviationInfo:
    """Enhanced deviation information structure"""
    title: str = ""
    deviation_type: Optional[DeviationType] = None
    priority: Priority = Priority.MINOR
    status: Status = Status.OPEN
    date_of_occurrence: Optional[datetime] = None
    department: List[Department] = field(default_factory=list)
    batch_lot_number: str = ""
    quantity_impacted: str = ""
    is_planned_deviation: bool = False
    description: str = ""
    product_name: str = ""  # Added based on examples
    material_id: str = ""   # Added based on examples

@dataclass
class HeaderInfo:
    """Report header information"""
    record_id: str = ""
    initiator: str = ""
    report_generated: datetime = field(default_factory=datetime.now)
    company_name: str = "CostPlus Drug Company"

@dataclass
class Investigation:
    """Enhanced investigation structure"""
    assignee: str = ""
    findings: str = ""
    conclusion: str = ""
    investigation_required: bool = True
    no_investigation_justification: str = ""
    investigation_completion_date: Optional[datetime] = None

@dataclass
class RiskAssessment:
    """Enhanced risk assessment structure"""
    risk_description: str = ""
    impact_assessment: str = ""
    probability_assessment: str = ""
    risk_level: str = ""  # High, Medium, Low
    mitigation_measures: List[str] = field(default_factory=list)
    conclusion: str = ""
    product_impact: str = ""
    patient_safety_impact: str = ""

@dataclass
class AIGeneratedContent:
    """Structure to track AI-generated suggestions and content"""
    enhanced_description: str = ""
    root_cause_suggestions: List[str] = field(default_factory=list)
    corrective_action_suggestions: List[str] = field(default_factory=list)
    risk_assessment_suggestions: str = ""
    effectiveness_check_suggestions: str = ""
    similar_deviations_references: List[str] = field(default_factory=list)

@dataclass
class DeviationReport:
    """Complete deviation report structure"""
    header: HeaderInfo = field(default_factory=HeaderInfo)
    deviation_info: DeviationInfo = field(default_factory=DeviationInfo)
    investigation: Investigation = field(default_factory=Investigation)
    risk_assessment: RiskAssessment = field(default_factory=RiskAssessment)
    root_cause_analysis: RootCauseAnalysis = field(default_factory=RootCauseAnalysis)
    immediate_actions: List[str] = field(default_factory=list)
    corrective_actions: List[CorrectiveAction] = field(default_factory=list)
    preventive_actions: List[CorrectiveAction] = field(default_factory=list)
    effectiveness_checks: List[EffectivenessCheck] = field(default_factory=list)
    ai_generated: AIGeneratedContent = field(default_factory=AIGeneratedContent)
    referenced_records: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for easier template processing"""
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return {key: serialize_datetime(value) for key, value in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [serialize_datetime(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            return obj

        return serialize_datetime(self.__dict__)

    def to_json(self) -> str:
        """Convert report to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)

# Utility functions for data validation and conversion
class DeviationDataValidator:
    """Validation utilities for deviation data"""

    @staticmethod
    def validate_priority(priority_str: str) -> Priority:
        """Convert string to Priority enum with validation"""
        priority_map = {
            "critical": Priority.CRITICAL,
            "major": Priority.MAJOR,
            "minor": Priority.MINOR,
            "high": Priority.CRITICAL,
            "medium": Priority.MAJOR,
            "low": Priority.MINOR
        }
        return priority_map.get(priority_str.lower(), Priority.MINOR)

    @staticmethod
    def validate_deviation_type(type_str: str) -> Optional[DeviationType]:
        """Convert string to DeviationType enum with validation"""
        type_map = {
            "equipment": DeviationType.EQUIPMENT,
            "process": DeviationType.PROCESS,
            "procedure": DeviationType.PROCEDURE,
            "documentation": DeviationType.DOCUMENTATION,
            "personnel": DeviationType.PERSONNEL,
            "material": DeviationType.MATERIAL,
            "facility": DeviationType.FACILITY,
            "formulation": DeviationType.FORMULATION
        }
        return type_map.get(type_str.lower())

    @staticmethod
    def parse_departments(dept_string: str) -> List[Department]:
        """Parse comma-separated department string into Department enums"""
        if not dept_string:
            return []

        dept_map = {
            "manufacturing": Department.MANUFACTURING,
            "qa": Department.QA,
            "qc": Department.QC,
            "engineering": Department.ENGINEERING,
            "warehouse": Department.WAREHOUSE,
            "regulatory": Department.REGULATORY
        }

        departments = []
        for dept in dept_string.split(','):
            dept_clean = dept.strip().lower()
            if dept_clean in dept_map:
                departments.append(dept_map[dept_clean])

        return departments

    @staticmethod
    def parse_date(date_string: str) -> Optional[datetime]:
        """Parse various date formats into datetime object"""
        if not date_string:
            return None

        # Common date formats
        formats = [
            "%m/%d/%Y",
            "%m/%d/%Y %H:%M",
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y",
            "%B %d, %Y"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_string.strip(), fmt)
            except ValueError:
                continue

        return None

# Template data for common pharmaceutical deviations
COMMON_DEVIATION_TEMPLATES = {
    "mixing_speed": {
        "type": DeviationType.PROCESS,
        "common_causes": ["Human error", "Procedure misunderstanding", "Equipment setting error"],
        "typical_actions": ["Review procedure", "Retrain operator", "Verify equipment calibration"]
    },
    "documentation_error": {
        "type": DeviationType.DOCUMENTATION,
        "common_causes": ["Transcription error", "Inadequate training", "Procedure unclear"],
        "typical_actions": ["Document correction", "Training reinforcement", "Procedure clarification"]
    },
    "line_clearance": {
        "type": DeviationType.PROCEDURE,
        "common_causes": ["Procedure not followed", "Time pressure", "Communication breakdown"],
        "typical_actions": ["Immediate clearance verification", "Process review", "Training update"]
    }
}
