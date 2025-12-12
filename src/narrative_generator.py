
class MedicalNarrativeGenerator:
    """
    A rule-based deterministic system to generate clinical narratives 
    without external LLM dependencies.
    """
    
    @staticmethod
    def generate_narrative(patient_data: dict, cluster_name: str, risk_profile: str, actions: list) -> str:
        # 1. Opening Statement based on Cluster
        opening = f"Patient presents with a metabolic profile consistent with '{cluster_name}'."
        
        # 2. Risk Assessment
        if risk_profile == "High":
            risk_text = "Immediate intervention is recommended due to elevated cardiovascular and metabolic risk markers."
        else:
            risk_text = "Current biometric indicators suggest a stable maintainable state, though preventive measures are advised."

        # 3. Specific Deviations (The "Intelligence")
        deviations = []
        if patient_data.get('BMI', 0) > 30:
            deviations.append("significant adiposity (BMI > 30)")
        elif patient_data.get('BMI', 0) < 18.5:
             deviations.append("underweight status")

        if patient_data.get('Glucose', 0) > 120:
             deviations.append("hyperglycemic tendency")
        
        if patient_data.get('SystolicBP', 0) > 130:
             deviations.append("elevated systolic pressure")

        dev_text = ""
        if deviations:
            dev_text = f"Key clinical concerns include {', '.join(deviations)}."

        # 4. Action Summary
        if actions:
            action_text = f"The optimization protocol prioritizes {actions[0].lower().replace('_', ' ')} and {actions[1].lower().replace('_', ' ') if len(actions) > 1 else 'monitoring'}."
        else:
            action_text = "No specific lifestyle alterations are mandated at this time."

        # Assemble
        return f"{opening} {risk_text} {dev_text} {action_text}"
