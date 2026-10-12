import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyDiagnosticSystem:
    def __init__(self):
        self.setup_fuzzy_system()
    
    def setup_fuzzy_system(self):
        """Setup fuzzy logic system for heart disease diagnosis"""
        
        # Input variables
        self.age = ctrl.Antecedent(np.arange(20, 100, 1), 'age')
        self.cholesterol = ctrl.Antecedent(np.arange(100, 400, 1), 'cholesterol')
        self.heart_rate = ctrl.Antecedent(np.arange(60, 220, 1), 'heart_rate')
        self.oldpeak = ctrl.Antecedent(np.arange(0, 6.2, 0.1), 'oldpeak')
        
        # Output variable
        self.risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')
        
        # Membership functions for Age
        self.age['young'] = fuzz.trimf(self.age.universe, [20, 30, 45])
        self.age['middle'] = fuzz.trimf(self.age.universe, [35, 50, 65])
        self.age['old'] = fuzz.trimf(self.age.universe, [55, 70, 100])
        
        # Membership functions for Cholesterol
        self.cholesterol['low'] = fuzz.trimf(self.cholesterol.universe, [100, 150, 200])
        self.cholesterol['medium'] = fuzz.trimf(self.cholesterol.universe, [180, 220, 260])
        self.cholesterol['high'] = fuzz.trimf(self.cholesterol.universe, [240, 300, 400])
        
        # Membership functions for Heart Rate
        self.heart_rate['low'] = fuzz.trimf(self.heart_rate.universe, [60, 80, 100])
        self.heart_rate['normal'] = fuzz.trimf(self.heart_rate.universe, [90, 120, 150])
        self.heart_rate['high'] = fuzz.trimf(self.heart_rate.universe, [140, 170, 220])
        
        # Membership functions for Oldpeak
        self.oldpeak['low'] = fuzz.trimf(self.oldpeak.universe, [0, 0.5, 1.5])
        self.oldpeak['medium'] = fuzz.trimf(self.oldpeak.universe, [1.0, 2.0, 3.0])
        self.oldpeak['high'] = fuzz.trimf(self.oldpeak.universe, [2.5, 4.0, 6.2])
        
        # Membership functions for Risk
        self.risk['low'] = fuzz.trimf(self.risk.universe, [0, 0, 30])
        self.risk['medium'] = fuzz.trimf(self.risk.universe, [25, 50, 75])
        self.risk['high'] = fuzz.trimf(self.risk.universe, [70, 100, 100])
        
        # Define rules
        rule1 = ctrl.Rule(self.age['old'] & self.cholesterol['high'], self.risk['high'])
        rule2 = ctrl.Rule(self.age['old'] & self.cholesterol['medium'], self.risk['medium'])
        rule3 = ctrl.Rule(self.age['middle'] & self.cholesterol['high'], self.risk['medium'])
        rule4 = ctrl.Rule(self.heart_rate['high'] & self.oldpeak['high'], self.risk['high'])
        rule5 = ctrl.Rule(self.heart_rate['normal'] & self.oldpeak['low'], self.risk['low'])
        rule6 = ctrl.Rule(self.age['young'] & self.cholesterol['low'], self.risk['low'])
        rule7 = ctrl.Rule(self.oldpeak['high'], self.risk['high'])
        rule8 = ctrl.Rule(self.heart_rate['high'], self.risk['medium'])
        rule9 = ctrl.Rule(self.age['old'] & self.heart_rate['high'], self.risk['high'])
        rule10 = ctrl.Rule(self.cholesterol['high'] & self.oldpeak['medium'], self.risk['high'])
        
        # Create control system
        self.risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, 
                                            rule6, rule7, rule8, rule9, rule10])
        self.risk_sim = ctrl.ControlSystemSimulation(self.risk_ctrl)
    
    def diagnose(self, age, cholesterol, heart_rate, oldpeak):
        """Diagnose heart disease risk using fuzzy logic"""
        try:
            self.risk_sim.input['age'] = age
            self.risk_sim.input['cholesterol'] = cholesterol
            self.risk_sim.input['heart_rate'] = heart_rate
            self.risk_sim.input['oldpeak'] = oldpeak
            
            self.risk_sim.compute()
            risk_score = self.risk_sim.output['risk']
            
            # Interpret risk score
            if risk_score <= 30:
                risk_level = "Low Risk"
            elif risk_score <= 60:
                risk_level = "Medium Risk"
            else:
                risk_level = "High Risk"
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'interpretation': f"Risk score: {risk_score:.1f}% - {risk_level}"
            }
        except Exception as e:
            return {
                'risk_score': 0,
                'risk_level': "Error",
                'interpretation': f"Error in fuzzy diagnosis: {str(e)}"
            }