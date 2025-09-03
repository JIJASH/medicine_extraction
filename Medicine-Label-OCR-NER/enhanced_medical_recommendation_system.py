"""
Enhanced Medical Recommendation System with Advanced Safety Measures
Author: GitHub Copilot
Date: August 25, 2025

MEDICAL DISCLAIMER: This system is for educational and informational purposes only.
Always consult a qualified healthcare professional for medical advice.
"""

import pandas as pd
import numpy as np
import mysql.connector
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedMedicalRecommendationSystem:
    """
    Advanced Medicine Recommendation System with Enhanced Safety Features
    """
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.medicines_df = None
        self.tfidf_vectorizer = None
        self.similarity_matrix = None
        self.therapeutic_classes = {}
        self.safety_threshold = 0.7  # Higher threshold for safety
        self.min_recommendations = 2
        self.max_recommendations = 3  # Limit recommendations
        
        # Enhanced safety parameters
        self.CRITICAL_SIMILARITY_THRESHOLD = 0.8  # Very high threshold
        self.DOSAGE_WEIGHT = 0.4      # Increased weight for dosage matching
        self.INDICATION_WEIGHT = 0.3   # High weight for therapeutic indication
        self.CONTRAINDICATION_WEIGHT = 0.2  # Safety considerations
        self.INTERACTION_WEIGHT = 0.1   # Drug interaction considerations
        
    def load_medicines_from_database(self):
        """
        Load medicines from MySQL database with enhanced data validation
        """
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            # Enhanced query with more fields for better analysis
            query = """
            SELECT 
                medicine_name,
                dosage_form,
                indication,
                contraindications,
                adverse_effects,
                drug_interactions,
                therapeutic_class,
                active_ingredients,
                strength,
                manufacturer,
                approval_status
            FROM medicines 
            WHERE approval_status = 'APPROVED' 
            AND contraindications IS NOT NULL
            ORDER BY medicine_name
            """
            
            cursor.execute(query)
            medicines_data = cursor.fetchall()
            
            if not medicines_data:
                print("❌ No approved medicines found in database")
                return False
                
            self.medicines_df = pd.DataFrame(medicines_data)
            
            # Data validation and cleaning
            self.medicines_df = self.medicines_df.dropna(subset=['medicine_name', 'indication'])
            self.medicines_df = self.medicines_df.drop_duplicates(subset=['medicine_name'])
            
            print(f"✅ Loaded {len(self.medicines_df)} approved medicines from database")
            return True
            
        except mysql.connector.Error as e:
            print(f"❌ Database error: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def extract_therapeutic_class(self, medicine_info):
        """
        Extract or infer therapeutic class from medicine information
        """
        indication = str(medicine_info.get('indication', '')).lower()
        dosage_form = str(medicine_info.get('dosage_form', '')).lower()
        
        # Enhanced therapeutic classification
        therapeutic_mapping = {
            'analgesic': ['pain', 'analgesic', 'painkiller'],
            'antibiotic': ['infection', 'bacterial', 'antibiotic'],
            'antihypertensive': ['hypertension', 'blood pressure', 'cardiac'],
            'antidiabetic': ['diabetes', 'glucose', 'insulin'],
            'antihistamine': ['allergy', 'histamine', 'allergic'],
            'antacid': ['acid', 'heartburn', 'gastric'],
            'antipyretic': ['fever', 'pyretic', 'temperature'],
            'anti-inflammatory': ['inflammation', 'inflammatory', 'swelling']
        }
        
        for therapeutic_class, keywords in therapeutic_mapping.items():
            if any(keyword in indication for keyword in keywords):
                return therapeutic_class
                
        return 'general'
    
    def create_enhanced_features(self, row):
        """
        Create enhanced feature representation with medical domain expertise
        """
        features = []
        
        # 1. Active Ingredients (Highest Priority - 4x weight)
        active_ingredients = str(row.get('active_ingredients', '')).lower()
        if active_ingredients and active_ingredients != 'nan':
            features.extend([active_ingredients] * 4)
        
        # 2. Therapeutic Class (Very High Priority - 3x weight) 
        therapeutic_class = self.extract_therapeutic_class(row)
        features.extend([therapeutic_class] * 3)
        
        # 3. Primary Indication (High Priority - 3x weight)
        indication = str(row.get('indication', '')).lower()
        if indication and indication != 'nan':
            # Extract primary indication (first sentence or main condition)
            primary_indication = indication.split('.')[0].split(',')[0]
            features.extend([primary_indication] * 3)
        
        # 4. Dosage Form Category (Medium Priority - 2x weight)
        dosage_form = str(row.get('dosage_form', '')).lower()
        if dosage_form and dosage_form != 'nan':
            # Extract form type (tablet, injection, etc.)
            form_type = re.search(r'(tablet|capsule|injection|cream|ointment|syrup)', dosage_form)
            if form_type:
                features.extend([form_type.group(1)] * 2)
        
        # 5. Strength/Dosage (Medium Priority - 2x weight)
        strength = str(row.get('strength', '')).lower()
        if strength and strength != 'nan':
            features.extend([strength] * 2)
        
        # 6. Contraindications (Safety Priority - 1x weight)
        contraindications = str(row.get('contraindications', '')).lower()
        if contraindications and contraindications != 'nan':
            features.append(contraindications)
        
        return ' '.join(features) if features else 'unknown_medicine'
    
    def build_enhanced_similarity_matrix(self):
        """
        Build similarity matrix with enhanced medical considerations
        """
        if self.medicines_df is None or len(self.medicines_df) == 0:
            print("❌ No medicine data available")
            return False
        
        # Create enhanced features
        self.medicines_df['enhanced_features'] = self.medicines_df.apply(
            self.create_enhanced_features, axis=1
        )
        
        # Enhanced TF-IDF with medical domain optimization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,        # Focused feature set
            stop_words='english',     
            ngram_range=(1, 3),       # Include medical phrases (up to 3 words)
            min_df=1,                 
            max_df=0.7,              # More restrictive to avoid common terms
            lowercase=True,
            token_pattern=r'[a-zA-Z][a-zA-Z0-9]*',  # Medical terminology pattern
            strip_accents='unicode'
        )
        
        # Build TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.medicines_df['enhanced_features']
        )
        
        # Calculate cosine similarity
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print(f"✅ Enhanced similarity matrix built: {self.similarity_matrix.shape}")
        return True
    
    def check_safety_compatibility(self, medicine1_idx, medicine2_idx):
        """
        Enhanced safety check between two medicines
        """
        med1 = self.medicines_df.iloc[medicine1_idx]
        med2 = self.medicines_df.iloc[medicine2_idx]
        
        safety_score = 1.0
        warnings = []
        
        # 1. Therapeutic Class Compatibility
        class1 = self.extract_therapeutic_class(med1)
        class2 = self.extract_therapeutic_class(med2)
        
        if class1 != class2:
            safety_score *= 0.7  # Penalize different therapeutic classes
            warnings.append(f"Different therapeutic classes: {class1} vs {class2}")
        
        # 2. Contraindication Cross-Check
        contra1 = str(med1.get('contraindications', '')).lower()
        contra2 = str(med2.get('contraindications', '')).lower()
        
        # Check for overlapping contraindications
        if contra1 and contra2:
            contra1_words = set(contra1.split())
            contra2_words = set(contra2.split())
            overlap = len(contra1_words.intersection(contra2_words))
            
            if overlap > 2:  # Significant contraindication overlap
                safety_score *= 0.8
                warnings.append("Similar contraindication profiles")
        
        # 3. Dosage Form Compatibility
        form1 = str(med1.get('dosage_form', '')).lower()
        form2 = str(med2.get('dosage_form', '')).lower()
        
        # Extract form types
        form1_type = re.search(r'(tablet|capsule|injection|cream|ointment|syrup)', form1)
        form2_type = re.search(r'(tablet|capsule|injection|cream|ointment|syrup)', form2)
        
        if form1_type and form2_type:
            if form1_type.group(1) != form2_type.group(1):
                safety_score *= 0.9  # Minor penalty for different forms
                warnings.append(f"Different dosage forms: {form1_type.group(1)} vs {form2_type.group(1)}")
        
        return safety_score, warnings
    
    def get_safe_recommendations(self, medicine_name, max_recommendations=3):
        """
        Get safe medicine recommendations with enhanced validation
        """
        # Find medicine index
        medicine_name_clean = medicine_name.strip().upper()
        
        # Try exact match first
        medicine_matches = self.medicines_df[
            self.medicines_df['medicine_name'].str.upper() == medicine_name_clean
        ]
        
        if medicine_matches.empty:
            # Try partial match
            medicine_matches = self.medicines_df[
                self.medicines_df['medicine_name'].str.contains(medicine_name_clean, na=False, case=False)
            ]
        
        if medicine_matches.empty:
            return {
                'status': 'error',
                'message': f"❌ Medicine '{medicine_name}' not found in approved medicines database.",
                'recommendations': []
            }
        
        medicine_idx = medicine_matches.index[0]
        target_medicine = self.medicines_df.iloc[medicine_idx]
        
        # Get similarity scores
        similarity_scores = self.similarity_matrix[medicine_idx]
        
        # Create candidate list with safety scores
        candidates = []
        for idx, sim_score in enumerate(similarity_scores):
            if idx != medicine_idx:  # Exclude the medicine itself
                safety_score, safety_warnings = self.check_safety_compatibility(medicine_idx, idx)
                
                # Combined score: similarity + safety
                combined_score = sim_score * safety_score
                
                if combined_score >= self.CRITICAL_SIMILARITY_THRESHOLD:
                    candidates.append({
                        'idx': idx,
                        'similarity_score': sim_score,
                        'safety_score': safety_score,
                        'combined_score': combined_score,
                        'safety_warnings': safety_warnings
                    })
        
        # Sort by combined score
        candidates = sorted(candidates, key=lambda x: x['combined_score'], reverse=True)
        
        # Limit recommendations for safety
        candidates = candidates[:max_recommendations]
        
        # Format recommendations
        recommendations = []
        for i, candidate in enumerate(candidates):
            rec_medicine = self.medicines_df.iloc[candidate['idx']]
            
            recommendations.append({
                'rank': i + 1,
                'medicine_name': rec_medicine['medicine_name'],
                'similarity_score': round(candidate['similarity_score'], 4),
                'safety_score': round(candidate['safety_score'], 4),
                'combined_score': round(candidate['combined_score'], 4),
                'dosage_form': rec_medicine.get('dosage_form', 'Not specified'),
                'indication': rec_medicine.get('indication', 'Not specified'),
                'contraindications': rec_medicine.get('contraindications', 'Not specified'),
                'therapeutic_class': self.extract_therapeutic_class(rec_medicine),
                'safety_warnings': candidate['safety_warnings'],
                'manufacturer': rec_medicine.get('manufacturer', 'Not specified')
            })
        
        return {
            'status': 'success',
            'original_medicine': {
                'name': target_medicine['medicine_name'],
                'dosage_form': target_medicine.get('dosage_form', ''),
                'indication': target_medicine.get('indication', ''),
                'therapeutic_class': self.extract_therapeutic_class(target_medicine)
            },
            'recommendations': recommendations,
            'safety_note': "⚠️ MEDICAL DISCLAIMER: These are computational suggestions only. Always consult a qualified healthcare professional.",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def format_safe_recommendations(self, result):
        """
        Format recommendations with enhanced safety information
        """
        if result['status'] == 'error':
            return result['message']
        
        if not result['recommendations']:
            return f"❌ No safe alternatives found for '{result['original_medicine']['name']}' with current safety thresholds.\n\n⚠️ CONSULT YOUR DOCTOR for alternative medications."
        
        output = []
        output.append("🏥 ENHANCED MEDICAL RECOMMENDATION SYSTEM")
        output.append("=" * 70)
        output.append(f"📊 Generated on: {result['timestamp']}")
        output.append(f"🎯 Original Medicine: {result['original_medicine']['name']}")
        output.append(f"🏷️ Therapeutic Class: {result['original_medicine']['therapeutic_class']}")
        output.append("")
        output.append("💊 SAFE ALTERNATIVE RECOMMENDATIONS:")
        output.append("=" * 50)
        
        for rec in result['recommendations']:
            output.append(f"\n🏆 RANK {rec['rank']}: {rec['medicine_name']}")
            output.append("-" * 40)
            output.append(f"   📊 Similarity Score: {rec['similarity_score']} (Max: 1.0)")
            output.append(f"   🛡️ Safety Score: {rec['safety_score']} (Max: 1.0)")
            output.append(f"   🎯 Combined Score: {rec['combined_score']} (Min: {self.CRITICAL_SIMILARITY_THRESHOLD})")
            output.append(f"   🏷️ Therapeutic Class: {rec['therapeutic_class']}")
            output.append(f"   💉 Dosage Form: {rec['dosage_form']}")
            output.append(f"   🎯 Indication: {rec['indication'][:100]}...")
            output.append(f"   ⚠️ Contraindications: {rec['contraindications'][:100]}...")
            output.append(f"   🏭 Manufacturer: {rec['manufacturer']}")
            
            if rec['safety_warnings']:
                output.append(f"   ⚠️ Safety Notes:")
                for warning in rec['safety_warnings']:
                    output.append(f"      • {warning}")
        
        output.append("\n" + "=" * 70)
        output.append("🚨 CRITICAL MEDICAL DISCLAIMER:")
        output.append("• This is a computational analysis tool ONLY")
        output.append("• NOT a substitute for professional medical advice")
        output.append("• ALWAYS consult your doctor before changing medications")
        output.append("• Consider patient-specific factors: allergies, age, pregnancy")
        output.append("• Verify drug interactions with current medications")
        output.append("• Confirm appropriate dosing for patient condition")
        output.append("=" * 70)
        
        return "\n".join(output)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # Update with your credentials
    'password': 'password',  # Update with your credentials
    'database': 'medicine_db'  # Update with your database name
}

# Example usage
if __name__ == "__main__":
    # Initialize the enhanced system
    enhanced_system = EnhancedMedicalRecommendationSystem(DB_CONFIG)
    
    # Load medicines from database
    if enhanced_system.load_medicines_from_database():
        # Build similarity matrix
        if enhanced_system.build_enhanced_similarity_matrix():
            # Test with a medicine
            result = enhanced_system.get_safe_recommendations("PARACETAMOL", max_recommendations=3)
            formatted_output = enhanced_system.format_safe_recommendations(result)
            print(formatted_output)
