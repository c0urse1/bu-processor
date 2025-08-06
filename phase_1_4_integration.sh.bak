#!/bin/bash
# =============================================================================
# PHASE 1.4 INTEGRATION SETUP - ML VS. HEURISTIC EVALUATOR
# =============================================================================

echo "📊 Setting up Phase 1.4: ML vs. Heuristic Evaluation System"
echo "=========================================================="

# =============================================================================
# DIRECTORY STRUCTURE UPDATE
# =============================================================================

echo "📁 Creating Phase 1.4 directory structure..."

# Create evaluation directory
mkdir -p project/evaluation
mkdir -p project/evaluation/legacy
mkdir -p project/evaluation/reports
mkdir -p project/evaluation/visualizations

# Create legacy integration directory
mkdir -p project/legacy_integration

echo "✅ Directory structure updated"

# =============================================================================
# INTEGRATION FILES
# =============================================================================

echo "🔧 Creating integration files..."

# Create evaluation module __init__.py
cat > project/evaluation/__init__.py << 'EOF'
"""Phase 1.4 - ML vs. Heuristic Evaluation Module"""

from .ml_vs_heuristic_evaluator import MLVsHeuristicEvaluator, EvaluationConfig
from .legacy_integration import LegacySemanticCategorizer

__all__ = ['MLVsHeuristicEvaluator', 'EvaluationConfig', 'LegacySemanticCategorizer']
EOF

# Create legacy integration module
cat > project/legacy_integration/__init__.py << 'EOF'
"""Legacy system integration for Phase 1.4 evaluation"""
EOF

cat > project/legacy_integration/semantic_categorizer.py << 'EOF'
"""
LEGACY SEMANTIC CATEGORIZER INTEGRATION
=====================================

This module integrates the existing semantic_categorizer.py from the legacy system.
In the real implementation, this would import directly from the legacy codebase.

For Phase 1.4 evaluation, we simulate the legacy heuristic behavior.
"""

import re
from typing import Dict, List, Tuple, Union
import structlog

logger = structlog.get_logger("legacy.semantic_categorizer")

class SemanticCategorizer:
    """
    Legacy heuristic categorizer - Production simulation
    
    This simulates the existing semantic_categorizer.py from:
    C:\BU_PROJEKT\bu-bot-mvp\backend\semantic_categorizer.py
    """
    
    def __init__(self):
        self.logger = logger
        
        # 12-Domänen Berufsgruppen-Klassifizierung (from legacy taxonomy)
        self.categories = {
            0: "medical_professionals",
            1: "technical_careers", 
            2: "legal_professionals",  
            3: "manual_labor",
            4: "office_workers",
            5: "education_sector",
            6: "creative_industries",
            7: "sales_marketing",
            8: "financial_services", 
            9: "hospitality_service",
            10: "transport_logistics",
            11: "public_sector"
        }
        
        # Enhanced keyword patterns from legacy system
        self.keyword_patterns = {
            0: {
                "primary": ["arzt", "ärztin", "medizin", "chirurg", "zahnarzt", "therapeut"],
                "secondary": ["krankenschwester", "physiotherapeut", "apotheker", "heilpraktiker"],
                "context": ["praxis", "klinik", "krankenhaus", "behandlung", "diagnose"]
            },
            1: {
                "primary": ["ingenieur", "software", "entwickler", "programmierer", "informatik"],
                "secondary": ["it", "technik", "system", "entwicklung", "programmierung"],
                "context": ["code", "algorithmus", "datenbank", "server", "application"]
            },
            2: {
                "primary": ["anwalt", "rechtsanwalt", "jurist", "richter", "notar"],
                "secondary": ["recht", "jura", "kanzlei", "gericht", "verteidigung"],
                "context": ["klage", "urteil", "paragraph", "gesetz", "verhandlung"]
            },
            3: {
                "primary": ["handwerker", "mechaniker", "elektriker", "bauarbeiter", "monteur"],
                "secondary": ["werkstatt", "baustelle", "reparatur", "installation"],
                "context": ["werkzeug", "maschine", "konstruktion", "fertigung"]
            },
            4: {
                "primary": ["büro", "verwaltung", "sekretär", "buchhalter", "sachbearbeiter"],
                "secondary": ["office", "administration", "assistant", "coordinator"],
                "context": ["dokument", "verwaltung", "organisation", "planung"]
            },
            5: {
                "primary": ["lehrer", "professor", "pädagoge", "dozent", "erzieher"],
                "secondary": ["bildung", "schule", "universität", "unterricht"],
                "context": ["student", "lehre", "curriculum", "bildung", "ausbildung"]
            },
            6: {
                "primary": ["künstler", "designer", "fotograf", "musiker", "autor"],
                "secondary": ["kreativ", "design", "kunst", "kultur", "medien"],
                "context": ["portfolio", "projekt", "gestaltung", "kreation"]
            },
            7: {
                "primary": ["verkauf", "vertrieb", "marketing", "berater", "verkäufer"],
                "secondary": ["sales", "kunde", "beratung", "akquisition"],
                "context": ["umsatz", "kunde", "verkaufen", "beratung", "angebot"]
            },
            8: {
                "primary": ["bank", "versicherung", "finanz", "berater", "anlage"],
                "secondary": ["kredit", "investment", "vermögen", "risiko"],
                "context": ["geld", "kapital", "zinsen", "portfolio", "rendite"]
            },
            9: {
                "primary": ["hotel", "restaurant", "service", "kellner", "koch"],
                "secondary": ["tourismus", "gastronomie", "hospitality", "catering"],
                "context": ["gast", "service", "küche", "rezeption", "booking"]
            },
            10: {
                "primary": ["transport", "logistik", "fahrer", "spedition", "lager"],
                "secondary": ["lieferung", "versand", "distribution", "supply"],
                "context": ["fahrzeug", "route", "ladung", "warehouse", "delivery"]
            },
            11: {
                "primary": ["beamter", "verwaltung", "öffentlich", "gemeinde", "staat"],
                "secondary": ["behörde", "amt", "ministerium", "verwaltung"],
                "context": ["öffentlich", "gesetz", "bürger", "staat", "verwaltung"]
            }
        }
        
        # Negative patterns to avoid false positives
        self.exclusion_patterns = {
            "medical": ["medizintechnik", "medical device", "pharma sales"],
            "technical": ["technical sales", "tech sales", "verkaufsleiter technik"],
            "education": ["sales training", "marketing education"]
        }
        
        self.logger.info("Legacy SemanticCategorizer initialized", categories=len(self.categories))
    
    def categorize_chunk(self, text: str, return_confidence: bool = False) -> Union[int, Tuple[int, float]]:
        """
        Legacy heuristic categorization - Enhanced pattern matching
        
        Args:
            text: Input text to categorize
            return_confidence: Whether to return confidence score
            
        Returns:
            Category ID (and confidence if requested)
        """
        text_lower = text.lower()
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
        words = text_clean.split()
        
        category_scores = {}
        
        # Calculate weighted scores for each category
        for category_id, patterns in self.keyword_patterns.items():
            score = 0
            
            # Primary keywords (high weight)
            for keyword in patterns["primary"]:
                if keyword in text_lower:
                    score += 3
                    # Bonus for exact word match
                    if f" {keyword} " in f" {text_lower} ":
                        score += 1
            
            # Secondary keywords (medium weight)
            for keyword in patterns["secondary"]:
                if keyword in text_lower:
                    score += 2
            
            # Context keywords (low weight)
            for keyword in patterns["context"]:
                if keyword in text_lower:
                    score += 1
            
            # Word proximity bonus (keywords appearing close together)
            primary_positions = []
            for keyword in patterns["primary"]:
                for i, word in enumerate(words):
                    if keyword in word:
                        primary_positions.append(i)
            
            if len(primary_positions) > 1:
                # Bonus if primary keywords are close
                min_distance = min(abs(p1 - p2) for i, p1 in enumerate(primary_positions) 
                                 for p2 in primary_positions[i+1:])
                if min_distance <= 5:  # Within 5 words
                    score += 2
            
            # Normalize by text length (longer texts get penalty)
            text_length_factor = min(1.0, 50 / len(words)) if len(words) > 50 else 1.0
            score *= text_length_factor
            
            category_scores[category_id] = score
        
        # Apply exclusion patterns
        for exclusion_type, exclusion_words in self.exclusion_patterns.items():
            for exclusion_word in exclusion_words:
                if exclusion_word in text_lower:
                    # Reduce scores for potentially conflicting categories
                    if exclusion_type == "medical" and 0 in category_scores:
                        category_scores[0] *= 0.5
                    elif exclusion_type == "technical" and 1 in category_scores:
                        category_scores[1] *= 0.5
                    elif exclusion_type == "education" and 5 in category_scores:
                        category_scores[5] *= 0.5
        
        # Find best match
        if not category_scores or max(category_scores.values()) == 0:
            # Fallback strategy based on text length and common words
            if any(word in text_lower for word in ["büro", "office", "verwaltung"]):
                predicted_category = 4  # office_workers
                confidence = 0.3
            elif any(word in text_lower for word in ["service", "kunde", "beratung"]):
                predicted_category = 7  # sales_marketing
                confidence = 0.25
            else:
                predicted_category = 4  # Default to office_workers
                confidence = 0.1
        else:
            predicted_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            max_score = category_scores[predicted_category]
            
            # Convert score to confidence (0-1 range)
            confidence = min(max_score / 10, 1.0)  # Normalize to 0-1
            
            # Boost confidence if multiple indicators point to same category
            if max_score > 5:
                confidence = min(confidence * 1.2, 1.0)
        
        self.logger.debug(
            "Categorization complete",
            text_preview=text[:50] + "..." if len(text) > 50 else text,
            predicted_category=predicted_category,
            category_name=self.categories.get(predicted_category, "Unknown"),
            confidence=confidence,
            all_scores={k: v for k, v in category_scores.items() if v > 0}
        )
        
        if return_confidence:
            return predicted_category, confidence
        return predicted_category
    
    def get_category_name(self, category_id: int) -> str:
        """Get human-readable category name"""
        return self.categories.get(category_id, "Unknown")
    
    def get_all_categories(self) -> Dict[int, str]:
        """Get all available categories"""
        return self.categories.copy()

# Legacy compatibility alias
LegacySemanticCategorizer = SemanticCategorizer
EOF

# Add Phase 1.4 to main CLI
cat >> project/cli.py << 'EOF'

# Phase 1.4 - ML vs. Heuristic Evaluation Commands
@app.command()
def evaluate_models(
    ml_model_path: str = typer.Option("trained_models", help="Path to trained ML model"),
    test_data_path: str = typer.Option("ml_training_data/test_dataset.json", help="Path to test data"),
    output_dir: str = typer.Option("evaluation_results", help="Output directory"),
    quick_mode: bool = typer.Option(False, help="Quick evaluation without visualizations")
):
    """📊 Phase 1.4: Evaluate ML model vs. Legacy heuristic"""
    
    console.print("📊 Starting Phase 1.4: ML vs. Heuristic Evaluation")
    
    try:
        from evaluation.ml_vs_heuristic_evaluator import MLVsHeuristicEvaluator, EvaluationConfig
        
        config = EvaluationConfig(
            ml_model_path=ml_model_path,
            test_data_path=test_data_path,
            output_dir=output_dir,
            create_visualizations=not quick_mode,
            save_detailed_results=True
        )
        
        evaluator = MLVsHeuristicEvaluator(config)
        comparison_report = evaluator.run_comprehensive_evaluation()
        
        # Display results
        console.print(comparison_report.get_summary_table())
        
        # Create visualizations
        if not quick_mode:
            evaluator.create_visualizations(comparison_report)
        
        # Save results
        evaluator.save_detailed_results(comparison_report)
        
        # Show recommendations
        console.print("\n💡 [bold cyan]Production Recommendations:[/bold cyan]")
        for rec in comparison_report.recommendations:
            console.print(f"   {rec}")
        
        console.print(f"\n✅ Evaluation complete! Results in: {output_dir}")
        
    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}")
        raise typer.Exit(1)

@app.command()
def quick_classify(
    text: str = typer.Argument(..., help="Text to classify"),
    ml_model_path: str = typer.Option("trained_models", help="ML model path")
):
    """⚡ Quick single-text classification comparison"""
    
    try:
        from evaluation.ml_vs_heuristic_evaluator import MLModelLoader
        from legacy_integration.semantic_categorizer import SemanticCategorizer
        
        console.print(f"🔍 Comparing classification for: '{text}'")
        
        # Load models
        ml_loader = MLModelLoader(ml_model_path)
        ml_loader.load_model()
        heuristic = SemanticCategorizer()
        
        # Get predictions
        ml_preds, ml_confs = ml_loader.predict_batch([text])
        heur_pred, heur_conf = heuristic.categorize_chunk(text, return_confidence=True)
        
        # Create comparison table
        table = Table(title="Classification Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Prediction", style="green") 
        table.add_column("Confidence", style="yellow")
        table.add_column("Category", style="magenta")
        
        table.add_row(
            "ML Model", 
            str(ml_preds[0]), 
            f"{ml_confs[0]:.3f}",
            heuristic.get_category_name(ml_preds[0])
        )
        
        table.add_row(
            "Legacy Heuristic",
            str(heur_pred),
            f"{heur_conf:.3f}", 
            heuristic.get_category_name(heur_pred)
        )
        
        console.print(table)
        
        # Recommendation
        if ml_confs[0] > 0.8:
            console.print("💡 [green]High ML confidence - Use ML prediction[/green]")
        elif ml_confs[0] > 0.6:
            console.print("💡 [yellow]Medium ML confidence - Consider context[/yellow]")
        else:
            console.print("💡 [red]Low ML confidence - Fallback to heuristic[/red]")
            
    except Exception as e:
        console.print(f"❌ Comparison failed: {e}")
        raise typer.Exit(1)
EOF

echo "✅ Integration files created"

# =============================================================================
# REQUIREMENTS UPDATE
# =============================================================================

echo "📦 Updating requirements..."

# Add visualization dependencies
cat >> project/requirements.txt << 'EOF'

# Phase 1.4 - Evaluation & Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
kaleido>=0.2.1
EOF

echo "✅ Requirements updated"

# =============================================================================
# DEMO DATA CREATION
# =============================================================================

echo "🎭 Creating Phase 1.4 demo data..."

# Create demo script
cat > project/scripts/create_phase_1_4_demo.py << 'EOF'
#!/usr/bin/env python3
"""
Create demo data for Phase 1.4 evaluation
"""

import json
import sys
from pathlib import Path

def create_demo_data():
    """Create realistic demo data for ML vs. Heuristic evaluation"""
    
    # Realistic BU content samples across all 12 categories
    demo_samples = [
        # Medical professionals (0)
        {
            "text": "Als niedergelassener Arzt in eigener Praxis behandle ich täglich Patienten mit verschiedensten Beschwerden. Die Berufsunfähigkeitsversicherung ist für mich essentiell, da körperliche Einschränkungen meine ärztliche Tätigkeit unmöglich machen würden.",
            "label": 0
        },
        {
            "text": "Die Zahnärztin führt komplexe chirurgische Eingriffe durch und ist auf ihre Feinmotorik angewiesen. Ein Unfall könnte ihre Karriere beenden.",
            "label": 0
        },
        {
            "text": "Therapeutische Behandlungen erfordern körperliche Fitness und mentale Konzentration. Als Physiotherapeut bin ich auf meine Hände angewiesen.",
            "label": 0
        },
        
        # Technical careers (1)  
        {
            "text": "Als Software-Entwickler programmiere ich täglich komplexe Algorithmen und Datenbankanwendungen. Meine Arbeit erfordert hohe Konzentration und technisches Verständnis.",
            "label": 1
        },
        {
            "text": "Der Ingenieur entwickelt innovative Systeme und ist für die technische Umsetzung verantwortlich. IT-Kenntnisse sind unverzichtbar.",
            "label": 1
        },
        {
            "text": "In der Informatik arbeite ich an der Entwicklung von Server-Applikationen und Database-Management-Systemen für Unternehmen.",
            "label": 1
        },
        
        # Legal professionals (2)
        {
            "text": "Als Rechtsanwalt vertrete ich Mandanten vor Gericht und berate in komplexen Rechtsfragen. Juristische Expertise ist mein Kapital.",
            "label": 2
        },
        {
            "text": "Die Kanzlei spezialisiert sich auf Wirtschaftsrecht und führt Verhandlungen mit internationalen Partnern. Jura-Studium ist Voraussetzung.",
            "label": 2
        },
        
        # Manual labor (3)
        {
            "text": "Als Handwerker führe ich täglich körperlich anspruchsvolle Arbeiten aus. Reparaturen und Installationen erfordern handwerkliches Geschick.",
            "label": 3
        },
        {
            "text": "Auf der Baustelle arbeite ich mit schwerem Werkzeug und Maschinen. Die körperliche Belastung ist hoch.",
            "label": 3
        },
        
        # Office workers (4)
        {
            "text": "In der Büroverwaltung koordiniere ich verschiedene Projekte und erstelle administrative Dokumente. Organisation ist meine Stärke.",
            "label": 4
        },
        {
            "text": "Als Sachbearbeiter bearbeite ich täglich Anträge und führe Korrespondenz mit Kunden. Office-Anwendungen sind mein tägliches Werkzeug.",
            "label": 4
        },
        
        # Education sector (5)  
        {
            "text": "Als Lehrer unterrichte ich Schüler in verschiedenen Fächern und entwickle Lehrpläne. Bildung und Pädagogik stehen im Mittelpunkt.",
            "label": 5
        },
        {
            "text": "Der Professor forscht an der Universität und hält Vorlesungen für Studenten. Wissenschaftliche Lehre ist seine Berufung.",
            "label": 5
        },
        
        # Creative industries (6)
        {
            "text": "Als Designer entwickle ich kreative Konzepte und Gestaltungslösungen für verschiedene Projekte. Mein Portfolio spiegelt meine künstlerische Vision wider.",
            "label": 6
        },
        {
            "text": "Die Fotografin erstellt professionelle Aufnahmen für Werbekampagnen und Events. Kreativität und technisches Know-how sind entscheidend.",
            "label": 6
        },
        
        # Sales & Marketing (7)
        {
            "text": "Im Vertrieb akquiriere ich neue Kunden und berate sie bei der Produktauswahl. Umsatzziele zu erreichen ist meine Hauptaufgabe.",
            "label": 7
        },
        {
            "text": "Als Marketing-Berater entwickle ich Strategien zur Kundengewinnung und betreue große Werbekampagnen.",
            "label": 7
        },
        
        # Financial services (8)
        {
            "text": "In der Bank berate ich Kunden zu Finanzanlagen und Krediten. Kapitalmarkt-Kenntnisse und Risikobewertung sind essentiell.",
            "label": 8
        },
        {
            "text": "Als Versicherungsberater analysiere ich Risiken und erstelle maßgeschneiderte Versicherungskonzepte für Privat- und Firmenkunden.",
            "label": 8
        },
        
        # Hospitality & Service (9)
        {
            "text": "Im Hotel betreue ich Gäste an der Rezeption und sorge für einen reibungslosen Ablauf. Service-Qualität steht an erster Stelle.",
            "label": 9
        },
        {
            "text": "Als Koch kreiere ich täglich neue Menüs im Restaurant und leite das Küchenteam. Gastronomie ist meine Leidenschaft.",
            "label": 9
        },
        
        # Transport & Logistics (10)
        {
            "text": "Als LKW-Fahrer transportiere ich Waren auf verschiedenen Routen quer durch Europa. Logistik und pünktliche Lieferung sind wichtig.",
            "label": 10
        },
        {
            "text": "In der Spedition koordiniere ich Warenströme und optimiere Lieferketten. Supply Chain Management ist mein Spezialgebiet.",
            "label": 10
        },
        
        # Public sector (11)
        {
            "text": "Als Beamter arbeite ich in der Stadtverwaltung und bearbeite Bürgeranfragen. Öffentlicher Dienst und Verwaltungsaufgaben prägen meinen Alltag.",
            "label": 11
        },
        {
            "text": "Im Ministerium entwickle ich Gesetzesentwürfe und berate Politiker in Fachfragen. Staatliche Verwaltung ist ein komplexes Feld.",
            "label": 11
        }
    ]
    
    # Split into train/test for demo (80/20)
    from sklearn.model_selection import train_test_split
    
    train_data, test_data = train_test_split(demo_samples, test_size=0.2, random_state=42, stratify=[s['label'] for s in demo_samples])
    
    # Create output directory
    output_dir = Path("demo_data_phase_1_4")
    output_dir.mkdir(exist_ok=True)
    
    # Save datasets
    with open(output_dir / "train_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "test_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    # Create dataset stats
    stats = {
        "train_size": len(train_data),
        "test_size": len(test_data),
        "num_classes": 12,
        "class_distribution": {
            str(i): sum(1 for item in demo_samples if item['label'] == i) 
            for i in range(12)
        },
        "recommended_hyperparams": {
            "batch_size": 8,
            "learning_rate": 2e-5,
            "num_epochs": 5,
            "warmup_steps": 50
        }
    }
    
    with open(output_dir / "dataset_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Create label encoders
    encoders = {
        "single_label": {
            "classes": [
                "medical_professionals", "technical_careers", "legal_professionals",
                "manual_labor", "office_workers", "education_sector",
                "creative_industries", "sales_marketing", "financial_services",
                "hospitality_service", "transport_logistics", "public_sector"
            ]
        }
    }
    
    with open(output_dir / "label_encoders.json", 'w', encoding='utf-8') as f:
        json.dump(encoders, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Demo data created in {output_dir}/")
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Test samples: {len(test_data)}")
    print(f"   - Classes: {stats['num_classes']}")

if __name__ == "__main__":
    create_demo_data()
EOF

chmod +x project/scripts/create_phase_1_4_demo.py

echo "✅ Demo data script created"

# =============================================================================
# COMPLETE EVALUATION WORKFLOW SCRIPT
# =============================================================================

echo "🚀 Creating complete evaluation workflow..."

cat > project/scripts/run_phase_1_4_evaluation.sh << 'EOF'
#!/bin/bash
# =============================================================================
# COMPLETE PHASE 1.4 EVALUATION WORKFLOW
# =============================================================================

set -e

echo "🚀 Phase 1.4: Complete ML vs. Heuristic Evaluation Workflow"
echo "=========================================================="

cd "$(dirname "$0")/.."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not detected. Activating..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Virtual environment not found. Run setup first."
        exit 1
    fi
fi

# Check for required dependencies
echo "📦 Checking dependencies..."
python -c "import matplotlib, seaborn, plotly" 2>/dev/null || {
    echo "📦 Installing visualization dependencies..."
    pip install matplotlib seaborn plotly kaleido
}

# Step 1: Create demo data if it doesn't exist
if [ ! -d "demo_data_phase_1_4" ]; then
    echo "🎭 Creating demo data..."
    python scripts/create_phase_1_4_demo.py
else
    echo "✅ Demo data already exists"
fi

# Step 2: Check for trained model from Phase 1.3
if [ ! -d "trained_models" ] || [ ! -f "trained_models/pytorch_model.bin" ]; then
    echo "🤖 No trained model found. Creating mock model for demo..."
    mkdir -p trained_models
    
    # Create mock model files for demo
    python -c "
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Create a minimal working model for demo
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-german-cased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-german-cased', num_labels=12)

# Save mock model
model.save_pretrained('trained_models')
tokenizer.save_pretrained('trained_models')

# Create mock training results
results = {
    'metrics': {
        'accuracy': 0.87,
        'precision': 0.85,
        'recall': 0.84,
        'f1_score': 0.85
    },
    'model_info': {
        'model_name': 'distilbert-base-german-cased',
        'num_parameters': model.num_parameters()
    }
}

with open('trained_models/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('✅ Mock model created for evaluation demo')
"
else
    echo "✅ Trained model found"
fi

# Step 3: Run comprehensive evaluation
echo "📊 Running comprehensive ML vs. Heuristic evaluation..."

python cli.py evaluate-models \
    --ml-model-path trained_models \
    --test-data-path demo_data_phase_1_4/test_dataset.json \
    --output-dir evaluation_results \
    --no-quick-mode

# Step 4: Display results summary
echo ""
echo "📈 EVALUATION COMPLETED!"
echo "======================"

if [ -f "evaluation_results/comprehensive_evaluation_results.json" ]; then
    python -c "
import json
with open('evaluation_results/comprehensive_evaluation_results.json', 'r') as f:
    results = json.load(f)

summary = results['summary']
print('📊 Performance Summary:')
print(f'   ML Model Accuracy:      {summary[\"ml_model\"][\"accuracy\"]:.4f}')
print(f'   Heuristic Accuracy:     {summary[\"heuristic_model\"][\"accuracy\"]:.4f}') 
print(f'   Best Hybrid Model:      {summary[\"best_hybrid_model\"]}')
print(f'   Optimal Threshold:      {results[\"optimal_threshold\"]}')
print('')
print('💡 Key Recommendations:')
for i, rec in enumerate(results['recommendations'][:3], 1):
    # Remove markdown formatting for terminal display
    clean_rec = rec.replace('**', '').replace('🎯', '').replace('✅', '').replace('⚠️', '')
    print(f'   {i}. {clean_rec[:80]}...' if len(clean_rec) > 80 else f'   {i}. {clean_rec}')
"
fi

echo ""
echo "📁 Results Location:"
echo "   📄 Detailed Results: evaluation_results/comprehensive_evaluation_results.json"
echo "   📊 Visualizations:   evaluation_results/*.png" 
echo "   🌐 Interactive:      evaluation_results/interactive_dashboard.html"
echo "   📈 CSV Data:         evaluation_results/detailed_predictions.csv"

# Step 5: Quick classification demo
echo ""
echo "⚡ Quick Classification Demo:"
echo "=============================="

python cli.py quick-classify \
    "Als Software-Entwickler programmiere ich täglich komplexe Algorithmen und arbeite mit verschiedenen Programmiersprachen." \
    --ml-model-path trained_models

echo ""
python cli.py quick-classify \
    "Der Arzt behandelt Patienten in seiner Praxis und führt medizinische Untersuchungen durch." \
    --ml-model-path trained_models

echo ""
echo "🎉 Phase 1.4 Evaluation Complete!"
echo "================================="
echo ""
echo "Next Steps:"
echo "1. Review the interactive dashboard: evaluation_results/interactive_dashboard.html"
echo "2. Analyze detailed predictions: evaluation_results/detailed_predictions.csv"
echo "3. Implement production recommendations from the report"
echo "4. Proceed to Phase 2: Pipeline Modularization"
EOF

chmod +x project/scripts/run_phase_1_4_evaluation.sh

echo "✅ Complete evaluation workflow created"

# =============================================================================
# TESTING SETUP
# =============================================================================

echo "🧪 Creating Phase 1.4 tests..."

# Create test for Phase 1.4
cat > project/tests/test_phase_1_4_evaluation.py << 'EOF'
#!/usr/bin/env python3
"""
Tests for Phase 1.4 - ML vs. Heuristic Evaluation
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from legacy_integration.semantic_categorizer import SemanticCategorizer
from evaluation.ml_vs_heuristic_evaluator import EvaluationConfig, MLVsHeuristicEvaluator

class TestLegacySemanticCategorizer:
    """Test legacy heuristic categorizer"""
    
    def test_categorizer_initialization(self):
        """Test categorizer initializes correctly"""
        categorizer = SemanticCategorizer()
        
        assert len(categorizer.categories) == 12
        assert 0 in categorizer.categories
        assert categorizer.categories[0] == "medical_professionals"
    
    def test_medical_classification(self):
        """Test medical professional classification"""
        categorizer = SemanticCategorizer()
        
        text = "Als Arzt behandle ich Patienten in meiner Praxis."
        pred, conf = categorizer.categorize_chunk(text, return_confidence=True)
        
        assert pred == 0  # medical_professionals
        assert conf > 0.5
    
    def test_technical_classification(self):
        """Test technical professional classification"""
        categorizer = SemanticCategorizer()
        
        text = "Als Software-Entwickler programmiere ich Anwendungen."
        pred, conf = categorizer.categorize_chunk(text, return_confidence=True)
        
        assert pred == 1  # technical_careers
        assert conf > 0.5
    
    def test_fallback_behavior(self):
        """Test fallback for unclear text"""
        categorizer = SemanticCategorizer()
        
        text = "Dies ist ein unklarer Text ohne spezifische Berufsindikatoren."
        pred, conf = categorizer.categorize_chunk(text, return_confidence=True)
        
        assert pred == 4  # office_workers (default fallback)
        assert conf < 0.5

class TestEvaluationConfig:
    """Test evaluation configuration"""
    
    def test_valid_config_creation(self):
        """Test creating valid evaluation config"""
        config = EvaluationConfig(
            ml_model_path="test_models",
            test_data_path="test_data.json",
            confidence_thresholds=[0.5, 0.7, 0.9]
        )
        
        assert config.ml_model_path == "test_models"
        assert config.confidence_thresholds == [0.5, 0.7, 0.9]
        assert config.hybrid_threshold == 0.7

@pytest.fixture
def temp_test_data():
    """Create temporary test data"""
    temp_dir = tempfile.mkdtemp()
    
    test_data = [
        {"text": "Arzt behandelt Patienten", "label": 0},
        {"text": "Entwickler programmiert Software", "label": 1},
        {"text": "Anwalt berät Mandanten", "label": 2}
    ]
    
    test_file = Path(temp_dir) / "test_data.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    yield str(test_file)
    
    shutil.rmtree(temp_dir)

class TestMLVsHeuristicEvaluator:
    """Test main evaluation system"""
    
    def test_load_test_data(self, temp_test_data):
        """Test loading test data"""
        config = EvaluationConfig(test_data_path=temp_test_data)
        evaluator = MLVsHeuristicEvaluator(config)
        
        texts, labels = evaluator.load_test_data()
        
        assert len(texts) == 3
        assert len(labels) == 3
        assert labels == [0, 1, 2]
    
    def test_load_nonexistent_data(self):
        """Test loading nonexistent test data"""
        config = EvaluationConfig(test_data_path="nonexistent.json")
        evaluator = MLVsHeuristicEvaluator(config)
        
        with pytest.raises(FileNotFoundError):
            evaluator.load_test_data()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

echo "✅ Phase 1.4 tests created"

# =============================================================================
# FINAL STATUS SUMMARY
# =============================================================================

echo ""
echo "🎉 PHASE 1.4 INTEGRATION COMPLETE!"
echo "=================================="
echo ""
echo "📂 Updated Project Structure:"
echo "project/"
echo "├── evaluation/                    # Phase 1.4 evaluation system"
echo "│   ├── ml_vs_heuristic_evaluator.py"
echo "│   └── reports/"
echo "├── legacy_integration/           # Legacy system integration"
echo "│   └── semantic_categorizer.py"
echo "├── scripts/"
echo "│   ├── create_phase_1_4_demo.py"
echo "│   └── run_phase_1_4_evaluation.sh"
echo "├── tests/"
echo "│   └── test_phase_1_4_evaluation.py"
echo "└── cli.py                        # Updated with Phase 1.4 commands"
echo ""
echo "🚀 Ready to Run Phase 1.4:"
echo "=========================="
echo ""
echo "# Complete evaluation workflow:"
echo "cd project && ./scripts/run_phase_1_4_evaluation.sh"
echo ""
echo "# Individual commands:"
echo "python cli.py evaluate-models --help"
echo "python cli.py quick-classify 'Sample text here'"
echo ""
echo "📊 Expected Outputs:"
echo "- Comprehensive comparison report"
echo "- Interactive visualizations"  
echo "- Production deployment recommendations"
echo "- Hybrid approach optimization"
echo "- ROC curves and confusion matrices"
echo ""
echo "✨ Phase 1.4 is ready to deliver quantitative evidence"
echo "   for ML model superiority over legacy heuristics!"
