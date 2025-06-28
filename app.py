import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           roc_curve, auc, precision_score, 
                           recall_score, f1_score)
import pickle
import io
import time

# CSS optimisé pour une meilleure visibilité
st.markdown("""
<style>
    /* Variables CSS pour cohérence */
    :root {
        --primary-blue: #1e3a8a;
        --light-blue: #3b82f6;
        --accent-blue: #60a5fa;
        --bg-light: #f8fafc;
        --text-dark: #1e293b;
        --text-light: #64748b;
        --white: #ffffff;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Arrière-plan principal */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: var(--text-dark);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Titres avec meilleur contraste */
    h1, h2, h3 {
        color: var(--primary-blue) !important;
        font-weight: 700;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Boutons avec effet moderne */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-blue), var(--light-blue));
        color: var(--white);
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, var(--light-blue), var(--accent-blue));
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar moderne */
    .sidebar .sidebar-content {
        background: var(--white);
        border-right: 1px solid #e2e8f0;
        box-shadow: var(--shadow);
    }
    
    /* Cartes avec ombres */
    .stMetric {
        background: var(--white);
        border-radius: 16px;
        padding: 20px;
        box-shadow: var(--shadow);
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    
    /* Métriques avec couleurs contrastées */
    .stMetric label {
        color: var(--text-light) !important;
        font-weight: 500;
        font-size: 14px;
    }
    .stMetric [data-testid="metric-value"] {
        color: var(--primary-blue) !important;
        font-weight: 700;
        font-size: 28px;
    }
    
    /* Formulaires */
    .stForm {
        background: var(--white);
        border-radius: 16px;
        padding: 24px;
        box-shadow: var(--shadow);
        border: 1px solid #e2e8f0;
    }
    
    /* Inputs avec bordures visibles */
    .stSelectbox, .stSlider, .stNumberInput {
        background: var(--white);
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: var(--text-dark) !important;
        font-weight: 600;
        font-size: 16px;
    }
    
    /* DataFrames */
    .stDataFrame {
        background: var(--white);
        border-radius: 12px;
        box-shadow: var(--shadow);
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    /* Alertes avec couleurs vives */
    .stSuccess {
        background: linear-gradient(135deg, #10b981, #34d399);
        color: var(--white);
        border-radius: 12px;
        padding: 16px;
        font-weight: 600;
        box-shadow: var(--shadow);
    }
    .stError {
        background: linear-gradient(135deg, #ef4444, #f87171);
        color: var(--white);
        border-radius: 12px;
        padding: 16px;
        font-weight: 600;
        box-shadow: var(--shadow);
    }
    .stInfo {
        background: linear-gradient(135deg, var(--light-blue), var(--accent-blue));
        color: var(--white);
        border-radius: 12px;
        padding: 16px;
        font-weight: 600;
        box-shadow: var(--shadow);
    }
    
    /* Onglets */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--white);
        border-radius: 12px;
        padding: 8px;
        box-shadow: var(--shadow);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-dark);
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary-blue);
        color: var(--white);
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Prédiction de Prêt - Analyse Avancée", layout="wide", initial_sidebar_state="expanded")

st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🏦 Système Intelligent d'Analyse de Prêt Bancaire</h1>", unsafe_allow_html=True)

if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "🏠 Accueil"

@st.cache_data
def load_data():
    data = pd.read_csv("loan_approval_dataset.csv")
    data.columns = data.columns.str.strip()
    le = LabelEncoder()
    data['education'] = le.fit_transform(data['education'])
    data['self_employed'] = le.fit_transform(data['self_employed'])
    data['loan_status'] = le.fit_transform(data['loan_status'])
    return data, le

@st.cache_resource
def train_and_evaluate_models(data):
    X = data.drop(['loan_id', 'loan_status'], axis=1)
    y = data['loan_status']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        "Régression Logistique": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else [0]*len(y_test)
        train_time = time.time() - start_time
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "roc_auc": roc_curve(y_test, y_pred_proba) if hasattr(model, "predict_proba") else None,
            "auc": auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1]) if hasattr(model, "predict_proba") else 0,
            "train_time": train_time,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores)
        }
    
    return results, scaler, X_test, y_test

def create_modern_plot():
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('white')
    return fig, ax

def main():
    data, le = load_data()
    results, scaler, X_test, y_test = train_and_evaluate_models(data)
    
    with st.sidebar:
        st.markdown("<h2 style='color: #1e3a8a; margin-bottom: 1.5rem;'>📊 Navigation</h2>", unsafe_allow_html=True)
        nav_options = ["🏠 Accueil", "📊 Exploration des Données", "⚖️ Comparaison des Modèles", "🔮 Prédiction", "📈 Performance Détaillée"]
        for option in nav_options:
            if st.button(option, key=f"nav_{option}"):
                st.session_state.app_mode = option
    
    app_mode = st.session_state.app_mode
    
    if app_mode == "🏠 Accueil":
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: white; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;'>
            <h3 style='color: #1e3a8a; margin-bottom: 1rem;'>🚀 Solution Avancée de Machine Learning</h3>
            <p style='font-size: 18px; color: #64748b;'>
                Prédiction intelligente d'approbation des prêts bancaires avec 4 modèles performants
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### ✨ Fonctionnalités Principales
            - 📈 Analyse approfondie des données de prêt
            - 🤖 Comparaison de 4 modèles ML performants
            - ⚡ Prédictions en temps réel
            - 📊 Visualisations interactives
            - 💾 Téléchargement des modèles entraînés
            
            ### 🔧 Modèles Disponibles
            - 📊 Régression Logistique
            - 🌳 Random Forest
            - 🚀 XGBoost
            - 🎯 SVM
            """)
        with col2:
            st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=400", caption="Intelligence Artificielle Bancaire")
    
    elif app_mode == "📊 Exploration des Données":
        st.markdown("<h2>📊 Exploration des Données</h2>", unsafe_allow_html=True)
        
        tabs = st.tabs(["👀 Aperçu", "📊 Statistiques", "📈 Distributions", "✅ Approbations"])
        
        with tabs[0]:
            st.markdown("### 📋 Aperçu des Données")
            st.dataframe(data.head(), use_container_width=True)
        
        with tabs[1]:
            st.markdown("### 📊 Statistiques Descriptives")
            st.dataframe(data.describe(), use_container_width=True)
        
        with tabs[2]:
            st.markdown("### 📈 Distribution des Variables")
            col = st.selectbox("Sélectionnez une variable", data.columns.drop('loan_id'), key="dist_select")
            
            fig, ax = create_modern_plot()
            if data[col].nunique() < 10:
                sns.countplot(x=col, data=data, ax=ax, palette=['#3b82f6', '#60a5fa'])
            else:
                sns.histplot(data[col], kde=True, ax=ax, color='#3b82f6')
            ax.set_title(f'Distribution - {col}', fontsize=14, fontweight='bold', color='#1e3a8a')
            st.pyplot(fig)
        
        with tabs[3]:
            st.markdown("### ✅ Répartition des Approbations")
            fig, ax = create_modern_plot()
            sns.countplot(x='loan_status', data=data, ax=ax, palette=['#ef4444', '#10b981'])
            ax.set_xticklabels(['❌ Rejeté', '✅ Approuvé'])
            ax.set_title('Statut des Prêts', fontsize=14, fontweight='bold', color='#1e3a8a')
            st.pyplot(fig)
    
    elif app_mode == "⚖️ Comparaison des Modèles":
        st.markdown("<h2>⚖️ Comparaison des Modèles</h2>", unsafe_allow_html=True)
        
        metrics_df = pd.DataFrame({
            "🤖 Modèle": list(results.keys()),
            "🎯 Accuracy": [results[name]["accuracy"] for name in results],
            "📊 Précision": [results[name]["precision"] for name in results],
            "🔍 Rappel": [results[name]["recall"] for name in results],
            "⚖️ F1-Score": [results[name]["f1"] for name in results],
            "📈 AUC": [results[name]["auc"] for name in results],
            "⏱️ Temps (s)": [results[name]["train_time"] for name in results]
        })
        
        st.markdown("### 📊 Métriques de Performance")
        st.dataframe(metrics_df.style.format({
            "🎯 Accuracy": "{:.2%}",
            "📊 Précision": "{:.2%}",
            "🔍 Rappel": "{:.2%}",
            "⚖️ F1-Score": "{:.2%}",
            "📈 AUC": "{:.2f}",
            "⏱️ Temps (s)": "{:.3f}"
        }), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Comparaison Visuelle")
            fig, ax = create_modern_plot()
            metrics_to_plot = ["Accuracy", "Précision", "Rappel", "F1-Score"]
            x = np.arange(len(metrics_to_plot))
            width = 0.2
            
            colors = ['#1e3a8a', '#3b82f6', '#60a5fa', '#93c5fd']
            for i, (name, res) in enumerate(results.items()):
                ax.bar(x + i*width, [res["accuracy"], res["precision"], res["recall"], res["f1"]], 
                      width, label=name, alpha=0.8, color=colors[i])
            
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title('Comparaison des Métriques', fontsize=14, fontweight='bold', color='#1e3a8a')
            ax.set_xticks(x + width*1.5)
            ax.set_xticklabels(metrics_to_plot)
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📈 Courbes ROC")
            fig, ax = create_modern_plot()
            colors = ['#1e3a8a', '#3b82f6', '#60a5fa', '#93c5fd']
            for i, (name, res) in enumerate(results.items()):
                if res["roc_auc"] is not None:
                    fpr, tpr, _ = res["roc_auc"]
                    ax.plot(fpr, tpr, lw=3, label=f'{name} (AUC = {res["auc"]:.2f})', color=colors[i])
            
            ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Taux de Faux Positifs', fontweight='bold')
            ax.set_ylabel('Taux de Vrais Positifs', fontweight='bold')
            ax.set_title('Courbes ROC', fontsize=14, fontweight='bold', color='#1e3a8a')
            ax.legend()
            st.pyplot(fig)
        
        best_model_name = max(results.items(), key=lambda x: x[1]["accuracy"])[0]
        st.success(f"🏆 Meilleur modèle: {best_model_name} ({results[best_model_name]['accuracy']:.2%})")
    
    elif app_mode == "🔮 Prédiction":
        st.markdown("<h2>🔮 Prédiction en Temps Réel</h2>", unsafe_allow_html=True)
        
        with st.form(key="prediction_form"):
            model_name = st.selectbox("🤖 Sélectionnez le modèle", list(results.keys()))
            model = results[model_name]["model"]
            
            st.markdown("<h3 style='color: #1e3a8a;'>👤 Informations du Demandeur</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                no_of_dependents = st.slider("👨‍👩‍👧‍👦 Personnes à charge", 0, 5, 1)
                education = st.selectbox("🎓 Éducation", ['Graduate', 'Not Graduate'])
                self_employed = st.selectbox("💼 Travail indépendant", ['No', 'Yes'])
                income_annum = st.number_input("💰 Revenu annuel (₹)", min_value=0, max_value=100000000, value=5000000, step=100000)
                loan_amount = st.number_input("🏦 Montant du prêt (₹)", min_value=0, max_value=50000000, value=10000000, step=100000)
                loan_term = st.slider("📅 Durée du prêt (années)", 1, 20, 5)
            
            with col2:
                cibil_score = st.slider("📊 Score CIBIL", 300, 900, 700)
                residential_assets_value = st.number_input("🏠 Actifs résidentiels (₹)", min_value=0, max_value=50000000, value=5000000, step=100000)
                commercial_assets_value = st.number_input("🏢 Actifs commerciaux (₹)", min_value=0, max_value=50000000, value=5000000, step=100000)
                luxury_assets_value = st.number_input("💎 Actifs de luxe (₹)", min_value=0, max_value=50000000, value=5000000, step=100000)
                bank_asset_value = st.number_input("🏦 Actifs bancaires (₹)", min_value=0, max_value=50000000, value=5000000, step=100000)
            
            submit_button = st.form_submit_button("🚀 Lancer la Prédiction")
            
            if submit_button:
                input_data = {
                    'no_of_dependents': no_of_dependents,
                    'education': 1 if education == 'Graduate' else 0,
                    'self_employed': 1 if self_employed == 'Yes' else 0,
                    'income_annum': income_annum,
                    'loan_amount': loan_amount,
                    'loan_term': loan_term,
                    'cibil_score': cibil_score,
                    'residential_assets_value': residential_assets_value,
                    'commercial_assets_value': commercial_assets_value,
                    'luxury_assets_value': luxury_assets_value,
                    'bank_asset_value': bank_asset_value
                }
                
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled) if hasattr(model, "predict_proba") else [[0.5, 0.5]]
                
                st.markdown("### 🎯 Résultat")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if prediction[0] == 1:
                        st.success("✅ Prêt Approuvé!")
                        st.metric("🎯 Probabilité d'approbation", f"{prediction_proba[0][1]*100:.1f}%")
                    else:
                        st.error("❌ Prêt Rejeté!")
                        st.metric("🎯 Probabilité de rejet", f"{prediction_proba[0][0]*100:.1f}%")
                
                st.info(f"""
                **🤖 Modèle:** {model_name}  
                **📊 Score CIBIL:** {cibil_score}  
                **💰 Ratio Prêt/Revenu:** {(loan_amount/income_annum):.2f}  
                **🏠 Actifs totaux:** ₹{residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value:,}  
                **📅 Durée:** {loan_term} années
                """)
    
    elif app_mode == "📈 Performance Détaillée":
        st.markdown("<h2>📈 Analyse Détaillée des Performances</h2>", unsafe_allow_html=True)
        
        model_name = st.selectbox("🤖 Sélectionnez le modèle", list(results.keys()))
        model_results = results[model_name]
        
        st.markdown(f"### 📊 Métriques - {model_name}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🎯 Accuracy", f"{model_results['accuracy']:.2%}")
        col2.metric("📊 Précision", f"{model_results['precision']:.2%}")
        col3.metric("🔍 Rappel", f"{model_results['recall']:.2%}")
        col4.metric("⚖️ F1-Score", f"{model_results['f1']:.2%}")
        col5.metric("📈 AUC-ROC", f"{model_results['auc']:.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔥 Matrice de Confusion")
            fig, ax = create_modern_plot()
            sns.heatmap(model_results["confusion_matrix"], annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['❌ Rejeté', '✅ Approuvé'], yticklabels=['❌ Rejeté', '✅ Approuvé'], ax=ax)
            ax.set_xlabel('Prédit', fontweight='bold')
            ax.set_ylabel('Réel', fontweight='bold')
            ax.set_title(f'Matrice de Confusion - {model_name}', fontsize=12, fontweight='bold', color='#1e3a8a')
            st.pyplot(fig)
        
        with col2:
            if model_results["roc_auc"] is not None:
                st.markdown("### 📈 Courbe ROC")
                fpr, tpr, _ = model_results["roc_auc"]
                fig, ax = create_modern_plot()
                ax.plot(fpr, tpr, color='#3b82f6', lw=3, label=f'ROC (AUC = {model_results["auc"]:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Taux de Faux Positifs', fontweight='bold')
                ax.set_ylabel('Taux de Vrais Positifs', fontweight='bold')
                ax.set_title(f'Courbe ROC - {model_name}', fontsize=12, fontweight='bold', color='#1e3a8a')
                ax.legend()
                st.pyplot(fig)
        
        if hasattr(model_results["model"], "feature_importances_"):
            st.markdown("### 🎯 Importance des Features")
            feature_importances = pd.DataFrame({
                "Feature": data.drop(['loan_id', 'loan_status'], axis=1).columns,
                "Importance": model_results["model"].feature_importances_
            }).sort_values("Importance", ascending=False)
            
            fig, ax = create_modern_plot()
            sns.barplot(x="Importance", y="Feature", data=feature_importances, palette='viridis', ax=ax)
            ax.set_title(f'Importance des Features - {model_name}', fontsize=14, fontweight='bold', color='#1e3a8a')
            st.pyplot(fig)
        
        st.markdown("### 💾 Exportation")
        model_bytes = pickle.dumps(model_results["model"])
        st.download_button(
            label=f"📥 Télécharger {model_name}",
            data=io.BytesIO(model_bytes),
            file_name=f"modele_{model_name.lower().replace(' ', '_')}.pkl",
            mime="application/octet-stream"
        )

if __name__ == "__main__":
    main()