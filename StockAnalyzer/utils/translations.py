"""
Translation module for multilingual support in FinVision application.
"""

# Dictionary of supported languages with translations
translations = {
    "en": {  # English
        "app_title": "FinVision - Stock Market Analysis",
        "app_subtitle": "Advanced Stock Market Analysis",
        "nav_dashboard": "Dashboard",
        "nav_stock_analysis": "Stock Analysis",
        "nav_portfolio": "Portfolio",
        "nav_portfolio_analysis": "Portfolio Analysis",
        "nav_market_overview": "Market Overview",
        "nav_advanced_tools": "Advanced Tools",
        "nav_charts": "Advanced Charts",
        "nav_portfolio_tool": "Portfolio Analyzer",
        "nav_portfolio_balance": "Portfolio Balance",
        "nav_real_time_prices": "Real-Time Prices",
        "nav_backtest": "Backtest",
        "portfolio_balance_title": "Portfolio Balance",
        "dashboard_title": "Financial Dashboard",
        "stock_analysis_title": "Detailed Analysis",
        "portfolio_title": "Portfolio Tracker",
        "portfolio_analysis_title": "AI Portfolio Analysis",
        "market_overview_title": "Market Overview",
        "backtest_title": "Portfolio Backtest",
        # Portfolio section
        "portfolio_overview": "Your Portfolio",
        "portfolio_value": "Total Portfolio Value",
        "total_investment": "Total Investment",
        "total_return": "Total Return",
        "portfolio_empty": "Your portfolio is empty. Add some stocks to get started.",
        "add_stock": "Add Stock",
        "stock_symbol": "Stock Symbol",
        "shares": "Number of Shares",
        "avg_price": "Average Purchase Price ($)",
        "add_button": "Add/Update Stock",
        "remove_stock": "Remove Stock",
        "select_stock": "Select Stock to Remove",
        "remove_all": "Remove all shares",
        "remove_button": "Remove Stock",
        "portfolio_allocation": "Portfolio Allocation",
        "portfolio_performance": "Portfolio Performance (Last Month)",
        # Import portfolio section
        "import_portfolio": "Import Portfolio from CSV",
        "upload_csv": "Upload a CSV file with your portfolio data. The file should include columns for Symbol, Shares, and Average Purchase Price.",
        "expected_format": "Expected CSV format:",
        "choose_csv": "Choose a CSV file",
        "replace_portfolio": "Replace my entire portfolio",
        "add_to_portfolio": "Add to my existing portfolio",
        "import_button": "Import Portfolio",
        # Analysis section
        "portfolio_overview_analysis": "Portfolio Overview", 
        "ai_risk_assessment": "AI Risk Assessment",
        "portfolio_stability": "Portfolio Stability",
        "ai_improvement": "AI Improvement Suggestions",
        "future_prediction": "Future Performance Prediction",
        "select_days": "Select days for prediction",
        "prediction_summary": "Prediction Summary (based on historical performance):",
        "current_value": "Current portfolio value",
        "predicted_value": "Predicted value on",
        "predicted_return": "Predicted return over",
        "prediction_confidence": "Prediction confidence interval",
        "prediction_note": "Note: This prediction is based on historical volatility and returns. Actual market performance may vary significantly.",
        "portfolio_optimization": "Portfolio Optimization",
        "risk_reduction": "Risk Reduction",
        "return_maximization": "Return Maximization",
        # Footer
        "footer": "FinVision - Advanced Stock Market Analysis Platform",
        # Real-time prices section
        "Real-Time Prices": "Real-Time Prices",
        "Symbol": "Symbol",
        "Price": "Price",
        "Change": "Change",
        "Last Update": "Last Update",
        "Update Interval (seconds)": "Update Interval (seconds)",
        "Auto-updating every": "Auto-updating every",
        "seconds": "seconds",
        "Note: Prices are fetched from Yahoo Finance API and may be delayed by a few minutes for some exchanges.": "Note: Prices are fetched from Yahoo Finance API and may be delayed by a few minutes for some exchanges.",
        "About Real-Time Updates": "About Real-Time Updates",
        "This table automatically updates every": "This table automatically updates every",
        "Trend": "Trend"
    },
    "it": {  # Italian
        "app_title": "FinVision - Analisi del Mercato Azionario",
        "app_subtitle": "Analisi Avanzata del Mercato Azionario",
        "nav_dashboard": "Dashboard",
        "nav_stock_analysis": "Analisi Titoli",
        "nav_portfolio": "Portafoglio",
        "nav_portfolio_analysis": "Analisi Portafoglio",
        "nav_market_overview": "Panoramica Mercato",
        "nav_advanced_tools": "Strumenti Avanzati",
        "nav_charts": "Grafici Avanzati",
        "nav_portfolio_tool": "Analizzatore Portafoglio",
        "nav_portfolio_balance": "Bilanciamento Portafoglio",
        "nav_real_time_prices": "Prezzi in Tempo Reale",
        "nav_backtest": "Backtest",
        "portfolio_balance_title": "Bilanciamento del Portafoglio",
        "dashboard_title": "Dashboard Finanziaria",
        "stock_analysis_title": "Analisi Dettagliata",
        "portfolio_title": "Gestione Portafoglio",
        "portfolio_analysis_title": "Analisi Portafoglio con IA",
        "market_overview_title": "Panoramica del Mercato",
        "backtest_title": "Backtest Portafoglio",
        # Portfolio section
        "portfolio_overview": "Il Tuo Portafoglio",
        "portfolio_value": "Valore Totale Portafoglio",
        "total_investment": "Investimento Totale",
        "total_return": "Rendimento Totale",
        "portfolio_empty": "Il tuo portafoglio è vuoto. Aggiungi azioni per iniziare.",
        "add_stock": "Aggiungi Azione",
        "stock_symbol": "Simbolo Azione",
        "shares": "Numero di Azioni",
        "avg_price": "Prezzo Medio di Acquisto ($)",
        "add_button": "Aggiungi/Aggiorna Azione",
        "remove_stock": "Rimuovi Azione",
        "select_stock": "Seleziona Azione da Rimuovere",
        "remove_all": "Rimuovi tutte le azioni",
        "remove_button": "Rimuovi Azione",
        "portfolio_allocation": "Allocazione Portafoglio",
        "portfolio_performance": "Performance Portafoglio (Ultimo Mese)",
        # Import portfolio section
        "import_portfolio": "Importa Portafoglio da CSV",
        "upload_csv": "Carica un file CSV con i dati del tuo portafoglio. Il file deve includere colonne per Simbolo, Numero di Azioni e Prezzo Medio di Acquisto.",
        "expected_format": "Formato CSV previsto:",
        "choose_csv": "Scegli un file CSV",
        "replace_portfolio": "Sostituisci tutto il portafoglio",
        "add_to_portfolio": "Aggiungi al portafoglio esistente",
        "import_button": "Importa Portafoglio",
        # Analysis section
        "portfolio_overview_analysis": "Panoramica Portafoglio", 
        "ai_risk_assessment": "Valutazione Rischio IA",
        "portfolio_stability": "Stabilità Portafoglio",
        "ai_improvement": "Suggerimenti di Miglioramento IA",
        "future_prediction": "Previsione Performance Futura",
        "select_days": "Seleziona giorni per la previsione",
        "prediction_summary": "Riepilogo Previsione (basato su performance storica):",
        "current_value": "Valore attuale portafoglio",
        "predicted_value": "Valore previsto il",
        "predicted_return": "Rendimento previsto in",
        "prediction_confidence": "Intervallo di confidenza previsione",
        "prediction_note": "Nota: Questa previsione è basata su volatilità e rendimenti storici. La performance effettiva del mercato potrebbe variare significativamente.",
        "portfolio_optimization": "Ottimizzazione Portafoglio",
        "risk_reduction": "Riduzione del Rischio",
        "return_maximization": "Massimizzazione del Rendimento",
        # Footer
        "footer": "FinVision - Piattaforma Avanzata di Analisi del Mercato Azionario",
        # Real-time prices section
        "Real-Time Prices": "Prezzi in Tempo Reale",
        "Symbol": "Simbolo",
        "Price": "Prezzo",
        "Change": "Variazione",
        "Last Update": "Ultimo Aggiornamento",
        "Update Interval (seconds)": "Intervallo di Aggiornamento (secondi)",
        "Auto-updating every": "Aggiornamento automatico ogni",
        "seconds": "secondi",
        "Note: Prices are fetched from Yahoo Finance API and may be delayed by a few minutes for some exchanges.": "Nota: I prezzi sono recuperati dall'API di Yahoo Finance e potrebbero essere ritardati di alcuni minuti per alcuni mercati.",
        "About Real-Time Updates": "Informazioni sugli Aggiornamenti in Tempo Reale",
        "This table automatically updates every": "Questa tabella si aggiorna automaticamente ogni",
        "Trend": "Tendenza"
    },
    "es": {  # Spanish
        "app_title": "FinVision - Análisis del Mercado Bursátil",
        "app_subtitle": "Análisis Avanzado del Mercado Bursátil",
        "nav_dashboard": "Panel Principal",
        "nav_stock_analysis": "Análisis de Acciones",
        "nav_portfolio": "Portafolio",
        "nav_portfolio_analysis": "Análisis de Portafolio",
        "nav_market_overview": "Visión del Mercado",
        "nav_advanced_tools": "Herramientas Avanzadas",
        "nav_charts": "Gráficos Avanzados",
        "nav_portfolio_tool": "Analizador de Cartera",
        "nav_portfolio_balance": "Equilibrio de Cartera",
        "nav_real_time_prices": "Precios en Tiempo Real",
        "nav_backtest": "Backtest",
        "portfolio_balance_title": "Equilibrio de Cartera",
        "dashboard_title": "Panel Financiero",
        "stock_analysis_title": "Análisis Detallado",
        "portfolio_title": "Gestor de Portafolio",
        "portfolio_analysis_title": "Análisis de Portafolio con IA",
        "market_overview_title": "Visión General del Mercado",
        "backtest_title": "Backtest de Portafolio",
        # Portfolio section
        "portfolio_overview": "Tu Portafolio",
        "portfolio_value": "Valor Total del Portafolio",
        "total_investment": "Inversión Total",
        "total_return": "Rentabilidad Total",
        "portfolio_empty": "Tu portafolio está vacío. Añade algunas acciones para comenzar.",
        "add_stock": "Añadir Acción",
        "stock_symbol": "Símbolo de la Acción",
        "shares": "Número de Acciones",
        "avg_price": "Precio Medio de Compra ($)",
        "add_button": "Añadir/Actualizar Acción",
        "remove_stock": "Eliminar Acción",
        "select_stock": "Seleccionar Acción a Eliminar",
        "remove_all": "Eliminar todas las acciones",
        "remove_button": "Eliminar Acción",
        "portfolio_allocation": "Distribución del Portafolio",
        "portfolio_performance": "Rendimiento del Portafolio (Último Mes)",
        # Import portfolio section
        "import_portfolio": "Importar Portafolio desde CSV",
        "upload_csv": "Sube un archivo CSV con los datos de tu portafolio. El archivo debe incluir columnas para Símbolo, Acciones y Precio Medio de Compra.",
        "expected_format": "Formato CSV esperado:",
        "choose_csv": "Elige un archivo CSV",
        "replace_portfolio": "Reemplazar todo mi portafolio",
        "add_to_portfolio": "Añadir a mi portafolio existente",
        "import_button": "Importar Portafolio",
        # Analysis section
        "portfolio_overview_analysis": "Resumen del Portafolio", 
        "ai_risk_assessment": "Evaluación de Riesgo IA",
        "portfolio_stability": "Estabilidad del Portafolio",
        "ai_improvement": "Sugerencias de Mejora IA",
        "future_prediction": "Predicción de Rendimiento Futuro",
        "select_days": "Selecciona días para la predicción",
        "prediction_summary": "Resumen de la Predicción (basado en rendimiento histórico):",
        "current_value": "Valor actual del portafolio",
        "predicted_value": "Valor previsto para el",
        "predicted_return": "Rendimiento previsto en",
        "prediction_confidence": "Intervalo de confianza de la predicción",
        "prediction_note": "Nota: Esta predicción se basa en la volatilidad y rendimientos históricos. El rendimiento real del mercado puede variar significativamente.",
        "portfolio_optimization": "Optimización del Portafolio",
        "risk_reduction": "Reducción de Riesgo",
        "return_maximization": "Maximización del Rendimiento",
        # Footer
        "footer": "FinVision - Plataforma Avanzada de Análisis del Mercado Bursátil",
        # Real-time prices section
        "Real-Time Prices": "Precios en Tiempo Real",
        "Symbol": "Símbolo",
        "Price": "Precio",
        "Change": "Cambio",
        "Last Update": "Última Actualización",
        "Update Interval (seconds)": "Intervalo de Actualización (segundos)",
        "Auto-updating every": "Actualización automática cada",
        "seconds": "segundos",
        "Note: Prices are fetched from Yahoo Finance API and may be delayed by a few minutes for some exchanges.": "Nota: Los precios se obtienen de la API de Yahoo Finance y pueden retrasarse unos minutos para algunos mercados.",
        "About Real-Time Updates": "Acerca de las Actualizaciones en Tiempo Real",
        "This table automatically updates every": "Esta tabla se actualiza automáticamente cada",
        "Trend": "Tendencia"
    },
    "fr": {  # French
        "app_title": "FinVision - Analyse du Marché Boursier",
        "app_subtitle": "Analyse Avancée du Marché Boursier",
        "nav_dashboard": "Tableau de Bord",
        "nav_stock_analysis": "Analyse d'Actions",
        "nav_portfolio": "Portefeuille",
        "nav_portfolio_analysis": "Analyse du Portefeuille",
        "nav_market_overview": "Aperçu du Marché",
        "nav_advanced_tools": "Outils Avancés",
        "nav_charts": "Graphiques Avancés",
        "nav_portfolio_tool": "Analyseur de Portefeuille",
        "nav_portfolio_balance": "Équilibrage de Portefeuille",
        "nav_real_time_prices": "Prix en Temps Réel",
        "nav_backtest": "Backtest",
        "portfolio_balance_title": "Équilibrage de Portefeuille",
        "dashboard_title": "Tableau de Bord Financier",
        "stock_analysis_title": "Analyse Détaillée",
        "portfolio_title": "Gestionnaire de Portefeuille",
        "portfolio_analysis_title": "Analyse du Portefeuille par IA",
        "market_overview_title": "Aperçu du Marché",
        "backtest_title": "Backtest du Portefeuille",
        # Portfolio section
        "portfolio_overview": "Votre Portefeuille",
        "portfolio_value": "Valeur Totale du Portefeuille",
        "total_investment": "Investissement Total",
        "total_return": "Rendement Total",
        "portfolio_empty": "Votre portefeuille est vide. Ajoutez des actions pour commencer.",
        "add_stock": "Ajouter une Action",
        "stock_symbol": "Symbole de l'Action",
        "shares": "Nombre d'Actions",
        "avg_price": "Prix Moyen d'Achat ($)",
        "add_button": "Ajouter/Mettre à jour l'Action",
        "remove_stock": "Supprimer une Action",
        "select_stock": "Sélectionner l'Action à Supprimer",
        "remove_all": "Supprimer toutes les actions",
        "remove_button": "Supprimer l'Action",
        "portfolio_allocation": "Allocation du Portefeuille",
        "portfolio_performance": "Performance du Portefeuille (Dernier Mois)",
        # Import portfolio section
        "import_portfolio": "Importer le Portefeuille depuis CSV",
        "upload_csv": "Téléchargez un fichier CSV avec les données de votre portefeuille. Le fichier doit inclure des colonnes pour le Symbole, les Actions et le Prix Moyen d'Achat.",
        "expected_format": "Format CSV attendu:",
        "choose_csv": "Choisir un fichier CSV",
        "replace_portfolio": "Remplacer tout mon portefeuille",
        "add_to_portfolio": "Ajouter à mon portefeuille existant",
        "import_button": "Importer le Portefeuille",
        # Analysis section
        "portfolio_overview_analysis": "Aperçu du Portefeuille", 
        "ai_risk_assessment": "Évaluation des Risques par IA",
        "portfolio_stability": "Stabilité du Portefeuille",
        "ai_improvement": "Suggestions d'Amélioration par IA",
        "future_prediction": "Prédiction de Performance Future",
        "select_days": "Sélectionnez les jours pour la prédiction",
        "prediction_summary": "Résumé de la Prédiction (basé sur la performance historique):",
        "current_value": "Valeur actuelle du portefeuille",
        "predicted_value": "Valeur prévue le",
        "predicted_return": "Rendement prévu sur",
        "prediction_confidence": "Intervalle de confiance de la prédiction",
        "prediction_note": "Note: Cette prédiction est basée sur la volatilité et les rendements historiques. La performance réelle du marché peut varier considérablement.",
        "portfolio_optimization": "Optimisation du Portefeuille",
        "risk_reduction": "Réduction des Risques",
        "return_maximization": "Maximisation du Rendement",
        # Footer
        "footer": "FinVision - Plateforme Avancée d'Analyse du Marché Boursier",
        # Real-time prices section
        "Real-Time Prices": "Prix en Temps Réel",
        "Symbol": "Symbole",
        "Price": "Prix",
        "Change": "Variation",
        "Last Update": "Dernière Mise à Jour",
        "Update Interval (seconds)": "Intervalle de Mise à Jour (secondes)",
        "Auto-updating every": "Mise à jour automatique toutes les",
        "seconds": "secondes",
        "Note: Prices are fetched from Yahoo Finance API and may be delayed by a few minutes for some exchanges.": "Note: Les prix sont récupérés depuis l'API Yahoo Finance et peuvent être retardés de quelques minutes pour certains marchés.",
        "About Real-Time Updates": "À propos des Mises à Jour en Temps Réel",
        "This table automatically updates every": "Ce tableau se met à jour automatiquement toutes les",
        "Trend": "Tendance"
    }
}

# Language codes with their display names and flag emojis
languages = {
    "en": {"name": "English", "flag": "🇬🇧"},
    "it": {"name": "Italiano", "flag": "🇮🇹"},
    "es": {"name": "Español", "flag": "🇪🇸"},
    "fr": {"name": "Français", "flag": "🇫🇷"}
}

def get_translation(key, lang_code):
    """
    Get a translated string for a given key and language code.
    
    Args:
        key (str): The translation key to look up
        lang_code (str): The language code (e.g., 'en', 'it')
        
    Returns:
        str: The translated string or the key itself if translation not found
    """
    try:
        return translations[lang_code][key]
    except KeyError:
        # Fallback to English if translation not found
        try:
            return translations["en"][key]
        except KeyError:
            # Return the key itself if not found in any language
            return key

def translate_ui(text, lang_code):
    """
    Translate any UI text based on current language.
    This function allows for translating any static text in the application,
    not just predefined keys.
    
    Args:
        text (str): The text to translate (in English)
        lang_code (str): The language code (e.g., 'en', 'it')
        
    Returns:
        str: The translated text if a translation exists, otherwise the original text
    """
    if lang_code == "en":
        return text
        
    # Create a simple hash for the text to use as a temporary key
    import hashlib
    temp_key = f"dynamic_{hashlib.md5(text.encode()).hexdigest()[:8]}"
    
    # Check if we have a translation for this text
    # First look for direct matches in the translations dictionary
    for key, value in translations["en"].items():
        if value == text and key in translations[lang_code]:
            return translations[lang_code][key]
    
    # If no match found, return the original text
    return text