import streamlit as st

def display_guida():
    """
    Display a comprehensive guide about the application and all its features
    """
    st.title("Guida Completa a FinVision")
    
    st.markdown("""
    ## Benvenuto nella Guida Completa di FinVision
    
    Questa guida ti aiuterà a comprendere tutte le funzionalità e le caratteristiche di FinVision, il tuo strumento avanzato per l'analisi dei mercati finanziari.
    
    FinVision è stato progettato per fornire analisi approfondite, strumenti tecnici avanzati e funzionalità di gestione del portafoglio in un'unica piattaforma intuitiva.
    """)
    
    # Table of contents
    st.sidebar.markdown("## Indice")
    st.sidebar.markdown("""
    - [Panoramica Generale](#panoramica-generale)
    - [Dashboard Principale](#dashboard-principale)
    - [Analisi di Mercato](#analisi-di-mercato)
    - [Grafici Avanzati](#grafici-avanzati)
    - [Strumenti Avanzati](#strumenti-avanzati)
    - [Gestione del Portafoglio](#gestione-del-portafoglio)
    - [Analizzatore del Portafoglio](#analizzatore-del-portafoglio)
    - [Ricerca e News](#ricerca-e-news)
    """)
    
    # Panoramica Generale
    st.header("Panoramica Generale")
    st.markdown("""
    FinVision è un'applicazione completa per l'analisi del mercato azionario che offre:
    
    - **Analisi tecnica avanzata**: Visualizza grafici candlestick, indicatori tecnici multipli, e analisi comparative
    - **Gestione del portafoglio**: Monitora le tue posizioni azionarie e analizza le performance
    - **Analisi di mercato**: Visualizza panoramiche dei principali indici, settori, e performance di mercato
    - **Screening e comparazione**: Filtra azioni in base a criteri specifici e confronta più titoli
    - **Analisi stagionale**: Valuta le performance storiche per identificare pattern stagionali
    - **Analisi del rischio**: Esamina metriche di rischio e simulazioni Monte Carlo per previsioni future
    
    L'applicazione è stata progettata con un'interfaccia intuitiva in lingua italiana, pensata per rendere accessibili strumenti di analisi professionale anche a investitori non esperti.
    """)
    
    # Usiamo emoji e testo formattato invece di immagini esterne che potrebbero non caricarsi
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📈 Analisi Tecnica
        
        L'analisi tecnica ti permette di:
        - Identificare trend
        - Riconoscere pattern di prezzo
        - Utilizzare indicatori come RSI, MACD
        - Individuare livelli di supporto e resistenza
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Portafoglio Diversificato
        
        Un portafoglio ben diversificato include:
        - Titoli di diversi settori
        - Differenti classi di attività
        - Bilanciamento tra crescita e valore
        - Allocazione geografica strategica
        """)
    
    # Dashboard Principale
    st.header("Dashboard Principale")
    st.markdown("""
    La dashboard principale offre una visione complessiva del mercato e delle tue posizioni:
    
    - **Barra di navigazione laterale**: Accesso rapido a tutte le funzionalità dell'applicazione
    - **Panoramica di mercato**: Visualizzazione dei principali indici e settori
    - **Ricerca azioni**: Ricerca rapida di simboli azionari
    - **Portafoglio in evidenza**: Visualizzazione sintetica delle performance del tuo portafoglio
    
    La dashboard è stata progettata per offrirti subito le informazioni più rilevanti e permetterti di approfondire con un click.
    """)
    
    # Analisi di Mercato
    st.header("Analisi di Mercato")
    st.markdown("""
    La sezione di analisi di mercato fornisce una visione d'insieme del mercato azionario:
    
    - **Indici principali**: Performance giornaliera e storica dei principali indici mondiali (S&P 500, Dow Jones, Nasdaq, FTSE MIB, ecc.)
    - **Analisi settoriale**: Performance per settore industriale con heat map
    - **Top performer e peggiori**: Liste delle azioni con le migliori e peggiori performance
    - **Mappa di calore del mercato**: Visualizzazione grafica delle performance per settore e capitalizzazione
    
    Questa sezione ti aiuta a comprendere rapidamente dove si stanno verificando movimenti significativi nel mercato e quali settori stanno sovraperformando o sottoperformando.
    """)
    
    # Grafici Avanzati
    st.header("Grafici Avanzati")
    st.markdown("""
    La sezione di grafici avanzati offre strumenti di visualizzazione professionale:
    
    - **Grafico Candlestick Avanzato**: Visualizza grafici a candele con molteplici indicatori tecnici:
      - Medie mobili (SMA) personalizzabili
      - Bande di Bollinger
      - EMA veloce
      - Volume
      - Personalizzazione dello stile grafico (Default, Dark, Light, Financial)
    
    - **Grafico Multi-Indicatori**: Visualizza molteplici indicatori tecnici su un unico grafico:
      - MACD (Moving Average Convergence Divergence)
      - RSI (Relative Strength Index)
      - Stochastic Oscillator
      - Volume
      - Bande di Bollinger
      - ATR (Average True Range)
      - OBV (On-Balance Volume)
      - CCI (Commodity Channel Index)
    
    - **Grafico Comparativo Multi-Azioni**: Confronta più azioni su un unico grafico:
      - Confronto normalizzato (base 100) o a valori reali
      - Visualizzazione a linee, aree o candlestick
      - Opzione per scala logaritmica
      - Inclusione del volume
    
    - **Analisi Stagionale**: Identifica pattern stagionali nelle performance storiche:
      - Analisi mensile con heatmap per anno
      - Analisi trimestrale
      - Analisi per giorno della settimana
    
    Questi strumenti grafici avanzati ti permettono di eseguire analisi tecniche approfondite e identificare pattern, supporti, resistenze e opportunità di trading.
    """)
    
    # Strumenti Avanzati
    st.header("Strumenti Avanzati")
    st.markdown("""
    La sezione di strumenti avanzati offre funzionalità specializzate per analisi più approfondite:
    
    - **Screener Azioni**: Filtra le azioni in base a parametri specifici:
      - Prezzo (min/max)
      - Capitalizzazione di mercato
      - Rendimento da dividendi
      - Performance (giornaliera, settimanale, mensile, annuale)
      - Volume di scambi
      - Rapporto P/E
    
    - **Analisi di Correlazione**: Valuta la correlazione tra diverse azioni:
      - Matrice di correlazione
      - Heatmap di correlazione
      - Identificazione di azioni con bassa correlazione per diversificazione del portafoglio
    
    - **Analisi Comparativa**: Confronta metriche fondamentali tra diverse azioni:
      - Rapporti P/E, P/B, P/S
      - Margini di profitto
      - Crescita dei ricavi
      - Rendimento del capitale
      - Liquidità e solvibilità
    
    Questi strumenti ti aiutano a individuare opportunità di investimento, costruire portafogli diversificati e confrontare diverse azioni su basi fondamentali e tecniche.
    """)
    
    # Gestione del Portafoglio
    st.header("Gestione del Portafoglio")
    st.markdown("""
    La sezione di gestione del portafoglio ti permette di monitorare e gestire i tuoi investimenti:
    
    - **Visualizzazione del portafoglio**: Lista completa delle tue posizioni con:
      - Prezzo attuale
      - Prezzo di acquisto
      - Variazione percentuale
      - Profitto/perdita
      - Allocazione percentuale
    
    - **Aggiunta/rimozione di posizioni**: Inserisci manualmente i tuoi acquisti/vendite
    
    - **Importazione da CSV**: Importa le tue posizioni da file CSV
  
    - **Visualizzazione grafica**:
      - Composizione del portafoglio (grafico a torta)
      - Performance storica (grafico a linee)
      - Confronto con benchmark (S&P 500, ecc.)
    
    Questa sezione ti offre una visione chiara e completa dei tuoi investimenti, aiutandoti a monitorare le performance e prendere decisioni informate.
    """)
    
    # Analizzatore del Portafoglio
    st.header("Analizzatore del Portafoglio")
    st.markdown("""
    La sezione di analisi del portafoglio offre strumenti avanzati per valutare e ottimizzare il tuo portafoglio:
    
    - **Analisi della composizione**: Visualizzazione dettagliata dell'allocazione:
      - Per azione
      - Per settore
      - Per area geografica
      - Per capitalizzazione
    
    - **Analisi del rischio**: Calcolo e visualizzazione delle metriche di rischio:
      - Volatilità (deviazione standard)
      - Beta (rispetto al mercato)
      - Sharpe Ratio
      - Drawdown massimo
      - Value at Risk (VaR)
    
    - **Analisi di correlazione**: Visualizzazione della correlazione tra le azioni nel portafoglio:
      - Matrice di correlazione
      - Heatmap di correlazione
      - Rete di correlazione
    
    - **Simulazione Monte Carlo**: Proiezioni future basate sui dati storici:
      - Simulazione di scenari multipli
      - Intervalli di confidenza
      - Distribuzione dei possibili risultati
      - Probabilità di raggiungere obiettivi specifici
    
    Questo set di strumenti ti permette di comprendere meglio il profilo rischio-rendimento del tuo portafoglio, identificare potenziali problemi di diversificazione e stimare le performance future in diversi scenari.
    """)
    
    # Ricerca e News
    st.header("Ricerca e News")
    st.markdown("""
    La sezione di ricerca e news ti permette di trovare informazioni su azioni specifiche e restare aggiornato sulle ultime notizie:
    
    - **Ricerca azioni**: Cerca azioni per simbolo o nome azienda
    
    - **Dettagli azione**: Visualizza informazioni dettagliate:
      - Prezzo attuale e variazione
      - Dati fondamentali
      - Grafici storici
      - Indicatori tecnici
    
    - **News finanziarie**: Visualizza le ultime notizie relative a:
      - Azioni specifiche
      - Mercati in generale
      - Settori industriali
    
    - **Analisi del sentiment**: Valutazione del sentiment delle notizie:
      - Punteggio di sentiment (positivo, neutro, negativo)
      - Visualizzazione grafica del sentiment
      - Tendenze nel sentiment delle notizie
    
    Questa sezione ti aiuta a rimanere informato sugli sviluppi che potrebbero influenzare i tuoi investimenti e a trovare rapidamente informazioni su specifiche azioni di tuo interesse.
    """)
    
    # Conclusione
    st.markdown("""
    ---
    
    ## Conclusione
    
    FinVision è stato progettato per offrirti tutti gli strumenti necessari per analizzare il mercato, gestire il tuo portafoglio e prendere decisioni di investimento informate. 
    
    L'applicazione combina strumenti di analisi tecnica avanzata, gestione del portafoglio, ricerca e news in un'unica piattaforma intuitiva in lingua italiana.
    
    Speriamo che questa guida ti aiuti a sfruttare al meglio tutte le funzionalità di FinVision. Buon investimento!
    
    ---
    
    ### Assistenza
    
    Per qualsiasi domanda o problema, non esitare a contattarci attraverso la sezione di supporto.
    """)