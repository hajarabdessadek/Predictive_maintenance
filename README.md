# Predictive_maintenance
# 🧠 Système de Maintenance Prédictive des Équipements Industriels – OCP Benguerir

## 🚀 Contexte du Projet  
Ce projet s’inscrit dans le cadre de mon **stage PFA** réalisé au sein du **Groupe OCP Benguerir**.  
L’objectif est de développer un **système de maintenance prédictive** dédié aux **équipements industriels critiques**, notamment les **motoréducteurs**, afin d’améliorer la **continuité** et la **fiabilité** du processus de production.  

Ce projet s’intègre pleinement dans la dynamique de **transformation numérique** du Groupe OCP et dans la vision de l’**Industrie 4.0**, visant à exploiter la puissance des **données industrielles** et du **Deep Learning** pour une production plus intelligente, plus sûre et plus durable.  

---

## 🎯 Objectif Principal  
Anticiper les **pannes potentielles** des équipements en exploitant les **données issues des capteurs industriels** à travers des **modèles de Deep Learning** et des **techniques d’analyse prédictive**, dans le but de :  
- Réduire les **arrêts non planifiés**  
- Optimiser les **interventions de maintenance**  
- Améliorer la **disponibilité et la fiabilité opérationnelle**  

---

## 🔍 Méthodologie Adoptée : CRISP-DM  
Le développement du projet suit la méthodologie **CRISP-DM** (*Cross Industry Standard Process for Data Mining*), une approche structurée pour la conception de projets data-driven :  

1. **Compréhension du problème métier** – Analyse du contexte industriel et des enjeux de maintenance.  
2. **Collecte et compréhension des données** – Acquisition et prétraitement des signaux vibratoires.  
3. **Préparation des données** – Nettoyage, filtrage et extraction des caractéristiques fréquentielles (FFT).  
4. **Modélisation** – Conception et entraînement des modèles de **Deep Learning GRU et MLP**.  
5. **Évaluation** – Validation des modèles sur des jeux de données réels.  
6. **Déploiement** – Intégration dans une application interactive pour la visualisation et la prise de décision.  

---

## 🧩 Modèles Utilisés

### 🔹 Modèle GRU (Gated Recurrent Unit)
Le modèle **GRU** a été utilisé pour **prédire les vibrations futures** des motoréducteurs à partir d’un historique de mesures.  
Grâce à sa capacité à capturer les dépendances temporelles à long terme, le GRU s’avère particulièrement efficace pour les **séries temporelles industrielles**.

📈 **Résultat :** prédiction d’un lot de 32 valeurs futures (une par seconde) après un décalage de 20 minutes, à partir d’un lot de mesures de 40 minutes.

---

### 🔹 Modèle MLP (Multi-Layer Perceptron)
Le modèle **MLP** a été utilisé pour **classifier l’état du moteur** en se basant sur les vibrations prédites par le modèle GRU.  
Les classes sont :  
- ✅ État normal  
- ⚠️ Problème mineur  
- ❌ Problème critique  

Cette combinaison GRU + MLP forme une **architecture Deep Learning prédiction–diagnostic**, robuste et efficace.

---

## 💻 Application Interactive  
Les modèles GRU et MLP ont été intégrés dans une **application Streamlit**, permettant :  
- Une **visualisation en temps réel** des signaux vibratoires et de leurs prédictions futures.  
- Une **évaluation automatique** de la condition du moteur.  
- Des **alertes visuelles** selon l’état détecté.  
- Une **génération automatique de rapports PDF**.  

🎥 **Vidéo de démonstration :**  
[🎬 Voir la vidéo](https://github.com/hajarabdessadek/Predictive_maintenance/blob/main/%C3%A9preuve_stage.mp4)

---

## 🛠️ Outils et Technologies  
- **Python** (Pandas, Numpy, TensorFlow, Keras)  
- **Streamlit** pour le déploiement interactif  
- **FFT (Fast Fourier Transform)** pour l’analyse fréquentielle  
- **Matplotlib / Plotly** pour la visualisation  
- **Git & GitHub** pour la gestion de version et la documentation  

---

## 🌟 Résultats et Impact  
✅ Réduction des risques de panne imprévue  
✅ Optimisation des opérations de maintenance  
✅ Amélioration de la fiabilité et de la sécurité des équipements  
✅ Intégration facile dans une stratégie globale **Industrie 4.0**

---

## 👩‍💻 Auteur  
**Hajar Abdessaedek**  
Étudiante ingénieure en **Robotique & Objets Connectés**  
📍 École Nationale de l’Intelligence Artificielle et du Digital  
---
