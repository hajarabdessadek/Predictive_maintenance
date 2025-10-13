# app_vibration_dashboard_corrected.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import tensorflow as tf
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from datetime import datetime
import io

# ----------------------------
# Config
# ----------------------------
MODEL1_FILE = "modele_gru_regression.h5"
MODEL2_FILE = "modele_vibrations_corrige.h5"
INPUT_LEN = 1200
PRED_LEN = 32
NUM_COEFFS = 17  # pour le modèle de classification
LABELS = ["Normal", "Problème léger", "Problème grave"]
COLORS = ["green", "orange", "red"]

st.set_page_config(page_title="Dashboard — Prédiction de panne", layout="wide")
st.image("logo_ocp.png", width=100)
st.markdown(
    "<h1 style='color:#009639;'>🔧 Analyse et Prédiction de l’État de santé du Moteur</h1>",
    unsafe_allow_html=True
)

# ----------------------------
# Helper : load models once
# ----------------------------
@st.cache_resource
def load_models():
    m1 = m2 = None
    errors = []
    try:
        m1 = tf.keras.models.load_model(MODEL1_FILE, compile=False)
    except Exception as e:
        errors.append(f"Impossible de charger le modèle de prédiction (modèle 1). Détail: {e}")
    try:
        m2 = tf.keras.models.load_model(MODEL2_FILE, compile=False)
    except Exception as e:
        msg = str(e)
        if "Lambda" in msg or "output_shape" in msg:
            errors.append("Erreur modèle 2 : le modèle inclut une couche Lambda sans output_shape. Ré-entrainez le modèle correctement.")
        else:
            errors.append(f"Impossible de charger le modèle de classification (modèle 2). Détail: {e}")
    return m1, m2, errors

model1, model2, load_errors = load_models()

if load_errors:
    for e in load_errors:
        st.error(e)
    st.info("Placez les fichiers .h5 dans le dossier de l'application puis relancez l'app.")

# ----------------------------
# Upload CSV
# ----------------------------
uploaded = st.file_uploader("📥 Importer le fichier CSV", type=["csv"])
if uploaded is None:
    st.info("➡️ Importez un fichier CSV pour commencer.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Erreur lecture CSV : {e}")
    st.stop()

if 'Vibration_1' not in df.columns:
    st.error("Colonnes manquante : Le CSV doit contenir 'Vibration_1'.")
    st.stop()

series = df['Vibration_1'].values.astype(float)
n = len(series)
if n < INPUT_LEN:
    st.warning(f"Seulement {n} valeurs, complétées par des zéros.")
    padded = np.zeros(INPUT_LEN)
    padded[:n] = series
    series1200 = padded
else:
    series1200 = series[-INPUT_LEN:]  # <-- dernières 1200 valeurs

# ----------------------------
# Plots et état
# ----------------------------
#st.subheader("📊 Analyse des vibrations")
st.markdown(f"<h2 style='color:#009639;'>📊 Analyse des vibrations</h2>", unsafe_allow_html=True)


# Courbes
fig1, ax1 = plt.subplots(figsize=(9,3))
ax1.plot(np.arange(len(series1200)), series1200, color='#00B050', label="Vibration actuelle")
ax1.set_xlabel("Index")
ax1.set_ylabel("Amplitude")
ax1.grid(True)
ax1.legend()

st.markdown("### Visualisation des vibrations du moteur (40 dernières minutes)")
st.pyplot(fig1)

# Prédiction futur
if model1 is None:
    st.warning("⚠️ Modèle de prédiction non chargé.")
    st.stop()
try:
    x_in = series1200.reshape(1, INPUT_LEN, 1)
    pred32 = model1.predict(x_in).reshape(-1)[:PRED_LEN]
except Exception as e:
    st.error(f"Erreur lors de la prédiction (modèle 1) : {e}")
    st.stop()

fig2, ax2 = plt.subplots(figsize=(9,3))
ax2.plot(np.arange(len(pred32)), pred32, marker='o', color='#00B050', label="Vibration prédite (32)")
ax2.set_xlabel("Index (futur)")
ax2.set_ylabel("Amplitude")
ax2.grid(True)
ax2.legend()
st.markdown("### Visualisation des vibrations prévues (après 20 min)")
st.pyplot(fig2)

# ----------------------------
# État de santé du moteur (cercle + tableau)
# ----------------------------
st.markdown(f"<h2 style='color:#009639;'>📌 État de santé du moteur</h2>", unsafe_allow_html=True)


if model2 is None:
    st.warning("⚠️ Modèle de classification non chargé.")
else:
    try:
        fft_coeffs = np.abs(np.fft.rfft(pred32))[:NUM_COEFFS]
        fft_coeffs = fft_coeffs / (np.max(fft_coeffs) + 1e-8)
        x2 = fft_coeffs.reshape(1, -1)
        probs = model2.predict(x2).reshape(-1)
        probs = probs / (probs.sum() + 1e-9)
    except Exception as e:
        st.error(f"Erreur lors de la classification : {e}")
        st.stop()

    idx = int(np.argmax(probs))
    dominant = LABELS[idx]

    # Notification
    if dominant == "Normal":
        st.success("✅ Le moteur fonctionne normalement. Aucun problème détecté.")
    elif dominant == "Problème léger":
        st.warning("⚠️ Un léger déséquilibre a été détecté. Veuillez planifier une vérification préventive avant aggravation.")
    elif dominant == "Problème grave":
        st.error("🚨 ÉTAT CRITIQUE : Arrêtez immédiatement le processus pour éviter des dommages majeurs aux équipements.")

    st.markdown("---")

    # Cercle sans texte
    figp, axp = plt.subplots(figsize=(4,4))
    axp.pie(
        probs,
    labels=None,
    colors=COLORS,
    startangle=90,
    wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}  # bord fin pour mieux voir
    )
    axp.axis('equal')
    figp.subplots_adjust(left=0.3, right=0.7, top=0.7, bottom=0.3)  # réduit le cercle visuellement
    axp.set_title("Répartition prédictive des états du moteur", color="#009639", fontsize=8, fontweight='bold')  # titre
    st.pyplot(figp, clear_figure=True, use_container_width=False)

    # Tableau avec couleurs représentatives
st.markdown(f"<h2 style='color:#009639;'>📋 Détails des probabilités</h2>", unsafe_allow_html=True)

#st.subheader("📋 Détails des probabilités")
dfp = pd.DataFrame({
    "État": LABELS,
    "Probabilité": [float(p) for p in probs]
}).sort_values("Probabilité", ascending=False)

def color_map_row(row):
    if row["État"] == "Normal":
        return ["color: green", "color: green"]
    elif row["État"] == "Problème léger":
        return ["color: orange", "color: orange"]
    elif row["État"] == "Problème grave":
        return ["color: red", "color: red"]
    else:
        return ["", ""]

st.dataframe(
    dfp.style.format({"Probabilité":"{:.3f}"}).apply(color_map_row, axis=1)
)

# ----------------------------
def generate_ocp_pdf(fig1, fig2, figp, dominant, probs, LABELS, student_name="Hajar Abdessaedek"):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from datetime import datetime
    import io

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = f"Rapport_OCP_Moteur_{ts}.pdf"
    doc = SimpleDocTemplate(pdf_name, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=60)

    # --- Styles ---
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontSize=22, textColor=colors.HexColor("#004225"))
    subtitle = ParagraphStyle("SubTitle", parent=styles["Heading2"], leading=18, spaceAfter=4, textColor=colors.HexColor("#007E33"))
    normal = styles["Normal"]

    story = []

    # === PAGE 1 : GARDE + ANALYSE DES VIBRATIONS ===
    try:
        story.append(Image("logo_ocp.png", width=5*cm, height=2*cm))
    except:
        pass
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Rapport de Diagnostic — Maintenance Prédictive</b>", title))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Entreprise :</b> OCP Group", normal))
    story.append(Paragraph(f"<b>Stagiaire :</b> {student_name}", normal))
    story.append(Paragraph(f"<b>Date :</b> {datetime.now().strftime('%d/%m/%Y à %H:%M')}", normal))
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "Ce rapport présente une analyse détaillée des vibrations du motoréducteur sur les 40 dernières minutes. "
        "L’étude utilise un modèle de prédiction basé sur l’apprentissage profond pour estimer l’état des vibrations "
        "après un décalage de 20 minutes, permettant ainsi d’anticiper l’état de santé du moteur et de prendre "
        "des mesures préventives avant qu’une panne ou un dysfonctionnement grave ne survienne.", normal))
    story.append(Spacer(1, 20))

    # --- Analyse des vibrations ---
    story.append(Paragraph("Visualisation des vibrations du moteur (40 dernières minutes)", subtitle))
    story.append(Spacer(1, 10))
    img1 = io.BytesIO(); fig1.savefig(img1, format="png", bbox_inches="tight"); img1.seek(0)
    story.append(Image(img1, width=15*cm, height=6*cm))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Visualisation des vibrations prévues (après 20 min)", subtitle))
    story.append(Spacer(1, 10))
    img2 = io.BytesIO(); fig2.savefig(img2, format="png", bbox_inches="tight"); img2.seek(0)
    story.append(Image(img2, width=15*cm, height=6*cm))
    story.append(PageBreak())

    # === PAGE 2 : CLASSIFICATION + RECOMMANDATION + CONCLUSION ===
    # story.append(Paragraph("Résultats de classification", subtitle))
    # story.append(Spacer(1, 2))
    # imgp = io.BytesIO(); figp.savefig(imgp, format="png", bbox_inches="tight"); imgp.seek(0)
    # story.append(Image(imgp, width=8*cm, height=8*cm))
    # story.append(Spacer(1, 2))

    # Tableau des probabilités
    data = [["État", "Probabilité (%)"]] + [[LABELS[i], f"{probs[i]*100:.2f}"] for i in range(len(LABELS))]
    table = Table(data, colWidths=[8*cm, 6*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))
    story.append(table)
    story.append(Spacer(1, 15))

    # Recommandation selon l’état
    rec_text = {
        "Normal": "<font color='green'><b>État Normal :</b></font> Aucun problème détecté. Poursuivre les opérations.",
        "Problème léger": "<font color='orange'><b>Problème Léger :</b></font> Déséquilibre mineur. Planifier une maintenance préventive.",
        "Problème grave": "<font color='red'><b>Problème Grave :</b></font> Risque de panne imminente. Arrêter la machine pour inspection."
    }[dominant]
    story.append(Paragraph(rec_text, normal))
    story.append(Spacer(1, 15))

    story.append(Spacer(1, 15))

    # Conclusion
    story.append(Paragraph("Conclusion et recommandations globales", subtitle))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Les modèles GRU et MLP ont permis de prédire l’évolution des vibrations, "
        "offrant un outil de maintenance prédictive performant."
        "Les résultats confirment que cette approche peut réduire les temps d’arrêt non planifiés et améliorer la fiabilité opérationnelle des motoréducteurs au sein du groupe OCP.", normal))
    story.append(Spacer(1, 15))
    story.append(Paragraph(f"<b>Généré le :</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", normal))

    # --- Construire le PDF ---
    doc.build(story)
    return pdf_name

# ----------------------------
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #009639;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.5em;
        font-weight: bold;
        font-size: 16px;
    }
    div.stButton > button:first-child:hover {
        background-color: #00b050; /* vert clair au survol */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

if st.button("📖 Générer rapport "):
    try:
        pdf_file = generate_ocp_pdf(fig1, fig2, figp, dominant, probs, LABELS)
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Télécharger le rapport PDF", f, file_name=pdf_file, mime="application/pdf")
        st.success("✅ Rapport  généré avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de la génération du rapport : {e}")
