from fpdf import FPDF

def export_report(history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Rapport de Phase TTU-MC3", ln=True, align='C')
    for item in history:
        pdf.multi_cell(0, 10, txt=f"Q: {item['q']}\nSubstance: {item['s'][:20]}...")
    return pdf.output(dest='S').encode('latin-1')

st.sidebar.download_button(
    label="📥 Exporter Rapport de Connaissance",
    data=export_report(st.session_state.kernel.history),
    file_name="rapport_ia_souveraine.pdf",
    mime="application/pdf"
)
