#!/usr/bin/env python3
"""
generate_izvestaj.py
Generiše izveštaj.pdf iz izveštaj.json podataka.
Koristi reportlab za lep PDF sa crvenim tekstom za ozbiljne greške.
"""

import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, red
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

DATA_FILE = "izveštaj.json"
OUTPUT_PDF = "izveštaj.pdf"

def create_styles():
    styles = getSampleStyleSheet()
    
    # Title
    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Heading1'],
        fontSize=22,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=HexColor('#1a1a2e')
    ))
    
    # Subtitle
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=30,
        textColor=HexColor('#4a4a4a')
    ))
    
    # Article header
    styles.add(ParagraphStyle(
        name='ArticleHeader',
        parent=styles['Heading2'],
        fontSize=13,
        spaceBefore=15,
        spaceAfter=6,
        textColor=HexColor('#16213e'),
        borderPadding=3,
    ))
    
    # Normal text
    styles.add(ParagraphStyle(
        name='ReportBodyText',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    ))
    
    # Typos list
    styles.add(ParagraphStyle(
        name='TyposList',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        leftIndent=15,
        spaceAfter=2,
        textColor=HexColor('#0f3460'),
    ))
    
    # Serious issues header (red)
    styles.add(ParagraphStyle(
        name='SeriousHeader',
        parent=styles['Heading3'],
        fontSize=10,
        spaceBefore=8,
        spaceAfter=4,
        textColor=red,
    ))
    
    # Serious issues text (red)
    styles.add(ParagraphStyle(
        name='SeriousText',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        leftIndent=10,
        textColor=red,
        spaceAfter=2,
    ))
    
    # Footer info
    styles.add(ParagraphStyle(
        name='Footer',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=HexColor('#666666'),
    ))
    
    return styles

def generate_pdf():
    if not os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} ne postoji. Kreiram prazan izveštaj.")
        data = {"articles": []}
    else:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    
    articles = data.get("articles", [])
    
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm
    )
    
    styles = create_styles()
    story = []
    
    # Header
    story.append(Paragraph("Izveštaj o slovnim greškama i typo-ovima", styles['ReportTitle']))
    story.append(Paragraph(
        "Blog: <b>dejanbatanjac.github.io</b><br/>"
        "Obrada samo <b>published: true</b> članaka<br/>"
        "Fokus: jednostavne slovne greške, typo, pravopis<br/>"
        "Ozbiljne logičke / katastrofalne greške označene <font color='red'><b>CRVENOM</b></font> bojom",
        styles['Subtitle']
    ))
    
    story.append(Spacer(1, 10))
    
    if not articles:
        story.append(Paragraph("<i>Nema obrađenih članaka još.</i>", styles['ReportBodyText']))
    else:
        for art in articles:
            filename = art.get("filename", "")
            title = art.get("title", filename)
            date = art.get("date", "")
            
            header = f"<b>{date}</b> — {title}<br/><font size='8' color='#555555'>{filename}</font>"
            story.append(Paragraph(header, styles['ArticleHeader']))
            
            # Typos fixed
            typos = art.get("typos_fixed", [])
            if typos:
                story.append(Paragraph("<b>Ispravljene slovne greške / typo:</b>", styles['ReportBodyText']))
                for t in typos:
                    story.append(Paragraph(f"• {t}", styles['TyposList']))
            else:
                story.append(Paragraph("<i>Nema jednostavnih slovnih grešaka.</i>", styles['TyposList']))
            
            # Serious issues (RED)
            serious = art.get("serious_issues", [])
            if serious:
                story.append(Paragraph("<b><font color='red'>⚠ KATASTROFALNE / LOGIČKE / VAŽNE GREŠKE:</font></b>", styles['SeriousHeader']))
                for s in serious:
                    story.append(Paragraph(f"<font color='red'>• {s}</font>", styles['SeriousText']))
            
            story.append(Spacer(1, 8))
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"Ukupno obrađeno članaka: <b>{len(articles)}</b> | Generisano automatski | Samo published članci",
        styles['Footer']
    ))
    
    # Green listing of changes for critical (red) errors: what was critical and what is new/fixed
    critical_articles = [a for a in articles if a.get("serious_issues")]
    if critical_articles:
        story.append(Spacer(1, 15))
        green_style = ParagraphStyle(
            name='GreenListing',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            textColor=HexColor('#006400'),  # dark green
            spaceAfter=4,
        )
        story.append(Paragraph("<b>Listing promena za kritične (crvene) greške - šta je bilo kritično i šta je novo/fiksirano (zeleno):</b>", styles['SeriousHeader']))
        for a in critical_articles:
            story.append(Paragraph(f"<b>{a['date']} — {a['title']} ({a['filename']})</b>", green_style))
            for fix in a.get("typos_fixed", []):
                story.append(Paragraph(f"  • Fiksirano/novo: {fix}", green_style))
            for issue in a.get("serious_issues", []):
                story.append(Paragraph(f"  • Bilo kritično: {issue}", green_style))
    
    doc.build(story)
    print(f"PDF generisan: {OUTPUT_PDF} ({len(articles)} članaka)")

if __name__ == "__main__":
    generate_pdf()
