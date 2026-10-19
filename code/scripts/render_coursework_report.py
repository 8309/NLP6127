"""Render the English coursework draft without inventing missing results.

Requires reportlab and pypdfium2 from the document runtime. It does not invoke
models, copy source data, update an index or publish an archive.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import posixpath
import re

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Flowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


WIDTH = A4[0] - 104


class Diagram(Flowable):
    def __init__(self, drawing):
        super().__init__()
        self.drawing = drawing
        self.width, self.height = drawing.width, drawing.height
        self.keepWithNext = True

    def draw(self):
        renderPDF.draw(self.drawing, self.canv, 0, 0)


def inline(text):
    # Plain text is escaped before adding the small, explicitly supported subset.
    text = escape(text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-"))
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    def link(match):
        label, target = match.groups()
        if target.startswith(("https://", "http://")):
            return f'<link href="{target}" color="#215a75">{label}</link>'
        if not re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target) and not target.startswith("/"):
            relative = posixpath.normpath("docs/" + target)
            if not relative.startswith("../"):
                url = "https://github.com/8309/6127NLP/blob/main/" + relative
                return f'<link href="{url}" color="#215a75">{label}</link>'
        return label + " (" + target + ")"
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)


def architecture():
    drawing = Drawing(WIDTH, 152)
    boxes = [
        (8, 105, 112, 34, "Offline corpus"),
        (152, 105, 133, 34, "Cutoff eligibility A"),
        (319, 105, 148, 34, "BM25 + dense -> RRF"),
        (319, 25, 148, 34, "Evidence-conditioned LLM"),
        (152, 25, 133, 34, "Claim support check B"),
        (8, 25, 112, 34, "Cited answer + sources"),
    ]
    for x, y, width, height, label in boxes:
        drawing.add(Rect(x, y, width, height, rx=4, ry=4,
                         fillColor=colors.HexColor("#edf4f8"), strokeColor=colors.HexColor("#32687a")))
        drawing.add(String(x + width / 2, y + 14, label, textAnchor="middle",
                           fontName="Helvetica", fontSize=8))
    for x1, y1, x2, y2 in [(120, 122, 152, 122), (285, 122, 319, 122),
                           (393, 105, 393, 59), (319, 42, 285, 42), (152, 42, 120, 42)]:
        drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#32687a")))
        if x1 == x2:
            drawing.add(Line(x2, y2, x2 - 3, y2 + 5))
            drawing.add(Line(x2, y2, x2 + 3, y2 + 5))
        else:
            direction = 1 if x2 > x1 else -1
            drawing.add(Line(x2, y2, x2 - 5 * direction, y2 - 3))
            drawing.add(Line(x2, y2, x2 - 5 * direction, y2 + 3))
    return drawing


def latency_figure(receipt):
    drawing = Drawing(WIDTH, 128)
    max_value = max(entry["p95_seconds"] for entry in receipt["latency"].values())
    for i, mode in enumerate(("bm25", "dense", "hybrid")):
        value = receipt["latency"][mode]["median_seconds"]
        y = 88 - i * 30
        drawing.add(String(10, y + 5, mode.upper(), fontSize=9))
        width = value / max_value * 265
        drawing.add(Rect(86, y, width, 18, fillColor=colors.HexColor("#32758b"), strokeColor=None))
        drawing.add(String(96 + width, y + 5, f"{value:.3f} s", fontSize=9))
    drawing.add(String(10, 3, "Warm median retrieval latency; 50 development queries per mode, top 10.", fontSize=8))
    return drawing


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#63717b"))
    canvas.drawString(52, 26, "DRAFT - cover identifiers and submission links pending")
    canvas.drawRightString(A4[0] - 52, 26, str(doc.page))
    canvas.restoreState()


def render(source, output, receipt=None):
    source, output = Path(source), Path(output)
    if not source.is_file() or source.resolve() == output.resolve():
        raise ValueError("Missing source or unsafe output")
    if output.exists():
        raise ValueError("Refuse to overwrite an existing report; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("OilBody", parent=styles["BodyText"], fontSize=9, leading=13,
                          spaceAfter=5, splitLongWords=True)
    reference_body = ParagraphStyle("OilReference", parent=body, fontSize=8.5, leading=11.5, spaceAfter=5)
    table_body = ParagraphStyle("OilCell", parent=body, fontSize=8, leading=11, spaceAfter=0)
    caption = ParagraphStyle("OilCaption", parent=body, fontSize=8, alignment=TA_CENTER,
                             keepWithNext=True)
    title = ParagraphStyle("OilTitle", parent=styles["Title"], fontSize=21, leading=25, spaceAfter=16)
    heading = ParagraphStyle("OilHeading", parent=styles["Heading2"], fontSize=13, leading=17,
                             spaceBefore=12, spaceAfter=7, keepWithNext=True)
    subheading = ParagraphStyle("OilSubheading", parent=styles["Heading3"], fontSize=10.5, leading=14,
                                keepWithNext=True)
    lines = source.read_text(encoding="utf-8").splitlines()
    story, paragraph, table_rows = [], [], []
    in_references = False

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), reference_body if in_references else body))
            paragraph.clear()

    def flush_table():
        if not table_rows:
            return
        count = len(table_rows[0])
        if any(len(row) != count for row in table_rows):
            raise ValueError("Malformed source table")
        weights = [max(8, min(50, max(len(row[i]) for row in table_rows))) for i in range(count)]
        for index, name in enumerate(table_rows[0]):
            if name == "Cutoff":
                weights[index] = max(weights[index], 16)
            if name == "Questions":
                weights[index] = max(weights[index], 16)
            if re.fullmatch(r"(?:Precision|MRR|nDCG)@\d+", name):
                weights[index] = max(weights[index], 14)
        widths = [WIDTH * weight / sum(weights) for weight in weights]
        content = [[Paragraph(inline(cell), table_body) for cell in row] for row in table_rows]
        table = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4edf2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#c2ced5")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        # Short tables should not leave one isolated data row on the previous
        # page. Larger tables keep repeatable headers and may split normally.
        table_flowable = KeepTogether([table]) if table.wrap(WIDTH, A4[1])[1] <= A4[1] * .4 else table
        story.extend([table_flowable, Spacer(1, 9)])
        table_rows.clear()

    for line in lines:
        if line.startswith("|"):
            flush_paragraph()
            if not re.fullmatch(r"[| :\-]+", line):
                table_rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
            continue
        flush_table()
        if not line.strip():
            flush_paragraph()
        elif line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(inline(line[2:]), title))
        elif line.startswith("## ") or line.startswith("### "):
            flush_paragraph()
            is_sub = line.startswith("### ")
            text = line[4:] if is_sub else line[3:]
            in_references = text == "References" or in_references
            story.append(Paragraph(inline(text), subheading if is_sub else heading))
            if text == "Retrieval methods":
                story.extend([Diagram(architecture()), Paragraph(
                    "Figure 1. Evidence flow; A and B can be disabled separately for ablation.", caption)])
            if text == "Quality and performance measures" and receipt:
                story.extend([Diagram(latency_figure(receipt)), Paragraph(
                    "Figure 2. Measured development retrieval latency, not answer accuracy.", caption)])
        elif re.match(r"(?:- |\d+\. )", line):
            flush_paragraph()
            story.append(Paragraph(inline(line), reference_body if in_references else body))
        else:
            paragraph.append(line.strip())
    flush_paragraph()
    flush_table()
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=52, leftMargin=52,
                            topMargin=40, bottomMargin=44, title="Draft Oil Evidence Coursework Report",
                            author="", allowSplitting=1)
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return output


def previews(pdf_path, folder):
    import pypdfium2
    folder = Path(folder)
    if folder.exists():
        raise ValueError("Refuse to overwrite an existing preview directory")
    folder.mkdir(parents=True)
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    count = len(pdf)
    for index in range(count):
        page = pdf[index]
        bitmap = page.render(scale=1.5)
        bitmap.to_pil().save(folder / f"page-{index + 1:02d}.png")
        bitmap.close()
        page.close()
    pdf.close()
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs/FINAL_REPORT.md"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-receipt", type=Path)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.retrieval_receipt.read_text(encoding="utf-8")) if args.retrieval_receipt else None
    path = render(args.source, args.output, receipt)
    print(json.dumps(dict(status="draft_rendered_not_submission_ready", pdf=str(path),
                          pages=previews(path, args.previews), figures_are_measured_or_architecture=True)))
