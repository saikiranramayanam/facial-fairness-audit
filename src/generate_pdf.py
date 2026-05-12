from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

from reportlab.lib.pagesizes import letter


doc = SimpleDocTemplate(
    "submission/deployment_memo.pdf",
    pagesize=letter
)

styles = getSampleStyleSheet()

story = []


with open(
    "submission/deployment_memo.md",
    "r",
    encoding="utf-8"
) as f:

    lines = f.readlines()


for line in lines:

    line = line.strip()

    if line == "":
        continue

    paragraph = Paragraph(
        line,
        styles["BodyText"]
    )

    story.append(paragraph)

    story.append(Spacer(1, 12))


doc.build(story)

print(
    "Deployment memo PDF created successfully"
)