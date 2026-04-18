from app.agents.state import AgentState
import os
import uuid

try:
    import markdown2
    MARKDOWN2_AVAILABLE = True
except ImportError:
    MARKDOWN2_AVAILABLE = False

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False

def report_node(state: AgentState) -> dict:
    """
    Generates PDF and HTML reports.
    """
    print(f"--- REPORT NODE ---")
    insights = state.get("insights", "No insights generated.")
    eda_summary = state.get("eda_summary", {})
    
    html_content = f"""
    <html>
    <head>
        <title>Data Science Autonomous Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ max-width: 800px; margin: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Autonomous Data Science Report</h1>
            <h2>EDA Summary</h2>
            <p>Dataset Shape: {eda_summary.get('shape')}</p>
            <h2>Insights & Results</h2>
            <div>{markdown2.markdown(insights) if MARKDOWN2_AVAILABLE else insights.replace(chr(10), '<br>')}</div>
        </div>
    </body>
    </html>
    """
    
    report_id = str(uuid.uuid4())[:8]
    html_path = f"reports/report_{report_id}.html"
    pdf_path = f"reports/report_{report_id}.pdf"
    
    os.makedirs("reports", exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    try:
        if PDFKIT_AVAILABLE:
            pdfkit.from_string(html_content, pdf_path)
        else:
            raise RuntimeError("pdfkit not available")
    except Exception as e:
        print(f"PDF generation failed, returning HTML only: {e}")
        pdf_path = html_path  # Fallback to HTML
        
    return {"report_path": pdf_path}
