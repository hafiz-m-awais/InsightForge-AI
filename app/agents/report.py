from app.agents.state import AgentState
import markdown2
import pdfkit
import os
import uuid

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
            <div>{markdown2.markdown(insights)}</div>
        </div>
    </body>
    </html>
    """
    
    report_id = str(uuid.uuid4())[:8]
    html_path = f"reports/report_{report_id}.html"
    pdf_path = f"reports/report_{report_id}.pdf"
    
    with open(html_path, "w") as f:
        f.write(html_content)
        
    try:
        # Generate PDF using pdfkit (requires wkhtmltopdf installed via Docker)
        pdfkit.from_string(html_content, pdf_path)
    except Exception as e:
        print(f"PDF generation failed, returning HTML only: {e}")
        pdf_path = html_path # Fallback
        
    return {"report_path": pdf_path}
