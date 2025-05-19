import os
from jinja2 import Template
from datetime import datetime

OUTPUT_DIR = "results"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Text Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .article { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
        .success { color: green; }
        .error { color: red; }
        .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
        .metric-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .missing { background-color: #ffeeee; }
        .extra { background-color: #eeffee; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Text Analysis Report</h1>
        <p>Generated at: {{ timestamp }}</p>
        <p>Total articles: {{ total_articles }} ({{ success_count }} success, {{ error_count }} errors)</p>
    </div>

    {% for article in articles %}
    <div class="article">
        <h2>Article #{{ loop.index }}: <a href="{{ article.url }}">{{ article.url }}</a></h2>
        <p>Status: <span class="{{ article.status }}">{{ article.status|upper }}</span></p>
        
        {% if article.status == 'success' %}
        <div class="metrics">
            <div class="metric-card">
                <h3>Similarity</h3>
                <p>{{ article.similarity }}%</p>
            </div>
            <div class="metric-card">
                <h3>Length Comparison</h3>
                <p>Original: {{ article.original_length }} chars</p>
                <p>Extracted: {{ article.lib_length }} chars</p>
            </div>
            <div class="metric-card">
                <h3>Differences</h3>
                <p>Missing: {{ article.missing_lines_count }} lines</p>
                <p>Extra: {{ article.extra_lines_count }} lines</p>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
            <div class="missing">
                <h3>Missing Content Examples</h3>
                <ul>
                    {% for example in article.example_missing %}
                    <li>{{ example|truncate(150) }}</li>
                    {% endfor %}
                </ul>
            </div>
            <div class="extra">
                <h3>Extra Content Examples</h3>
                <ul>
                    {% for example in article.example_extra %}
                    <li>{{ example|truncate(150) }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        {% else %}
        <div class="error-details">
            <h3>Error Details</h3>
            <p>{{ article.error }}</p>
        </div>
        {% endif %}
    </div>
    {% endfor %}
</body>
</html>
"""

def generate_html_report(results, output_dir="results"):  # Добавляем параметр output_dir
    """Генерирует HTML отчет с результатами сравнения"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    success_count = len([r for r in results if r['status'] == 'success'])
    error_count = len([r for r in results if r['status'] == 'error'])
    
    template = Template(HTML_TEMPLATE)
    html_content = template.render(
        timestamp=timestamp,
        total_articles=len(results),
        success_count=success_count,
        error_count=error_count,
        articles=results
    )
    
    # Создаем директорию, если ее нет
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n🌐 HTML отчет сохранен: {report_path}")
    return report_path