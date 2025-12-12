"""HTML report generation for security scan results."""

import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from rec_praxis_rlm import __version__
from rec_praxis_rlm.types import Finding, CVEFinding, SecretFinding, Severity, OWASPCategory


# Inline HTML template with Chart.js and Tailwind CSS via CDN
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>rec-praxis-rlm Security Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
        .severity-critical {{ background-color: #dc2626; color: white; }}
        .severity-high {{ background-color: #ea580c; color: white; }}
        .severity-medium {{ background-color: #f59e0b; color: white; }}
        .severity-low {{ background-color: #eab308; color: white; }}
        .severity-info {{ background-color: #3b82f6; color: white; }}
        @media print {{
            .no-print {{ display: none; }}
            .print-break {{ page-break-before: always; }}
        }}
    </style>
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <header class="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 shadow-lg">
        <div class="container mx-auto">
            <h1 class="text-3xl font-bold">🛡️ rec-praxis-rlm Security Report</h1>
            <p class="text-blue-100 mt-2">Generated on {timestamp}</p>
        </div>
    </header>

    <main class="container mx-auto px-4 py-8">
        <!-- Summary Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-gray-500 text-sm font-semibold uppercase">Total Findings</div>
                <div class="text-4xl font-bold text-gray-900 mt-2">{total_findings}</div>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-gray-500 text-sm font-semibold uppercase">Critical</div>
                <div class="text-4xl font-bold text-red-600 mt-2">{critical_count}</div>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-gray-500 text-sm font-semibold uppercase">High</div>
                <div class="text-4xl font-bold text-orange-600 mt-2">{high_count}</div>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-gray-500 text-sm font-semibold uppercase">Medium/Low</div>
                <div class="text-4xl font-bold text-yellow-600 mt-2">{medium_low_count}</div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <!-- Severity Distribution -->
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-xl font-bold text-gray-900 mb-4">Severity Distribution</h2>
                <canvas id="severityChart"></canvas>
            </div>

            <!-- OWASP Category Breakdown -->
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-xl font-bold text-gray-900 mb-4">OWASP Top 10 Breakdown</h2>
                <canvas id="owaspChart"></canvas>
            </div>
        </div>

        <!-- Findings Table -->
        <div class="bg-white rounded-lg shadow">
            <div class="p-6 border-b border-gray-200">
                <h2 class="text-xl font-bold text-gray-900">Detailed Findings</h2>
                <div class="mt-4 flex gap-2 no-print">
                    <button onclick="filterTable('all')" class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">All</button>
                    <button onclick="filterTable('CRITICAL')" class="px-4 py-2 severity-critical rounded hover:opacity-80">Critical</button>
                    <button onclick="filterTable('HIGH')" class="px-4 py-2 severity-high rounded hover:opacity-80">High</button>
                    <button onclick="filterTable('MEDIUM')" class="px-4 py-2 severity-medium rounded hover:opacity-80">Medium</button>
                    <button onclick="filterTable('LOW')" class="px-4 py-2 severity-low rounded hover:opacity-80">Low</button>
                </div>
            </div>
            <div class="overflow-x-auto">
                <table id="findingsTable" class="w-full">
                    <thead class="bg-gray-100 border-b border-gray-200">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Severity</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Title</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">File</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Line</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Category</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-200">
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- CVE Findings (if any) -->
        {cve_section}

        <!-- Footer -->
        <footer class="mt-12 text-center text-gray-500 text-sm">
            <p>Generated by <a href="https://github.com/jmanhype/rec-praxis-rlm" class="text-blue-600 hover:underline">rec-praxis-rlm</a> v{version}</p>
            <p class="mt-2 no-print">
                <button onclick="window.print()" class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                    🖨️ Print / Save as PDF
                </button>
            </p>
        </footer>
    </main>

    <script>
        // Severity Distribution Chart
        const severityCtx = document.getElementById('severityChart').getContext('2d');
        new Chart(severityCtx, {{
            type: 'doughnut',
            data: {{
                labels: {severity_labels},
                datasets: [{{
                    data: {severity_data},
                    backgroundColor: ['#dc2626', '#ea580c', '#f59e0b', '#eab308', '#3b82f6'],
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ position: 'bottom' }}
                }}
            }}
        }});

        // OWASP Category Chart
        const owaspCtx = document.getElementById('owaspChart').getContext('2d');
        new Chart(owaspCtx, {{
            type: 'bar',
            data: {{
                labels: {owasp_labels},
                datasets: [{{
                    label: 'Findings',
                    data: {owasp_data},
                    backgroundColor: '#6366f1',
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ beginAtZero: true, ticks: {{ stepSize: 1 }} }}
                }},
                plugins: {{
                    legend: {{ display: false }}
                }}
            }}
        }});

        // Table Filtering
        function filterTable(severity) {{
            const table = document.getElementById('findingsTable');
            const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');

            for (let row of rows) {{
                const severityCell = row.cells[0].textContent.trim();
                if (severity === 'all' || severityCell === severity) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }}
        }}
    </script>
</body>
</html>
"""


def generate_html_report(
    findings: List[Finding],
    output_path: str = "security-report.html",
    cve_findings: Optional[List[CVEFinding]] = None,
    secret_findings: Optional[List[SecretFinding]] = None
) -> str:
    """Generate an interactive HTML report from security findings.

    Args:
        findings: List of code review/security findings
        output_path: Path to save HTML file
        cve_findings: Optional list of CVE findings
        secret_findings: Optional list of secret findings

    Returns:
        Path to generated HTML file
    """
    # Count severity levels
    severity_counts = Counter(f.severity for f in findings)
    critical_count = severity_counts[Severity.CRITICAL]
    high_count = severity_counts[Severity.HIGH]
    medium_count = severity_counts[Severity.MEDIUM]
    low_count = severity_counts[Severity.LOW]
    info_count = severity_counts[Severity.INFO]

    # Count OWASP categories
    owasp_counts: Dict[str, int] = {}
    for f in findings:
        if f.owasp_category:
            category_name = f.owasp_category.value
            owasp_counts[category_name] = owasp_counts.get(category_name, 0) + 1

    # Sort OWASP categories by count (descending)
    sorted_owasp = sorted(owasp_counts.items(), key=lambda x: x[1], reverse=True)

    # Generate table rows
    table_rows = []
    for finding in sorted(findings, key=lambda f: (f.severity.value, f.file_path)):
        severity_class = f"severity-{finding.severity.name.lower()}"
        owasp_display = (
            html.escape(finding.owasp_category.value) if finding.owasp_category else "N/A"
        )
        cwe_display = f"CWE-{finding.cwe_id}" if finding.cwe_id else ""
        title = html.escape(finding.title)
        description = html.escape(finding.description)
        remediation = html.escape(finding.remediation)
        file_path = html.escape(finding.file_path)

        row = f"""
        <tr data-severity="{finding.severity.name}">
            <td class="px-6 py-4">
                <span class="px-3 py-1 text-xs font-semibold rounded {severity_class}">
                    {finding.severity.name}
                </span>
            </td>
            <td class="px-6 py-4">
                <div class="font-semibold text-gray-900">{title}</div>
                <div class="text-sm text-gray-600 mt-1">{description}</div>
                {f'<div class="text-xs text-gray-500 mt-1">{cwe_display}</div>' if cwe_display else ''}
                <details class="mt-2">
                    <summary class="text-sm text-blue-600 cursor-pointer hover:underline">Remediation</summary>
                    <div class="text-sm text-gray-700 mt-1 pl-4">{remediation}</div>
                </details>
            </td>
            <td class="px-6 py-4 text-sm font-mono text-gray-700">{file_path}</td>
            <td class="px-6 py-4 text-sm text-gray-600">{finding.line_number or 'N/A'}</td>
            <td class="px-6 py-4 text-sm text-gray-600">{owasp_display}</td>
        </tr>
        """
        table_rows.append(row)

    # CVE section (if applicable)
    cve_section = ""
    if cve_findings:
        cve_rows = []
        for cve in cve_findings:
            severity_class = f"severity-{cve.severity.name.lower()}"
            cve_id = html.escape(cve.cve_id)
            cve_desc = html.escape(cve.description)
            pkg_name = html.escape(cve.package_name)
            installed_ver = html.escape(cve.installed_version)
            fixed_ver = html.escape(cve.fixed_version) if cve.fixed_version else "N/A"
            cve_rows.append(f"""
            <tr>
                <td class="px-6 py-4">
                    <span class="px-3 py-1 text-xs font-semibold rounded {severity_class}">
                        {cve.severity.name}
                    </span>
                </td>
                <td class="px-6 py-4">
                    <div class="font-semibold text-gray-900">{cve_id}</div>
                    <div class="text-sm text-gray-600 mt-1">{cve_desc}</div>
                </td>
                <td class="px-6 py-4 text-sm font-mono text-gray-700">{pkg_name}</td>
                <td class="px-6 py-4 text-sm text-gray-600">{installed_ver}</td>
                <td class="px-6 py-4 text-sm text-green-600">{fixed_ver}</td>
            </tr>
            """)

        cve_section = f"""
        <div class="bg-white rounded-lg shadow mt-8 print-break">
            <div class="p-6 border-b border-gray-200">
                <h2 class="text-xl font-bold text-gray-900">CVE Vulnerabilities ({len(cve_findings)})</h2>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full">
                    <thead class="bg-gray-100 border-b border-gray-200">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Severity</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">CVE ID</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Package</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Installed</th>
                            <th class="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Fixed In</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-200">
                        {''.join(cve_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        """

    # Prepare chart data
    severity_labels = json.dumps(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'])
    severity_data = json.dumps([critical_count, high_count, medium_count, low_count, info_count])

    owasp_labels = json.dumps([item[0] for item in sorted_owasp[:10]])  # Top 10
    owasp_data = json.dumps([item[1] for item in sorted_owasp[:10]])

    # Format timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Render template
    html_content = HTML_TEMPLATE.format(
        timestamp=timestamp,
        version=__version__,
        total_findings=len(findings),
        critical_count=critical_count,
        high_count=high_count,
        medium_low_count=medium_count + low_count,
        severity_labels=severity_labels,
        severity_data=severity_data,
        owasp_labels=owasp_labels,
        owasp_data=owasp_data,
        table_rows=''.join(table_rows),
        cve_section=cve_section
    )

    # Write to file
    output_file = Path(output_path)
    output_file.write_text(html_content, encoding='utf-8')

    return str(output_file.absolute())
