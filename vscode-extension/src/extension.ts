import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

let diagnosticCollection: vscode.DiagnosticCollection;

export function activate(context: vscode.ExtensionContext) {
    console.log('rec-praxis-rlm extension is now active');

    // Create diagnostic collection for inline warnings
    diagnosticCollection = vscode.languages.createDiagnosticCollection('rec-praxis-rlm');
    context.subscriptions.push(diagnosticCollection);

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('rec-praxis-rlm.reviewFile', reviewCurrentFile)
    );
    context.subscriptions.push(
        vscode.commands.registerCommand('rec-praxis-rlm.auditFile', auditCurrentFile)
    );
    context.subscriptions.push(
        vscode.commands.registerCommand('rec-praxis-rlm.scanDependencies', scanDependencies)
    );
    context.subscriptions.push(
        vscode.commands.registerCommand('rec-praxis-rlm.reviewWorkspace', reviewWorkspace)
    );

    // Auto-review on save if enabled
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument((document) => {
            const config = vscode.workspace.getConfiguration('rec-praxis-rlm');
            if (config.get('enableDiagnostics') && document.languageId === 'python') {
                reviewDocument(document);
            }
        })
    );
}

async function reviewCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }

    if (editor.document.languageId !== 'python') {
        vscode.window.showWarningMessage('This command only works on Python files');
        return;
    }

    await reviewDocument(editor.document);
}

async function reviewDocument(document: vscode.TextDocument) {
    const config = vscode.workspace.getConfiguration('rec-praxis-rlm');
    const pythonPath = config.get<string>('pythonPath', 'python');
    const severity = config.get<string>('codeReview.severity', 'HIGH');
    const memoryDir = config.get<string>('memoryDir', '.rec-praxis-rlm');

    vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Running code review...',
        cancellable: false
    }, async () => {
        try {
            const { stdout } = await execAsync(
                `${pythonPath} -m rec_praxis_rlm.cli rec-praxis-review ${document.fileName} --severity=${severity} --memory-dir=${memoryDir} --json`
            );

            const results = JSON.parse(stdout);
            const diagnostics: vscode.Diagnostic[] = [];

            for (const finding of results.findings) {
                const line = finding.line ? finding.line - 1 : 0;
                const range = new vscode.Range(line, 0, line, 999);

                const severityMap: { [key: string]: vscode.DiagnosticSeverity } = {
                    'CRITICAL': vscode.DiagnosticSeverity.Error,
                    'HIGH': vscode.DiagnosticSeverity.Error,
                    'MEDIUM': vscode.DiagnosticSeverity.Warning,
                    'LOW': vscode.DiagnosticSeverity.Information
                };

                const diagnostic = new vscode.Diagnostic(
                    range,
                    `${finding.title}: ${finding.description}`,
                    severityMap[finding.severity] || vscode.DiagnosticSeverity.Warning
                );
                diagnostic.source = 'rec-praxis-rlm';
                diagnostic.code = finding.severity;
                diagnostics.push(diagnostic);
            }

            diagnosticCollection.set(document.uri, diagnostics);

            if (results.total_findings > 0) {
                vscode.window.showInformationMessage(
                    `rec-praxis-rlm: Found ${results.total_findings} issue(s) (${results.blocking_findings} blocking)`
                );
            } else {
                vscode.window.showInformationMessage('rec-praxis-rlm: No issues found ✓');
            }
        } catch (error: any) {
            vscode.window.showErrorMessage(`Code review failed: ${error.message}`);
        }
    });
}

async function auditCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }

    if (editor.document.languageId !== 'python') {
        vscode.window.showWarningMessage('This command only works on Python files');
        return;
    }

    const config = vscode.workspace.getConfiguration('rec-praxis-rlm');
    const pythonPath = config.get<string>('pythonPath', 'python');
    const failOn = config.get<string>('securityAudit.failOn', 'CRITICAL');
    const memoryDir = config.get<string>('memoryDir', '.rec-praxis-rlm');

    vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Running security audit...',
        cancellable: false
    }, async () => {
        try {
            const { stdout } = await execAsync(
                `${pythonPath} -m rec_praxis_rlm.cli rec-praxis-audit ${editor.document.fileName} --fail-on=${failOn} --memory-dir=${memoryDir} --json`
            );

            const results = JSON.parse(stdout);
            const diagnostics: vscode.Diagnostic[] = [];

            for (const finding of results.findings) {
                const line = finding.line ? finding.line - 1 : 0;
                const range = new vscode.Range(line, 0, line, 999);

                const severityMap: { [key: string]: vscode.DiagnosticSeverity } = {
                    'CRITICAL': vscode.DiagnosticSeverity.Error,
                    'HIGH': vscode.DiagnosticSeverity.Error,
                    'MEDIUM': vscode.DiagnosticSeverity.Warning,
                    'LOW': vscode.DiagnosticSeverity.Information
                };

                const diagnostic = new vscode.Diagnostic(
                    range,
                    `[${finding.owasp || 'Security'}] ${finding.title}: ${finding.description}`,
                    severityMap[finding.severity] || vscode.DiagnosticSeverity.Error
                );
                diagnostic.source = 'rec-praxis-rlm-security';
                diagnostic.code = finding.cwe || finding.owasp || 'SECURITY';
                diagnostics.push(diagnostic);
            }

            diagnosticCollection.set(editor.document.uri, diagnostics);

            if (results.blocking_findings > 0) {
                vscode.window.showWarningMessage(
                    `rec-praxis-rlm: Found ${results.blocking_findings} CRITICAL security issue(s)!`
                );
            } else if (results.total_findings > 0) {
                vscode.window.showInformationMessage(
                    `rec-praxis-rlm: Found ${results.total_findings} security issue(s)`
                );
            } else {
                vscode.window.showInformationMessage('rec-praxis-rlm: No security issues found ✓');
            }
        } catch (error: any) {
            vscode.window.showErrorMessage(`Security audit failed: ${error.message}`);
        }
    });
}

async function scanDependencies() {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        vscode.window.showWarningMessage('No workspace folder open');
        return;
    }

    const config = vscode.workspace.getConfiguration('rec-praxis-rlm');
    const pythonPath = config.get<string>('pythonPath', 'python');
    const memoryDir = config.get<string>('memoryDir', '.rec-praxis-rlm');
    const workspaceRoot = workspaceFolders[0].uri.fsPath;

    vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Scanning dependencies and secrets...',
        cancellable: false
    }, async () => {
        try {
            const { stdout } = await execAsync(
                `cd ${workspaceRoot} && ${pythonPath} -m rec_praxis_rlm.cli rec-praxis-deps --memory-dir=${memoryDir} --json`
            );

            const results = JSON.parse(stdout);

            const message = `Dependencies: ${results.dependencies_scanned}, CVEs: ${results.cve_count}, Secrets: ${results.secret_count}`;

            if (results.blocking_findings > 0) {
                vscode.window.showWarningMessage(`rec-praxis-rlm: ${message} (${results.blocking_findings} CRITICAL!)`);
            } else if (results.total_findings > 0) {
                vscode.window.showInformationMessage(`rec-praxis-rlm: ${message}`);
            } else {
                vscode.window.showInformationMessage(`rec-praxis-rlm: ${message} - All clear ✓`);
            }

            // Show detailed results in output channel
            const outputChannel = vscode.window.createOutputChannel('rec-praxis-rlm Dependency Scan');
            outputChannel.clear();
            outputChannel.appendLine('=== Dependency & Secret Scan Results ===\n');
            outputChannel.appendLine(`Dependencies scanned: ${results.dependencies_scanned}`);
            outputChannel.appendLine(`Files scanned: ${results.files_scanned}`);
            outputChannel.appendLine(`CVE vulnerabilities: ${results.cve_count}`);
            outputChannel.appendLine(`Secrets found: ${results.secret_count}\n`);

            for (const finding of results.findings) {
                outputChannel.appendLine(`[${finding.severity}] ${finding.title}`);
                outputChannel.appendLine(`  Type: ${finding.type}`);
                outputChannel.appendLine(`  Remediation: ${finding.remediation}\n`);
            }

            outputChannel.show();
        } catch (error: any) {
            vscode.window.showErrorMessage(`Dependency scan failed: ${error.message}`);
        }
    });
}

async function reviewWorkspace() {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        vscode.window.showWarningMessage('No workspace folder open');
        return;
    }

    const config = vscode.workspace.getConfiguration('rec-praxis-rlm');
    const pythonPath = config.get<string>('pythonPath', 'python');
    const severity = config.get<string>('codeReview.severity', 'HIGH');
    const memoryDir = config.get<string>('memoryDir', '.rec-praxis-rlm');
    const workspaceRoot = workspaceFolders[0].uri.fsPath;

    vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Reviewing workspace...',
        cancellable: false
    }, async () => {
        try {
            const { stdout } = await execAsync(
                `cd ${workspaceRoot} && find . -name "*.py" -type f | xargs ${pythonPath} -m rec_praxis_rlm.cli rec-praxis-review --severity=${severity} --memory-dir=${memoryDir} --json`
            );

            const results = JSON.parse(stdout);

            vscode.window.showInformationMessage(
                `rec-praxis-rlm: Workspace review complete - ${results.total_findings} issue(s) found`
            );

            // Show summary in output channel
            const outputChannel = vscode.window.createOutputChannel('rec-praxis-rlm Workspace Review');
            outputChannel.clear();
            outputChannel.appendLine('=== Workspace Code Review Results ===\n');
            outputChannel.appendLine(`Total issues: ${results.total_findings}`);
            outputChannel.appendLine(`Blocking issues: ${results.blocking_findings}\n`);

            const fileGroups: { [key: string]: any[] } = {};
            for (const finding of results.findings) {
                if (!fileGroups[finding.file]) {
                    fileGroups[finding.file] = [];
                }
                fileGroups[finding.file].push(finding);
            }

            for (const [file, findings] of Object.entries(fileGroups)) {
                outputChannel.appendLine(`\n${file}:`);
                for (const finding of findings) {
                    outputChannel.appendLine(`  [${finding.severity}] Line ${finding.line}: ${finding.title}`);
                }
            }

            outputChannel.show();
        } catch (error: any) {
            vscode.window.showErrorMessage(`Workspace review failed: ${error.message}`);
        }
    });
}

export function deactivate() {
    if (diagnosticCollection) {
        diagnosticCollection.dispose();
    }
}
