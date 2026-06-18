import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('quarks');
    const serverPath = config.get<string>('serverPath', 'quarks-lsp');

    const serverOptions: ServerOptions = {
        run: {
            command: serverPath,
            transport: TransportKind.stdio
        },
        debug: {
            command: serverPath,
            transport: TransportKind.stdio
        }
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'quarks' }],
        synchronize: {
            // No workspace-level config to sync in MP5.
        }
    };

    client = new LanguageClient(
        'quarks',
        'Quarks Language Server',
        serverOptions,
        clientOptions
    );

    client.start().catch((err) => {
        vscode.window.showErrorMessage(
            `Failed to start quarks-lsp: ${err}. Check 'quarks.serverPath' setting.`
        );
    });

    context.subscriptions.push({
        dispose: () => {
            client?.stop();
        }
    });
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
