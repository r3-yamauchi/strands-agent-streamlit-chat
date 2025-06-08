# Strands Agent Streamlit チャットアプリ

Strands フレームワークを使用し、OpenAI APIとAWS Bedrockの両方に対応したAIエージェントチャットアプリケーション。

![](docs/image01.png)

## 概要

このアプリケーションは、複数のLLMプロバイダー（OpenAI、AWS Bedrock）を通じて様々な大規模言語モデル（LLM）と対話するためのチャットインターフェースを提供します。以下の機能をサポート：

- **マルチプロバイダー対応**: OpenAI（GPT-4o、GPT-4.1等）とAWS Bedrock（Claude、Nova）の両方をサポート
- **画像入力と処理**: 画像入力をサポートするモデルでの画像アップロードと処理
- **MCP（Model Context Protocol）によるツール利用**: 外部ツールとの統合
- **チャット履歴管理**: 会話の保存と読み込み
- **プロンプトキャッシング**: パフォーマンス向上のための設定可能なキャッシング（Bedrockのみ）

## 機能

- **複数モデルサポート**: 
  - OpenAI: GPT-4o、GPT-4o Mini、GPT-4.1
  - AWS Bedrock: Amazon Nova、Anthropic Claudeモデル
- **画像処理**: 画像入力をサポートするモデルでの画像アップロードと処理
- **ツール統合**: MCP（Model Context Protocol）による外部ツールの使用
- **チャット履歴**: 過去の会話の保存と読み込み
- **プロバイダー別最適化**: 各プロバイダーの特性に応じた機能提供

## 前提条件

- Python 3.12以上
- OpenAI APIキー（OpenAIモデル使用時）
- AWSアカウントとBedrockアクセス権限（Bedrockモデル使用時）
- Docker（一部MCPツール用）

## インストール

1. リポジトリのクローン:
   ```bash
   git clone https://github.com/moritalous/strands-agent-streamlit-chat.git
   cd strands-agent-streamlit-chat
   ```

2. 依存関係のインストール:
   ```bash
   pip install -e .
   ```
   
   またはuvを使用:
   ```bash
   uv pip install -e .
   ```

3. 環境設定:
   ```bash
   cp .env.example .env
   # .envファイルを編集してOpenAI APIキーを設定
   ```

## 設定

### 環境変数（`.env`ファイル）

```bash
# OpenAI API設定
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# デフォルトプロバイダー（openai または bedrock）
DEFAULT_PROVIDER=openai

# AWS設定（Bedrock使用時）
# AWS_PROFILE=default
# AWS_REGION=ap-northeast-1

# 開発モード
DEV=false
```

### 1. `config/config.json`

プロバイダーとモデルの設定:

```json
{
    "chat_history_dir": "chat_history",
    "mcp_config_file": "config/mcp.json",
    "providers": {
        "openai": {
            "models": {
                "gpt-4o": {
                    "display_name": "GPT-4o (最新・画像対応)",
                    "image_support": true,
                    "max_tokens": 16384,
                    "cost_per_1k_tokens": {
                        "input": 0.005,
                        "output": 0.015
                    }
                },
                "gpt-4.1": {
                    "display_name": "GPT-4.1 (2025年4月リリース)",
                    "image_support": true,
                    "max_tokens": 16384,
                    "cost_per_1k_tokens": {
                        "input": 0.01,
                        "output": 0.03
                    }
                }
            }
        },
        "bedrock": {
            "region": "ap-northeast-1",
            "models": {
                "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {
                    "display_name": "Claude 3.5 Sonnet",
                    "cache_support": [],
                    "image_support": true
                }
            }
        }
    }
}
```

### 2. `config/mcp.json`

MCP（Model Context Protocol）サーバーの設定:

```json
{
    "mcpServers": {
        "awsdoc": {
            "command": "uvx",
            "args": [
                "awslabs.aws-documentation-mcp-server@latest"
            ],
            "env": {
                "FASTMCP_LOG_LEVEL": "ERROR"
            }
        },
        "sequentialthinking": {
            "command": "docker",
            "args": [
                "run",
                "--rm",
                "-i",
                "mcp/sequentialthinking"
            ]
        }
    }
}
```

## 使用方法

1. Streamlitアプリケーションの起動:
   ```bash
   streamlit run app.py
   ```

2. Webインターフェースでの操作:
   - サイドバーでプロバイダー（OpenAI/Amazon Bedrock）を選択
   - ドロップダウンからモデルを選択
   - OpenAIの場合はTemperatureを調整可能
   - Bedrockの場合はプロンプトキャッシングの有効/無効を設定
   - 使用するMCPツールを選択
   - 新しいチャットを開始または既存のチャットを継続
   - メッセージを入力し、必要に応じて画像をアップロード
   - チャットインターフェースでツールの使用状況を確認

## チャット履歴

チャット履歴は設定されたディレクトリにYAMLファイルとして保存されます：
- 「新規チャット」ボタンで新しいチャットを開始
- サイドバーのファイル名をクリックして過去のチャットを読み込み

## サポートモデル

### OpenAI モデル
- **GPT-4o**: 最新の統合モデル、画像入力対応
- **GPT-4o Mini**: 高速・低コスト版、画像入力対応
- **GPT-4.1**: 2025年4月リリースの新モデル、画像入力対応

### AWS Bedrock モデル
- **Claude Sonnet 4**: 最新版、プロンプトキャッシング対応
- **Claude 3.7 Sonnet**: プロンプトキャッシング対応
- **Claude 3.5 Sonnet**: スタンダード版
- **Claude 3.5 Haiku**: 高速版
- **Amazon Nova Premier/Pro/Lite/Micro**: Amazon独自モデル

## MCPツール

アプリケーションは様々なMCPツールをサポート：
- AWSドキュメントツール
- 順次思考ツール
- MCP設定を通じて追加ツールの統合が可能

## 開発

使用技術：
- **Streamlit**: Webインターフェース
- **Strands Framework v0.1.6+**: エージェント機能
- **OpenAI API**: GPTモデルアクセス
- **AWS Bedrock**: Claudeモデルアクセス
- **MCP**: ツール統合

## トラブルシューティング

### OpenAI使用時のエラー
- `OPENAI_API_KEY`が正しく設定されているか確認
- APIキーの有効性を確認
- モデルのmax_tokens制限に注意

### Bedrock使用時のエラー
- AWS認証情報が正しく設定されているか確認
- Bedrockへのアクセス権限を確認
- リージョン設定が正しいか確認

## ライセンス

このプロジェクトはApache License 2.0でライセンスされています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 謝辞

- Strands Agents フレームワーク
- OpenAI
- AWS Bedrock
- Streamlit
- MCP (Model Context Protocol)